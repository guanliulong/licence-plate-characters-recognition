# -*- coding: utf-8 -*-
"""
--------------------------------------------------
    @File    : train.py
    @Author  : 丁建栋
    @Email   : jiandongding@qq.com
--------------------------------------------------
"""
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
import pandas as pd
import json
import os


def data_split(path=None, test_size=0.2, num_classes=65):
    """
    将path下的csv文件里的字符数据划分为train data 和test data

    :param path: csv file path
    :param test_size: test data percentage
    :param num_classes: classes number. default 67
    :return: (x_train, y_train), (x_test, y_test)
    """

    # read csv data
    data = pd.read_csv(path)
    # get feature data
    x_data = data.iloc[:, 1:401].values.reshape(-1, 20, 20, 1)
    # get tag data
    y_data = data['tag'].astype('str').values.reshape(-1, 1)

    # 将tag按配置文件转换为tag_index
    with open('./chars_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        for y in y_data:
            y[0] = config[y[0]]

    # split data to train and test data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)
    # set y as one-hot type
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)
    # return train and test data
    return (x_train, y_train), (x_test, y_test)


# 训练模型
def build_model(x_train, y_train, x_test, y_test, batch_size= 32, epochs=1):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3),))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test),
          shuffle= True)
    return model


def save_model(model=None, save_dir=None, mode_name=None):
    """
    保存模型至本地
    :param model: keras.model. 训练完成的模型
    :param save_dir: str. 保存目录
    :param mode_name: str. 模型文件名称
    :return:
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save('./save_models/'+model_name)
    print('model saved at', model_path)


if __name__ == '__main__':
    num_classes = 65
    batch_size = 32
    epochs = 100
    save_dir = os.path.join(os.getcwd(), 'save_models')
    model_name = 'chars_trained_model.h5'

    # init train & test data
    csv_path = './index/chars_all_index.csv'
    train_data, test_data = data_split(path=csv_path, num_classes=num_classes)
    x_train, y_train = train_data
    x_test, y_test = test_data

    print('The number of train samples:', len(x_train))
    print('The number of test samples:', len(x_test))

    # 标准化数据
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # 训练模型
    model = build_model(x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=epochs)

    # 保存模型
    save_model(model=model, save_dir=save_dir, mode_name=model_name)

    # evaluate test data
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    # should get accuracy around 99.6%
