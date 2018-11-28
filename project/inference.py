# -*- coding: utf-8 -*-
"""
--------------------------------------------------
    @File    : inference.py
    @Author  : 丁建栋
    @Email   : jiandongding@qq.com
--------------------------------------------------
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
import json
import glob
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def get_tag(tag_index=None):
    """
    按tag_index返回tag

    :param tag_index: int. tag_index
    :return: str. tag
    """
    new_dict = dict()
    with open('./chars_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        new_dict = {v: k for k, v in config.items()}
    if len(new_dict[tag_index]) == 1:
        return new_dict[tag_index]
    else:
        with open('./chars_chinese_map.json', 'r', encoding='utf-8') as f:
            chinese_map = json.load(f)
            return chinese_map[new_dict[tag_index]]


def recognize_char(path=None, model=None):
    """
    识别path下的图片的字符并输出图片和识别结果

    :param path: str. 图片路径
    :param model: keras.model. 训练完成的模型
    :return:
    """
    img = Image.open(path).convert('L')
    x = np.array(img).reshape(-1, 20, 20, 1)
    x = x.astype('float32') / 255
    index_one_hot = model.predict(x)
    tag_index = np.argmax(index_one_hot)
    char = get_tag(tag_index=tag_index)
    plt.xticks([]), plt.yticks([])
    plt.title('Result: '+char)
    plt.imshow(img, cmap='gray')
    plt.show()


def recognize_chars(dir_path=None, model=None):
    """
    识别dir_path所有图片的字符并输出图片和识别结果
    :param dir_path: str. 图片目录路径
    :param model: keras.model. 训练完成的模型
    :return:
    """
    paths = glob.glob(dir_path+'\\*')
    col_num = 10
    if len(paths) <= col_num:
        col_num = len(paths)
        row_num = 1
    else:
        row_num = len(paths) // col_num + 1
    c = 1
    plt.figure(figsize=(16, 2*row_num))
    for path in paths:
        img = Image.open(path).convert('L')
        x = np.array(img).reshape(-1, 20, 20, 1)
        x = x.astype('float32') / 255
        index_one_hot = model.predict(x)
        tag_index = np.argmax(index_one_hot)
        char = get_tag(tag_index=tag_index)

        plt.subplot(row_num, col_num, c)
        plt.xticks([]), plt.yticks([])
        plt.title('Result: '+char)
        c += 1
        plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    loaded_model = keras.models.load_model('./save_models/chars_trained_model.h5')
    recognize_char(path='.\\test_images\\test_lu.jpg', model=loaded_model)
    recognize_chars(dir_path='.\\test_images\\original', model=loaded_model)
    recognize_chars(dir_path='.\\test_images\\generated', model=loaded_model)
