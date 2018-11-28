from PIL import Image
import numpy as np
import pandas as pd
import glob


def images_to_csv(dir_path=None, name_prefix='images', verbose=True):
    """
    将路径下的所有图片的像素信息和标签写入csv文件

    :param dir_path: 图片所在路径
    :param name_prefix: 生成csv文件名的前缀
    :param verbose: 日志显示 False 不显示; True 显示; 默认 True
    :return:
    """

    # 获取图像路径
    img_paths = glob.glob(dir_path + '\\*')
    # 确定csv文件路径
    csv_path = '.\\index\\'+name_prefix+'_index.csv'
    # 用于标记当前图像
    n = 0
    # 总的图像数目
    images_num = len(img_paths)
    if images_num == 0:
        print('WARNING: There is no files of the path of \'{}\', '
              'please make sure that the path is correct' .format(dir_path))
        return
    # 初始化DataFrame
    data = pd.DataFrame(columns=np.append(np.arange(400), 'tag'))
    print('\n------{} images to be indexed------\n' .format(images_num))
    for img_path in img_paths:

        # 读取图像 并转换为灰度图
        img = Image.open(img_path).convert('L')
        img = np.array(img).flatten()
        # 增加tag
        tag = img_path.split('\\')[2]
        img = np.append(img, tag)
        data.loc[img_path[3:]] = img
        # 输出索引化进程
        if verbose:
            n += 1
            if n % 100 == 0:
                print('[{}/{}] images indexed'.format(n, images_num))
    data.to_csv(csv_path)
    print('----------DONE----------')
    print('{} images all indexed' .format(images_num))
    print('----------DONE----------')


if __name__ == '__main__':
    images_to_csv(dir_path='..\\charsAll\\*', name_prefix='chars_all')