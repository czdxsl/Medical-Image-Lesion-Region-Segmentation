import cv2
import os
import random
import numpy as np
from tensorflow.keras.preprocessing import image


def load_dataset():
    train_path = "./data/cut/train/"  # 训练集目录
    train_files = os.listdir(train_path)  # 得到文件夹下所有的文件名称 Carotid artery intima 颈动脉内膜

    test_path = "./data/cut/test/"  # 测试集目录
    test_files = os.listdir(test_path)

    # x_train 对象用来存放训练集的图片。每张图片都需要转换成 numpy 向量形式
    x_train = np.empty((len(train_files), 63, 63, 3))
    y_train = np.empty((len(train_files), 2))

    train_count = 0
    # 遍历训练集下所有的图片
    for img_name in train_files:
        # 得到图片的路径
        img_path = train_path + img_name
        # 通过 image.load_img() 函数读取对应的图片，并转换成目标大小
        #  image 是 tensorflow.keras.preprocessing 中的一个对象
        img = image.load_img(img_path, target_size=(63, 63))
        # dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        # 将图片转换成 numpy 数组，并除以 255 ，归一化
        # 转换之后 img 的 shape 是 （63，63，3）
        img = image.img_to_array(img) / 255.0

        # 将处理好的图片装进定义好的 X_train 对象中
        x_train[train_count] = img
        # 将对应的标签装进 Y_train 对象中，因为是二分类，且文件名中带有标签，所以标签设为[1,0]或[0,1]

        lable = img_name.split('-')[3].split('.')[0]
        if lable == "0":
            y_train[train_count] = np.array((0, 1))
        else:
            y_train[train_count] = np.array((1, 0))
        train_count += 1

    # 下面的代码是准备测试集的数据，与上面的内容完全相同
    x_test = np.empty((len(test_files), 63, 63, 3))
    y_test = np.empty((len(test_files), 2))
    # test_count = 0
    # positive_sample_count = 0
    # negative_sample_count = 0
    # for img_name in test_files:
    #     img_path = test_path + img_name
    #     img = image.load_img(img_path, target_size=(63, 63))
    #     img = image.img_to_array(img) / 255.0
    #     x_test[test_count] = img
    #     lable = img_name.split('-')[3].split('.')[0]
    #     if lable == "0":
    #         y_test[test_count] = np.array((0, 1))
    #         negative_sample_count += 1
    #     else:
    #         y_test[test_count] = np.array((1, 0))
    #         positive_sample_count += 1
    #     test_count += 1
    # print("正样本数量为：")
    # print(positive_sample_count)
    # print("负样本数量为：")
    # print(negative_sample_count)
    # 打乱训练集中的数据
    index = [i for i in range(len(x_train))]
    random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    # # 打乱测试集中的数据
    # index = [i for i in range(len(x_test))]
    # random.shuffle(index)
    # x_test = x_test[index]
    # y_test = y_test[index]

    return x_train, y_train, x_test, y_test

