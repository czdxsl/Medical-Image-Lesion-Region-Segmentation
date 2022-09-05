from keras.models import load_model
import tensorflow as tf
import cv2
import os
from tensorflow.keras.preprocessing import image
import numpy as np

with tf.device('/gpu:0'):
    model = load_model("./CAImodel.h5")
    print("模型加载成功")

    for i in range(25, 26):
        # 读取原图片
        raw_img = cv2.imread("./data/image/{0}-1.bmp".format(i))
        # print(model.summary())
        test_path = "./data/cut/test/{0}/".format(i)  # 测试集目录
        test_files = os.listdir(test_path)
        # 下面的代码是准备测试集的数据，与上面的内容完全相同
        x_test = np.empty((len(test_files), 63, 63, 3))
        # y_test = np.empty((len(test_files), 2))
        position_test = np.empty((len(test_files), 2))
        test_count = 0
        # test_num = 0
        for img_name in test_files:
            row = img_name.split('-')[1]
            col = img_name.split('-')[2].split('.')[0]
            # lable = img_name.split('-')[3].split('.')[0]
            position_test[test_count] = np.array((row, col))
            img_path = test_path + img_name
            img = image.load_img(img_path, target_size=(63, 63))
            img = image.img_to_array(img) / 255.0
            x_test[test_count] = img
            # if lable == "0":
            #     y_test[test_count] = np.array((0, 1))
            # else:
            #     y_test[test_count] = np.array((1, 0))
            #     test_num += 1
            test_count += 1
        # print("测试集中正样本的数量为：")
        # print(test_num)
        # 打乱测试集中的数据
        # index = [i for i in range(len(x_test))]
        # random.shuffle(index)
        # x_test = x_test[index]
        # # y_test = y_test[index]
        # position_test = position_test[index]

        predict_test = model.predict_classes(x_test).astype('int')
        print(predict_test)
        print(type(predict_test))
        count = 0
        num = 0
        for j in range(predict_test.shape[0]):
            if predict_test[j] == 0:
                num += 1
                [row, col] = position_test[count]
                print(row, col)
                raw_img[int(row), int(col)] = [0, 0, 255]
            count += 1
        print("预测正样本的数量为：")
        print(num)
        cv2.imwrite("./data/cut/result/pic{0}_result.bmp".format(i), raw_img)
# test_loss, test_acc = model.evaluate(x_test, y_test)
