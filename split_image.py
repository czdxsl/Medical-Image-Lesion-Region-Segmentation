import cv2
import random


def spilt_train_image():
    # 正样本数量
    positive_sample_count = 0
    # 负样本数量
    negative_sample_count = 0
    for img_num in range(1, 25):
        # 读取原图片
        raw_img = cv2.imread("./data/image/{0}-2.bmp".format(img_num))
        # 读取填充颜色之后的图片
        img_fill = cv2.imread("./data/image/{0}-3.bmp".format(img_num))

        # print(img_fill.shape)  # 打印图像的高，宽，通道数（返回一个3元素的tuple）
        height = img_fill.shape[0]  # 将tuple中的元素取出，赋值给height，width，channels
        width = img_fill.shape[1]
        channels = img_fill.shape[2]

        # 遍历每个像素点
        for row in range(height):  # 遍历每一行
            for col in range(width):  # 遍历每一列
                if row > 32 and col > 32:
                    # 当像素点为红色时，以该像素为中心从原始图片上分割下一张固定大小的小图片(63X63)
                    if (img_fill[row, col] == [0, 0, 255]).all():  # 图像像素是按B,G,R顺序存储的
                        red_cropped = raw_img[row - 31:row + 32, col - 31:col + 32]
                        # if random.randint(1, 10) > 6:
                        cv2.imwrite("./data/cut/train/img{0}-{1}-{2}-1.bmp".format(img_num, row, col),
                                    red_cropped)  # 保存图片，正样本后缀为1
                        positive_sample_count += 1
                    # 当像素点为蓝色时，以该像素为中心从原始图片上分割下一张固定大小的小图片(63X63)
                    if (img_fill[row, col] == [255, 0, 0]).all():
                        if random.randint(1, 10) > 8:
                            blue_cropped = raw_img[row - 31:row + 32, col - 31:col + 32]
                            cv2.imwrite("./data/cut/train/img{0}-{1}-{2}-0.bmp".format(img_num, row, col),
                                        blue_cropped)  # 保存图片，负样本后缀为0
                            negative_sample_count += 1
    print("正样本数量为：")
    print(positive_sample_count)
    print("负样本数量为：")
    print(negative_sample_count)


def split_test_image():
    for i in range(1, 25):
        raw_img = cv2.imread("./data/image/{0}-2.bmp".format(i))
        height = raw_img.shape[0]  # 将tuple中的元素取出，赋值给height，width，channels
        width = raw_img.shape[1]
        for row in range(height):  # 遍历每一行
            for col in range(width):  # 遍历每一列
                if row > 32 and col > 32:
                    if (row % 4 == 0) and (col % 4 == 0):
                        test_cropped = raw_img[row - 31:row + 32, col - 31:col + 32]
                        cv2.imwrite("./data/cut/test/{0}/img{1}-{2}-{3}.bmp".format(i, i, row, col),
                                    test_cropped)


# split_test_image()

spilt_train_image()
