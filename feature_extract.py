import itertools
import os

import cv2
import numpy as np
import math

import matplotlib.pyplot as plt


def cal_hist(array):
    """
    用于print直方图具体信息
    :param array: 二维矩阵
    :return: hist与bins
    """
    array = array.astype(int)
    hist, bins = np.histogram(array, 72, range=(-180 / 5, 180 / 5))

    # hist, bins = np.histogram(array, 72, range = (-180, 180))
    # print(hist[36], hist[37])
    # print('\nhistogram:')
    # print('------------------------')
    # for i in range(len(bins) - 1):
    #     if hist[i] != 0:
    #         print(bins[i], '(', bins[i] * 5, '°)', ':', hist[i])
    # print('------------------------')

    return hist, bins


def hsv_processing(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_img_height = hsv_img.shape[0]
    hsv_img_width = hsv_img.shape[1]

    # HSV range H,[0,360); S,[0,1); V,[0,1)
    # opencv HSV range h,[0,180); s,[0,255); v,[0,255)

    h_quantized = np.zeros((hsv_img_height, hsv_img_width), dtype=np.uint8)
    s_quantized = np.zeros((hsv_img_height, hsv_img_width), dtype=np.uint8)
    v_quantized = np.zeros((hsv_img_height, hsv_img_width), dtype=np.uint8)

    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]

    h = 2 * h
    h_quantized[(h > 315) | (h <= 200)] = 0
    h_quantized[(h > 20) & (h <= 40)] = 1
    h_quantized[(h > 40) & (h <= 75)] = 2
    h_quantized[(h > 75) & (h <= 155)] = 3
    h_quantized[(h > 155) & (h <= 190)] = 4
    h_quantized[(h > 190) & (h <= 270)] = 5
    h_quantized[(h > 270) & (h <= 295)] = 6
    h_quantized[(h > 295) & (h <= 315)] = 7

    # 255*0.2 =51; 255*0.7=178
    s_quantized[(s <= 51)] = 0
    s_quantized[(s > 51) & (s <= 178)] = 1
    s_quantized[(s > 178)] = 2

    v_quantized[(v <= 51)] = 0
    v_quantized[(v > 51) & (v <= 178)] = 1
    v_quantized[(v > 178)] = 2

    final_score = 9 * h_quantized + 3 * s_quantized + v_quantized
    hist = cv2.calcHist([final_score], [0], None, [72], [0, 71]) / (hsv_img_height * hsv_img_width)
    # this hist is normalized
    hist_array = np.array(hist).flatten().tolist()
    return hist_array


def zeros_location(arr1, arr2):
    shape = arr1.shape
    tmpx_0, tmpy_0 = np.where(arr1 == 0)  # 获得图像中垂直方向梯度非零坐标
    tmpx_1, tmpy_1 = np.where(arr2 == 0)  # 获得图像中水平方向梯度非零坐标
    # union = np.union1d(tmpx_0, tmp_1)  # 获得图像中两个方向梯度都不为0的坐标并集

    zeros_location = np.zeros((shape[0], shape[1]))
    zeros_location[tmpx_0, tmpy_0] = zeros_location[tmpx_0, tmpy_0] + 1
    zeros_location[tmpx_1, tmpy_1] = zeros_location[tmpx_1, tmpy_1] + 1
    tmpx_final, tmpy_final = np.where(zeros_location == 2)

    return tmpx_final, tmpy_final


def edge_dir(img):
    shape = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img_gray, (3, 3), 0)
    canny = cv2.Canny(img1, 50, 100)

    gradient_0 = np.gradient(canny, axis=0)  # gradient0 方向向下
    gradient_1 = np.gradient(canny, axis=1)  # gradient1 方向向右

    tmpx_final, tmpy_final = zeros_location(gradient_0, gradient_1)

    gradient_dir = np.arctan2(gradient_1, gradient_0)
    gradient_dir[tmpx_final, tmpy_final] = None
    gradient_dir = gradient_dir / math.pi * 180
    gradient_dir = np.floor(gradient_dir / 5)
    # print(gradient_dir)
    gradient_dir = gradient_dir.astype(int)

    hist, bins = cal_hist(gradient_dir)  # print histogram information

    n_p = shape[0] * shape[1]  # total pixels number
    n_e = sum(hist)  # edge pixels number
    cnt = n_p - n_e  # the number of pixels that didn’t contribute to an edge

    # print('total pixels number\edge pixels number\\not edge pixels number:', n_p, n_e, cnt)

    hist = hist / n_e
    hist = np.append(hist, cnt / n_p)

    return hist


def cal_connection_areas(input_arr):
    num_objects, labels = cv2.connectedComponents(input_arr, connectivity=8)
    # print(num_objects)

    areas = np.zeros(num_objects, dtype=int)

    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            index_val = labels[row, col]
            if index_val != 0:
                areas[index_val] += 1

    # threshold = 0.001 * input_arr.shape[0] * input_arr.shape[1]
    threshold = 10  # 0.1%太大了，没有那么大的相干面积
    # print(threshold)

    vector = [0, 0]  # 第一个代表相干edge pixels总数， 第二个代表不相干edge pixels总数
    for area in areas:
        if area >= threshold:
            vector[0] += area
            # print("area", area)
        else:
            vector[1] += area
            # print(area)

    # cv2.waitKey(100)

    return vector


def coherence_vector(img):
    shape = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img_gray, (3, 3), 0)
    canny = cv2.Canny(img1, 50, 100)

    gradient_0 = np.gradient(canny, axis=0)  # gradient0 方向向下
    gradient_1 = np.gradient(canny, axis=1)  # gradient1 方向向右

    tmpx_final, tmpy_final = zeros_location(gradient_0, gradient_1)

    gradient_dir = np.arctan2(gradient_1, gradient_0)
    gradient_dir[tmpx_final, tmpy_final] = -37
    gradient_dir = gradient_dir / math.pi * 180
    gradient_dir = np.floor(gradient_dir / 5)
    # print(gradient_dir)
    gradient_dir = gradient_dir.astype(int)

    # plt.figure()
    # plt.imshow(gradient_dir, cmap=plt.cm.gray)

    vector_list = [(0, 0)] * 72  # 第一个代表相干edge pixels总数， 第二个代表不相干edge pixels总数
    # print()
    # print(len(vector_list))

    for i in range(-36, 36):
        tmpx_dir, tmpy_dir = np.where(gradient_dir == i)  # 选取第i个bin中所有的像素
        total_dir_pixel = len(tmpx_dir)
        if total_dir_pixel > 0:
            tmp_img = np.zeros((shape[0], shape[1]), 'uint8')
            tmp_img[tmpx_dir, tmpy_dir] = 255
            # print()
            # print("i:", i)

            vec = cal_connection_areas(tmp_img)

            vector_list[int(i + 36)] = vec

            # cv2.drawContours(tmp_img, contours, -1, (0, 0, 255), 2)
            # cv2.imshow("Origin", tmp_img)
            # cv2.waitKey(0)
            #
            # plt.show()

    return vector_list


def coherence_vector_plot(img):
    shape = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img_gray, (3, 3), 0)
    canny = cv2.Canny(img1, 50, 100)

    gradient_0 = np.gradient(canny, axis=0)  # gradient0 方向向下
    gradient_1 = np.gradient(canny, axis=1)  # gradient1 方向向右

    tmpx_final, tmpy_final = zeros_location(gradient_0, gradient_1)

    gradient_dir = np.arctan2(gradient_1, gradient_0)
    gradient_dir[tmpx_final, tmpy_final] = -37
    gradient_dir = gradient_dir / math.pi * 180
    gradient_dir = np.floor(gradient_dir / 5)
    # print(gradient_dir)
    gradient_dir = gradient_dir.astype(int)

    plt.figure()
    plt.imshow(gradient_dir, cmap=plt.cm.gray)

    vector_list = [(0, 0)] * 72  # 第一个代表相干edge pixels总数， 第二个代表不相干edge pixels总数
    print()
    print(len(vector_list))

    for i in range(-36, 36):
        tmpx_dir, tmpy_dir = np.where(gradient_dir == i)  # 选取第i个bin中所有的像素
        total_dir_pixel = len(tmpx_dir)
        if total_dir_pixel > 0:
            tmp_img = np.zeros((shape[0], shape[1]), 'uint8')
            tmp_img[tmpx_dir, tmpy_dir] = 255

            num_objects, labels = cv2.connectedComponents(tmp_img, connectivity=8)

            areas = np.zeros(num_objects, dtype=int)

            for row in range(labels.shape[0]):
                for col in range(labels.shape[1]):
                    index_val = labels[row, col]
                    if index_val != 0:
                        areas[index_val] += 1

            plt.figure()
            plt.imshow(labels, cmap='plasma')
            plt.show()

            # threshold = 0.001 * input_arr.shape[0] * input_arr.shape[1]
            threshold = 10  # 0.1%太大了，没有那么大的相干面积
            # print(threshold)

            vector = [0, 0]  # 第一个代表相干edge pixels总数， 第二个代表不相干edge pixels总数
            for area in areas:
                if area >= threshold:
                    vector[0] += area
                    # print("area", area)
                else:
                    vector[1] += area
                    # print(area)

            vector_list[int(i + 36)] = vector

            # cv2.drawContours(tmp_img, contours, -1, (0, 0, 255), 2)
            # cv2.imshow("Origin", tmp_img)
            # cv2.waitKey(0)
            #
            # plt.show()

    return vector_list


def histogram_plot(path):
    path = "natural_test"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    hist_5 = []
    cnt = 0

    for file in files:
        print(path + "/" + file)
        original_img = cv2.imread(path + "/" + file)
        # cv2.imshow("1", original_img)
        # cv2.waitKey(0)
        hist_5.append(hsv_processing(original_img))  # Edge direction histograms
        cnt += 1

    hist = np.zeros(73, dtype='float64')

    print(hist_5)

    for i in range(len(hist_5)):
        contain_nan = (True in np.isnan(hist_5[i]))
        if contain_nan:
            print(i)
        else:
            hist = hist + hist_5[i]

    hist = hist  # 归一化

    print()
    print("cnt:", cnt)
    print(hist)

    plt.figure()
    Y = np.arange(-180, 180, 5)
    plt.bar(Y, hist[0:-1], 1)
    plt.xlabel("angle")
    plt.ylabel("histogram")

    # vector_list = coherence_vector(original_img)  # Edge direction coherence vector
    # print(vector_list)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("manmade_test/sun_afjpfomkltxlgdqw.jpg")
    # coherence_vector_plot(img)
    hist = hsv_processing(img)
    hist = np.array(hist).flatten().tolist()
    print(hist)
    # plt.plot(hist)
    # plt.show()
    plt.figure()
    Y = np.arange(0, 72)
    plt.bar(Y, hist, 1)
    plt.xlabel("color segment")
    plt.ylabel("histogram")
    plt.show()

    cv2.imshow("1", img)

    cv2.waitKey(0)
