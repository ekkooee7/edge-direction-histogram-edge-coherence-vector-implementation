import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def cal_connection_areas(input_arr):
    num_objects, labels = cv2.connectedComponents(input_arr)

    areas = np.zeros(num_objects, dtype=int)

    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            index_val = labels[row, col]
            if index_val != 0:
                areas[index_val] += 1

    threshold = 0.001 * input_arr.shape[0] * input_arr.shape[1]

    vector = [0, 0]  # 第一个代表相干edge pixels总数， 第二个代表不相干edge pixels总数
    for area in areas:
        if area >= threshold:
            vector[0] += area
        else:
            vector[1] += area

    return vector


if __name__ == '__main__':
    test_img = np.zeros((200, 200, 3), 'uint8')
    # cv2.rectangle(test_img, (10, 10), (20, 30), (255, 0, 0), 12)
    # cv2.rectangle(test_img, (110, 110), (120, 130), (255, 0, 0), 6)
    test_img[10:20, 10:20, :] = 255
    test_img[110:111, 110:120, :] = 255
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    plt.figure()
    plt.imshow(img)
    plt.show()

    cal_connection_areas(img)

    plt.show()
