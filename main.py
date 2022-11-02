import cv2
import numpy as np


def chi_square_dis(his1, his2):
    dis = 0.5 * np.sum(np.square(his1-his2)/(his1+his2))
    return dis


if __name__ == '__main__':
    img1 = cv2.imread('mosaic_target1.jpg')
    img2 = cv2.imread('mosaic_target2.jpg')

    print(img1.shape)

    cv2.imshow("a", img1)
    cv2.imshow("b", img2)

    cv2.waitKey(0)



