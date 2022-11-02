import operator
import time

import cv2
import numpy as np

from feature_extract import edge_dir, coherence_vector

X_test = []
Y_train = []
Y_test = []

XX_train = np.loadtxt(open("XX_train_1000.csv", "rb"), delimiter=",", skiprows=0)

for i in range(500):
    Y_train.append("manmade")

for i in range(500):
    Y_train.append("natural")

for i in range(250):
    Y_test.append("manmade")

for i in range(250):
    Y_test.append("natural")

with open("manmade_test.txt", "r") as f:
    manmadetest = f.read().splitlines()

for i in range(250):
    X_test.append(manmadetest[i])

with open("natural_test.txt", "r") as f:
    naturaltest = f.read().splitlines()

for i in range(250):
    X_test.append(naturaltest[i])

XX_test = []
for i in X_test:
    # 读取图像
    # print i
    image = cv2.imread(i)

    # 图像像素大小一致
    img = cv2.resize(image, (400, 400),
                     interpolation=cv2.INTER_CUBIC)

    # 计算图像直方图并存储至X数组
    hist = edge_dir(img)
    hist = 1000 * hist
    vector = coherence_vector(img)
    vector = [x for y in vector for x in y]
    vector1 = np.asarray(vector)
    hist1 = np.append(hist, vector1)
    max = np.max(hist1)
    min = np.min(hist1)
    hist2 = []
    for j in hist1:
        k = (j - min) / (max - min)  # 归一化
        hist2.append(k)
    hist3 = np.asarray(hist2)
    XX_test.append((hist3.flatten()))

XX_test = np.asarray(XX_test)


def gaussian(dist, sigma=9.0):
    """ Input a distance and return it`s weight"""
    weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return weight


def KNN_classifier(k, dis, X_train, X_labels, X_test):
    assert (dis == 'E' or dis == 'M')
    num_test = X_test.shape[0]
    label_list = []

    if dis == 'E':
        for i in range(num_test):
            distances = np.sqrt(np.sum(((X_train - np.tile(X_test[i], (X_train.shape[0], 1))) ** 2), axis=1))
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            classCount = {}
            for j in topK:
                # index = nearest_k[j]
                # weight = gaussian(distances[index])
                classCount[X_labels[j]] = classCount.get(X_labels[j], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                                      reverse=True)  # operator.itemgetter(1):取classCount数组里的第二维
            label_list.append(sortedClassCount[0][0])
        return np.array(label_list)
    elif dis == 'M':
        for i in range(num_test):
            distances = np.sum(abs(X_train - np.tile(X_test[i], (X_train.shape[0], 1))), axis=1)
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            classCount = {}
            for i in topK:
                classCount[X_train[i]] = classCount.get(X_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            label_list.append(sortedClassCount[0][0])
        return np.array(label_list)


t1 = time.time()
num_test = 500
y_test_pred = KNN_classifier(5, 'E', XX_train, Y_train, XX_test)
num_correct = np.sum(y_test_pred == Y_test)
accuracy = float(num_correct) / num_test
print('M: Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
t2 = time.time()
print(t2 - t1)
