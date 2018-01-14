# -*- coding:utf-8 -*-

import numpy as np


def hypothesis(theta, x):
    exp = np.exp(np.dot(x, theta))
    probability = exp / np.sum(exp, axis=1).reshape(exp.shape[0], -1)
    predicted_y = np.argmax(probability, axis=1)
    return probability, predicted_y


def softmax_train(x, y, epoch=1000, alpha=0.001, weight_decay_rate=0.01):
    label_cnts = len(np.unique(y))
    theta = np.random.normal(0, 0.01, (x.shape[1], label_cnts))
    m = y.shape[0]
    epoch_n = 0
    for e in range(epoch):
        prob, pred_y = hypothesis(theta, x)
        theta_ = np.empty_like(theta)
        for i in range(label_cnts):
            y_ = (y == i).astype(int)
            theta_[:, [i]] = np.sum(x * (y_ - prob[:,[i]]), axis=0).reshape(-1, 1)
        theta = theta + alpha * 1/m * theta_ - weight_decay_rate * theta
        epoch_n = epoch_n + 1
        print("have trained %d times"%epoch_n)
    return theta


def accuracy(theta, x, y):
    predicted = np.argmax(np.dot(x, theta), axis=1)
    corrected = (predicted == y.ravel()).astype(int)
    return np.mean(corrected)


if __name__ == '__main__':
    x = (np.random.rand(5,3)*10).astype(int)
    y = np.array([[0,1,0,0,1]]).reshape(-1, 1)
    print(softmax_train(x, y))