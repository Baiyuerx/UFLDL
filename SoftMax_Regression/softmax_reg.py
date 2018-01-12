# -*- coding:utf-8 -*-

import numpy as np


def hypothesis(theta, x):
    exp = np.exp(np.dot(x, theta))
    probability = exp / np.sum(exp, axis=1).reshape(exp.shape[0], -1)
    predicted_y = np.argmax(probability, axis=1)
    return probability, predicted_y


def softmax_train(x, y, epoch=1000, alpha=0.001, weight_decay_rate=0.01):
    label_cnts = len(np.unique(y))
    theta = np.ones(x.shape[1], label_cnts)
    m = y.shape[0]
    for e in range(epoch):
        prob, pred_y = hypothesis(theta, x)
        theta_ = np.empty_like(theta)
        for i in range(label_cnts):
            y_ = (y == i).astype(int)
            theta_[:, [i]] = np.sum(x * (y_ - prob), axis=0).reshape(-1, 1)
        theta = theta + alpha * 1/m * theta_ - weight_decay_rate * theta
    return theta

if __name__ == '__main__':
    print(hypothesis(np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
