# -*- coding:utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf


def sigmod(x):
    result = 1.0 / (1.0 + np.exp(-x))
    return result


def cost(theta, train_x, train_y):
    train_x = np.append(np.ones((train_x.shape[0], 1)), train_x, axis=1)
    A = sigmod(np.dot(train_x, theta))
    cost = -np.sum(train_y * np.log(A) + (1 - train_y) * np.log(1 - A))
    return cost / train_x.shape[0]


def accuracy(theta, x, y):
    x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
    A = sigmod(np.dot(x, theta))
    predicted = (A > 0.5).astype(int)
    correct = np.sum(predicted.reshape(-1,1) == y.reshape(-1,1))
    return correct/len(y)


def logistic_gradient_batch(train_x, train_y, epoch=1000, alpha=0.001):
    X = np.append(np.ones((train_x.shape[0], 1)), train_x, axis=1)
    Y = train_y
    theta = np.zeros((X.shape[1], 1))
    m = X.shape[0]
    cost_data = [cost(theta, train_x, train_y)]

    for i in range(epoch):
        loss = Y - sigmod(np.dot(X, theta))
        theta = theta + 1.0 / m * alpha * np.dot(X.T, loss)
        cost_data.append(cost(theta, train_x, train_y))
    return theta, cost_data


def logistic_gradient_sto(train_x, train_y, epoch=1000, alpha=0.001):
    train_x = np.append(np.ones((train_x.shape[0], 1)), train_x, axis=1)
    theta = np.ones((train_x.shape[1], 1))

    for m in range(epoch):
        for i in range(train_x.shape[0]):
            loss = train_y[i] - sigmod(np.dot(train_x[i], theta))
            theta = theta + alpha * np.dot(train_x[i].reshape(1, -1), loss)
    return theta


def logistic_tf(train_x, train_y, epoch=1000, alpha=0.001):
    X = tf.placeholder(tf.float32, (None, train_x.shape[1]))
    Y = tf.placeholder(tf.float32, (None, 1))

    W = tf.Variable(tf.zeros([train_x.shape[1], 1]))
    b = tf.Variable(tf.zeros(1))

    predicted = tf.nn.sigmoid(tf.matmul(X, W) + b)

    cost = -tf.reduce_mean(Y * tf.log(predicted) + (1 - Y) * tf.log(1.0 - predicted))

    optimazer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for e in range(epoch):
            _ = sess.run([optimazer], feed_dict={X: train_x, Y: train_y})
        return np.append(b.eval(), W.eval())


def logistic_sk(train_x, train_y):
    lr = LogisticRegression(fit_intercept=True)
    lr.fit(train_x, train_y)
    return np.append(lr.intercept_, np.array(lr.coef_).reshape((train_x.shape[1], 1)))


if __name__ == '__main__':
    a1 = np.array([1, 2, 3]).reshape(3, 1)
    b1 = np.array([1, 0, 1]).reshape(-1, 1)
    print(logistic_gradient_batch(a1, b1))
