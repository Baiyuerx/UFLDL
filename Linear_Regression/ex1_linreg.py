#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('..')


def load_data(path):
    with open(path) as f:
        data = [list(map(float, i.strip().split())) for i in f.readlines()]
    data_array = np.array(data)
    m_, n_ = data_array.shape
    extra_ones = np.ones((m_, 1))
    data_array = np.append(extra_ones, data_array, axis=1)
    train_x = data_array[:int(m_ * 2 / 3), :14]
    train_y = data_array[:int(m_ * 2 / 3), [14]]
    test_x = data_array[int(m_ * 2 / 3):, :14]
    test_y = data_array[int(m_ * 2 / 3):, [14]]
    return train_x, test_x, train_y, test_y, m_, n_ + 1


def train_with_sklearn(train_x, train_y):
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(train_x, train_y)
    return linreg.coef_


def train_batch(train_x, train_y, alpha, epoch):
    m_, n_ = train_x.shape
    theta = np.zeros((n_, 1))
    cost_data = [cost(theta, train_x, train_y)]
    count = 0
    while True:
        predicted_y = np.dot(train_x, theta)
        loss = predicted_y - train_y
        # old_theta = theta
        theta = theta - alpha * (np.dot(loss.T, train_x)).T / m_
        count += 1
        cost_data.append(cost(theta, train_x, train_y))
        if count % epoch == 0:
            return theta, cost_data


def train_stochastic(train_x, train_y, alpha, epoch):
    m_, n_ = train_x.shape
    theta = np.zeros((n_, 1))
    cost_data = [cost(theta, train_x, train_y)]
    count = 0
    while True:
        for i in range(m_):
            predicted_y = np.dot(train_x[i], theta)
            loss = np.asscalar(predicted_y - train_y[i])
            # old_theta = theta
            theta = theta - alpha * loss * train_x[i].reshape((14, 1)) / m_
        count += 1
        cost_data.append(cost(theta, train_x, train_y))
        if count % epoch == 0:
            return theta, cost_data


def train_tf(train_x, train_y, alpha, epoch, optimizer=tf.train.GradientDescentOptimizer):
    X = tf.placeholder(tf.float64, shape=train_x.shape)
    Y = tf.placeholder(tf.float64, shape=train_y.shape)

    with tf.variable_scope('linear_regression'):
        theta = tf.get_variable('theta', (X.shape[1], 1), initializer=tf.constant_initializer(), dtype=tf.float64)
        y_predicted = tf.matmul(X, theta)
        loss = 1 / (2 * len(train_x)) * tf.matmul((y_predicted - Y), (y_predicted - Y), transpose_a=True)

    opt = optimizer(learning_rate=alpha)
    opt_operation = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_data = []
        for i in range(epoch):
            _, loss_val, theta_tf = sess.run([opt_operation, loss, theta], feed_dict={X: train_x, Y: train_y})
            loss_data.append(loss_val[0, 0])

            if len(loss_data) > 1 and np.abs(
                    loss_data[-1] - loss_data[-2]) < 10 ** -9:  # early break when it's converged
                # print('Converged at epoch {}'.format(i))
                break

    tf.reset_default_graph()
    return theta_tf, loss_data


def cost(theata, X, Y):
    loss = np.dot(X, theata) - Y
    m = X.shape[0]
    cost = 1 / (2 * m) * np.sum(np.square(loss))
    return cost


if __name__ == "__main__":
    file_path = "../stanford_dl_ex-master/ex1/housing.data"
    train_X, test_X, train_Y, test_Y, m, n = load_data(file_path)
    # theta_sto, cost_sto = train_stochastic(train_X, train_Y, 0.000001, 100000)
    # theta_batch, cost_batch = train_batch(train_X, train_Y, 0.000001, 100000)
    # theta_tf, cost_tf = train_tf(train_X, train_Y, 0.000001, 100000)
    #
    sk_theta = np.array(train_with_sklearn(train_X, train_Y)).T
    # #
    # # loss1 = np.dot(test_X, sk_theta) - test_Y
    # cost_1 = cost(theta_sto, test_X, test_Y)
    # cost_2 = cost(theta_batch, test_X, test_Y)
    # cost_3 = cost(theta_tf, test_X, test_Y)
    cost_sk = cost(sk_theta, test_X, test_Y)
    print(cost_sk)
    print(sk_theta)
