# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from load_minist import load_minist_images, load_minist_labels
from logistic_reg import logistic_gradient_batch, logistic_gradient_sto, cost, sigmod, logistic_sk, logistic_tf, accuracy

images = load_minist_images('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/train-images-idx3-ubyte.gz')
labels = load_minist_labels('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/train-labels-idx1-ubyte.gz')
images_test = load_minist_images('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/t10k-images-idx3-ubyte.gz')
labels_test = load_minist_labels('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/t10k-labels-idx1-ubyte.gz')
bool_index = np.logical_or((labels == 0), (labels == 1)).reshape(1, -1)
label01 = labels[np.ravel(bool_index)]
image01 = images[bool_index.ravel()] / 255.0

bool_index_test = np.logical_or((labels_test == 0), (labels_test == 1)).reshape(1, -1)
label01_test = labels_test[np.ravel(bool_index_test)]
image01_test = images_test[bool_index_test.ravel()] / 255.0


# theta, cost_data = logistic_gradient_batch(image01, label01, 1000, 0.005)
# theta_sk, b_sk = logistic_sk(image01, label01.ravel())

# theta_tf= logistic_tf(image01, label01, 500, 0.05)
theta_sk = logistic_sk(image01, label01.ravel())

# data = pd.read_csv('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/ex2data1.txt', names=['1', '2', '3'])
# train_x = data.iloc[:, :2].values / 100
# train_y = data['3'].values.reshape(100, 1)
# theta, cost_data = logistic_gradient_batch(train_x, train_y, 10000, 0.002)
# theta_tf, b_tf = logistic_tf(train_x, train_y, 10000, 0.001)
# theta_sk, b_sk = logistic_sk(train_x, train_y.ravel())
# print(theta)
# theta_sk = np.append(b_sk, theta_sk)
# print(theta_sk.reshape(-1,1))
print(accuracy(theta_sk, image01_test, label01_test))