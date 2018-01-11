# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from load_minist import load_minist_images, load_minist_labels
from logistic_reg import logistic_gradient_batch, logistic_gradient_sto, cost, sigmod, logistic_sk


images = load_minist_images('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/train-images-idx3-ubyte.gz')
labels = load_minist_labels('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/train-labels-idx1-ubyte.gz')
bool_index = np.logical_or((labels == 0), (labels == 1)).reshape(1, -1)
label01 = labels[np.ravel(bool_index)]
image01 = images[bool_index.ravel()] / 255.0
theta = logistic_gradient_batch(image01, label01, 1000, 0.005)
# theta_sk = logistic_reg.logistic_sk(image01, label01)


# data = pd.read_csv('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/ex2data1.txt', names=['1','2','3'])
# train_x = data.iloc[:,:2].values / 100
# train_y = data['3'].values.reshape(100,1)
# theta = logistic_gradient_sto(train_x, train_y, 1000, 0.002)
# theta_sk = logistic_sk(train_x,train_y.ravel())
print(theta)
# print(theta_sk)
print(cost(theta, image01, label01))
