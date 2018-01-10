# -*- coding:utf-8 -*-

import numpy as np
from load_minist import load_minist_images, load_minist_labels
import logistic_reg


images = load_minist_images('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/train-images-idx3-ubyte.gz')
labels = load_minist_labels('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/train-labels-idx1-ubyte.gz')
bool_index = np.logical_or((labels == 0), (labels == 1)).reshape(1, -1)
label01 = labels[np.ravel(bool_index)].ravel()
image01 = images[bool_index.ravel()]
# theta = logstic_gradient_batch(image01, label01, 10, 0.0001)
theta_sk = logistic_reg.logistic_sk(image01, label01)
print(theta_sk)
