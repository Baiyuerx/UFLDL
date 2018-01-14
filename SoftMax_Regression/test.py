#-*- coding:utf-8 -*-

import numpy as np
from Logistic_Regression.load_minist import load_minist_images, load_minist_labels
from SoftMax_Regression.softmax_reg import softmax_train, accuracy

images = load_minist_images('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/train-images-idx3-ubyte.gz')
labels = load_minist_labels('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/train-labels-idx1-ubyte.gz')
images_test = load_minist_images('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/t10k-images-idx3-ubyte.gz')
labels_test = load_minist_labels('/home/selwb/Document/Machine_learning/UFLDL/Data/ex1/t10k-labels-idx1-ubyte.gz')

images = images / 255.0

theta = softmax_train(images, labels, 500)
print(theta)
print(accuracy(theta, images_test, labels_test))



