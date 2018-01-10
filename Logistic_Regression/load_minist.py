# -*- coding:utf-8 -*-

import numpy as np
import struct
import gzip


def load_minist_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        filebuf = f.read()
    offset = 0
    magic_number, image_numbs, row_numbs, col_numbs = struct.unpack_from('>IIII', filebuf, offset)
    offset = offset + struct.calcsize('>IIII')
    images = []
    for i in range(image_numbs):
        image_values = struct.unpack_from('>784B', filebuf, offset)
        offset = offset + struct.calcsize('>784B')
        image_values = list(image_values)
        images.append(image_values)
    return np.array(images, dtype='uint8')


def load_minist_labels(filepath):
    with gzip.open(filepath) as f:
        filebuf = f.read()
    offset = 0
    labels = []
    magic_number, image_numbers = struct.unpack_from('>2I', filebuf, offset)
    offset = offset + struct.calcsize('>2I')
    for i in range(image_numbers):
        label_value = struct.unpack_from('>B', filebuf, offset)
        offset = offset + struct.calcsize('>B')
        labels.append(list(label_value))
    return np.array(labels, dtype='uint8')


if __name__ == '__main__':
    print(load_minist_images('E:\\Notes\\ufldl\\Data\\ex1\\train-images-idx3-ubyte.gz').shape)
    print(load_minist_labels('E:\\Notes\\ufldl\\Data\\ex1\\train-labels-idx1-ubyte.gz').shape)
