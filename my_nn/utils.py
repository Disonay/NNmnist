# -*- encoding: utf-8 -*-

import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize


def kron_delta(index):
    e = np.zeros((10, 1))
    e[index] = 1
    return e


def image_preprocessing(img):
    return resize(1 - rgb2gray(rgba2rgb(img / 255)), (28, 28))


def get_batches(inputs, target, batch_size):
    indexes = list(range(len(inputs)))
    np.random.shuffle(indexes)
    for batch_start in filter(lambda x: len(indexes) - x >= batch_size, indexes[::batch_size]):
        yield inputs[batch_start:batch_start+batch_size], target[batch_start:batch_start+batch_size]