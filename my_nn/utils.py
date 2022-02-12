# -*- encoding: utf-8 -*-

import numpy as np


def kron_delta(index):
    e = np.zeros((10, 1))
    e[index] = 1
    return e
