# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_example(data):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 9))
    imgs = np.random.choice(len(data), 9, replace=False)
    for row in range(3):
        for col in range(3):
            ax[row, col].imshow(data[imgs[row * 3 + col]])
    return fig


def plot_bar(data):
    fig, ax = plt.subplots(figsize=(16, 9))
    uniq, counts = np.unique(data, return_counts=True)
    ax.bar(uniq, counts)
    return fig
