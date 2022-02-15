# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from biokit.viz import corrplot
import pandas as pd
import numpy as np


def plot_example(data):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 9))
    imgs = np.random.choice(len(data), 9, replace=False)
    for row in range(3):
        for col in range(3):
            ax[row, col].imshow(data[imgs[row * 3 + col]])
    fig.tight_layout()
    return fig


def plot_bar(data):
    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots(figsize=(16, 9))
    uniq, counts = np.unique(data, return_counts=True)
    ax.bar(uniq, counts)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Count")
    ax.set_xticks(range(10), ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"])
    fig.tight_layout()
    return fig
