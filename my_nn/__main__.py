# -*- encoding: utf-8 -*-
import os
import pickle

import numpy as np
from keras.datasets import mnist

from my_nn.layers import Linear, MSECriterion, ReLU, Sigmoid, SoftPlus, CrossEntropyLoss, SoftMax
from my_nn.nn import NeuralNet
from my_nn.utils import kron_delta

nn = NeuralNet()
nn.add_layer(Linear(784, 128))
nn.add_layer(ReLU())
nn.add_layer(Linear(128, 64))
nn.add_layer(ReLU())
nn.add_layer(Linear(64, 10))
nn.add_layer(SoftMax())
nn.set_loss(CrossEntropyLoss())

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print(os.listdir())
nn.fit(
    train_X.reshape(60000, 784) / 255,
    np.array(list(map(kron_delta, train_y))).reshape(60000, 10),
    10,
    100,
)

with open("nn.pickle", "wb") as nn_file:
    pickle.dump(nn, nn_file)
