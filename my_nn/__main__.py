# -*- encoding: utf-8 -*-

from my_nn.nn import NeuralNet
from my_nn.layers import Linear, Sigmoid, MSECriterion
from my_nn.utils import kron_delta
from keras.datasets import mnist
import numpy as np
import pickle

nn = NeuralNet()
nn.add_layer(Linear(784, 512))
nn.add_layer(Sigmoid())
nn.add_layer(Linear(512, 10))
nn.add_layer(Sigmoid())
nn.set_loss(MSECriterion())

(train_X, train_y), (test_X, test_y) = mnist.load_data()
nn.fit(train_X.reshape(60000, 784) / 255, np.array(list(map(kron_delta, train_y))).reshape(60000, 10), 30, 100)

with open("nn.pickle", "wb") as nn_file:
    pickle.dump(nn, nn_file)