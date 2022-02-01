# -*- encoding: utf-8 -*-

import numpy as np
import scipy.special
import scipy.signal


class Linear:
    def __init__(self, in_features, out_features):
        uniform_param = 1.0 / np.sqrt(in_features)
        self.W = np.random.uniform(-uniform_param, uniform_param, size=(out_features, in_features))
        self.b = np.random.uniform(-uniform_param, uniform_param, size=out_features)

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.output = None
        self.layer_grad = None

    def update_output(self, inputs):
        self.output = np.dot(inputs, self.W.transpose()) + self.b
        return self.output

    def update_grad_input(self, next_layer_grad):
        self.layer_grad = np.dot(next_layer_grad, self.W)

        return self.layer_grad

    def update_grad_params(self, inputs, next_layer_grad):
        self.grad_W = np.dot(next_layer_grad.transpose(), inputs)
        self.grad_b = np.sum(next_layer_grad, axis=0)

    def update_params(self):
        self.W -= self.grad_W
        self.b -= self.grad_b

    def set_zeros_grad_params(self):
        self.grad_W.fill(0)
        self.grad_b.fill(0)

    def get_params(self):
        return [self.W, self.b]

    def get_grad_params(self):
        return [self.grad_W, self.grad_b]


class Sigmoid:
    def __init__(self):
        self.output = None
        self.layer_grad = None

    def update_output(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

        return self.output

    def update_grad_input(self, inputs, next_layer_grad):
        self.layer_grad = next_layer_grad * (np.ones(inputs.shape) - self.output) * self.output

        return self.layer_grad


class MSECriterion:
    def __init__(self):
        self.output = None
        self.layer_grad = None

    def update_output(self, inputs, target):
        self.output = np.power(np.linalg.norm(inputs - target), 2) / inputs.size

        return self.output

    def update_grad_input(self, inputs, target):
        self.layer_grad = (2 / inputs.size) * (inputs - target)

        return self.layer_grad
