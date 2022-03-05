# -*- encoding: utf-8 -*-

import numpy as np
from scipy.linalg import block_diag


class Linear:
    def __init__(self, in_features, out_features, lr=0.001):
        uniform_param = 1.0 / np.sqrt(in_features)
        self.W = np.random.uniform(-uniform_param, uniform_param, size=(out_features, in_features))
        self.b = np.random.uniform(-uniform_param, uniform_param, size=out_features)

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.lr = lr

        self.output = None
        self.layer_grad = None

    def update_output(self, inputs):
        self.output = np.dot(inputs, self.W.transpose()) + self.b
        return self.output

    def update_grad_input(self, inputs, next_layer_grad):
        self.layer_grad = np.dot(next_layer_grad, self.W)
        self.update_grad_params(inputs, next_layer_grad)
        self.update_params()

        return self.layer_grad

    def update_grad_params(self, inputs, next_layer_grad):
        self.grad_W = np.dot(next_layer_grad.transpose(), inputs)
        self.grad_b = np.sum(next_layer_grad, axis=0)

    def update_params(self):
        self.W -= self.lr*self.grad_W
        self.b -= self.lr*self.grad_b

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


class SoftPlus:
    def __init__(self):
        self.output = None
        self.layer_grad = None

    def update_output(self, inputs):
        self.output = np.logaddexp(0, inputs)

        return self.output

    def update_grad_input(self, inputs, next_layer_grad):
        self.layer_grad = next_layer_grad * np.exp(inputs) / (1 + np.exp(inputs))

        return self.layer_grad


class SoftMax:
    def __init__(self):
        self.output = None
        self.layer_grad = None

    def update_output(self, inputs):
        shift_exp = np.exp(inputs - np.max(inputs, axis=1)[:, None])
        self.output = shift_exp / np.sum(shift_exp, axis=1)[:, None]

        return self.output

    def update_grad_input(self, inputs, next_layer_grad):
        shift_exp = np.exp(inputs - np.max(inputs, axis=1)[:, None])
        s = shift_exp / np.sum(shift_exp, axis=1)[:, None]

        self.layer_grad = block_diag(*[-np.outer(s_i, s_i) + np.diag(s_i.flatten()) for s_i in s])

        return (next_layer_grad @ self.layer_grad).reshape(inputs.shape)


class ReLU:
    def __init__(self):
        self.output = None
        self.layer_grad = None

    def update_output(self, inputs):
        self.output = np.maximum(inputs, 0)

        return self.output

    def update_grad_input(self, inputs, next_layer_grad):
        self.layer_grad = next_layer_grad * (inputs > 0)

        return self.layer_grad
