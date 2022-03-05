# -*- encoding: utf-8 -*-

import numpy as np


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


class CrossEntropyLoss:
    def __init__(self):
        self.output = None
        self.layer_grad = None

    def update_output(self, inputs, target):
        self.output = np.mean(np.sum(-target*np.log(inputs), axis=1))

        return self.output

    def update_grad_input(self, inputs, target):
        self.layer_grad = (-target / inputs).reshape(1, -1)

        return self.layer_grad
