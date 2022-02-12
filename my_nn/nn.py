# -*- encoding: utf-8 -*-

import numpy as np


class NeuralNet:
    def __init__(self):
        self.layers = []
        self.loss_layer = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss_layer = loss

    def fit(self, inputs, target, epochs, batch_size):

        for epoch_number in range(epochs):
            for batch, batch_target in zip(
                np.array_split(inputs, len(inputs) // batch_size),
                np.array_split(target, len(inputs) // batch_size),
            ):
                self.forward(batch)
                self.backward(batch, batch_target)

                print(
                    "Epoch: {}, Loss: {}".format(
                        epoch_number,
                        self.loss_layer.update_output(self.layers[-1].output, batch_target),
                    )
                )

    def forward(self, inputs):
        for order, layer in enumerate(self.layers):
            if order != 0:
                layer.update_output(self.layers[order - 1].output)
            else:
                layer.update_output(inputs)
        return self.layers[-1].output

    def backward(self, inputs, target):
        next_layer_grad = self.loss_layer.update_grad_input(self.layers[-1].output, target)
        for layer_order in range(len(self.layers)-1, -1, -1):
            if layer_order != 0:
                next_layer_grad = self.layers[layer_order].update_grad_input(
                    self.layers[layer_order - 1].output, next_layer_grad
                )
            else:
                self.layers[layer_order].update_grad_input(inputs, next_layer_grad)

    def predict(self, inputs):
        return self.forward(inputs)
