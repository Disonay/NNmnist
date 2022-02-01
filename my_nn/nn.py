from my_nn.layers import Linear, MSECriterion, Sigmoid
import numpy as np


class NeuralNet:
    def __init__(self, inputs, target):
        self.inputs = inputs
        self.target = target

        self.first_linear_layer = Linear(len(inputs[0]), 32)
        self.sigmoid_layer = Sigmoid()
        self.second_linear_layer = Linear(32, 10)
        self.loss_layer = MSECriterion()

    def forward(self, inputs):
        self.first_linear_layer.update_output(inputs)
        self.sigmoid_layer.update_output(self.first_linear_layer.output)
        self.second_linear_layer.update_output(self.sigmoid_layer.output)

    def backward(self, inputs, target):
        self.second_linear_layer.update_grad_input(
            self.loss_layer.update_grad_input(self.second_linear_layer.output, target))
        self.second_linear_layer.update_grad_params(self.sigmoid_layer.output, self.loss_layer.layer_grad)

        self.first_linear_layer.update_grad_input(
            self.sigmoid_layer.update_grad_input(self.first_linear_layer.output, self.second_linear_layer.layer_grad))
        self.first_linear_layer.update_grad_params(inputs, self.sigmoid_layer.layer_grad)

        self.first_linear_layer.update_params()
        self.second_linear_layer.update_params()

    def fit(self, epochs, batch_size):
        for epoch_number in range(epochs):
            for batch, batch_target in zip(
                np.array_split(self.inputs, len(self.inputs) // batch_size),
                np.array_split(self.target, len(self.inputs) // batch_size),
            ):
                self.forward(batch)
                self.backward(batch, batch_target)

                print("Epoch: {}, Loss: {}".format(epoch_number, self.loss_layer.update_output(self.second_linear_layer.output, batch_target)))

    def predict(self, inputs):
        return self.second_linear_layer.update_output(self.sigmoid_layer.update_output(self.first_linear_layer.update_output(inputs)))
