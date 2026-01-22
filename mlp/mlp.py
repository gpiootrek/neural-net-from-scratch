import numpy as np


class MLP:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_loss):
        for layer in reversed(self.layers):
            grad_loss = layer.backward(grad_loss)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def grads(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.grads())
        return grads

    def zero_grads(self):
        for layer in self.layers:
            layer.zero_grads()