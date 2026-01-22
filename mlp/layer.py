import numpy as np

class Layer:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def grads(self):
        return []

    def zero_grads(self):
        pass

class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        scale = np.sqrt(1.0 / n_inputs)
        self.weights = np.random.randn(n_inputs, n_outputs) * scale
        self.bias = np.zeros((1, n_outputs))
        
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad_output):
        self.grad_weights += np.dot(self.inputs.T, grad_output)
        self.grad_bias += np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.weights.T)

    def parameters(self):
        return [self.weights, self.bias]

    def grads(self):
        return [self.grad_weights, self.grad_bias]

    def zero_grads(self):
        self.grad_weights.fill(0.0)
        self.grad_bias.fill(0.0)

class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        return grad_output * (self.inputs > 0)

class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output**2)

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.output * (1 - self.output))