import numpy as np


class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSE(Loss):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        n_samples = self.y_true.shape[0]
        return 2 * (self.y_pred - self.y_true) / n_samples


class CrossEntropy(Loss):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        
        return np.mean(sample_losses)

    def backward(self):
        eps = 1e-15
        y_pred_clipped = np.clip(self.y_pred, eps, 1 - eps)
        n_samples = self.y_true.shape[0]

        return -(self.y_true / y_pred_clipped) / n_samples