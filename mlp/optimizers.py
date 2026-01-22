import numpy as np


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def step(self, params, grads):
        raise NotImplementedError("This method should be overridden by subclasses.")
    

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        
    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.learning_rate * g
            
            
class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = []
        
    def step(self, params, grads):
        if not self.velocities:
            self.velocities = [np.zeros_like(p) for p in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad
            param += self.velocities[i]
            

class Adam(Optimizer):
    def __init__(self, learning_rate = 0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0
        
    def step(self, params, grads):
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)