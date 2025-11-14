import random
from value import Value

class Neuron:
  def __init__(self, nin, activation="relu", initialization="random"):
    self.w = self._initialize_weights(nin, initialization)
    self.b = Value(random.uniform(-1, 1)) # TODO: should bias be initialized same way as weights?
    self.activation = activation

  def __call__(self, x):
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    
    if self.activation == "tanh":
        out = act.tanh()
    elif self.activation == "relu":
        out = act.relu()
    else:
        out = act

    return out

  def parameters(self):
    return self.w + [self.b]
  
  def _initialize_weights(self, nin, initialization):
    if initialization == "random":
      limit = 1
    elif initialization == "xavier":
      limit = (6 / nin) ** 0.5
    elif initialization == "he":
      limit = (2 / nin) ** 0.5
    else:
      raise ValueError("Unsupported initialization method.")
    
    return [Value(random.uniform(-limit, limit)) for _ in range(nin)]