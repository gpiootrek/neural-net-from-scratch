from layer import Layer

class MLP:
  def __init__(self, inputs, layers_sizes):
    sz = [inputs] + layers_sizes # For example [3] + [4, 4, 1]
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(layers_sizes))]
    
  # def __init__(self, layers: list[Layer]):
  #   self.layers = layers

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [param for layer in self.layers for param in layer.parameters()]
  
  def compile():
    pass
  
  def fit():
    pass
  
  def evaluate():
    pass