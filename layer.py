from neuron import Neuron

class Layer:
  def __init__(self, inputs, nout, activation="relu"):
    self.neurons = [Neuron(inputs, activation) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [param for neuron in self.neurons for param in neuron.parameters()]