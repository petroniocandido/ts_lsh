import numpy as np
from ts_lsh.base import LSH
from ts_lsh.srp import SignedRandomProjectionLSH

class StackedLSH(LSH):
  def __init__(self, **kwargs):
    super(StackedLSH, self).__init__(**kwargs)
    self.layers = []
    self.batch = True

  def append(self, length, **kwargs):
    _type = kwargs.get("type", SignedRandomProjectionLSH)
    self.layers.append(_type(**kwargs))
    if len(self.layers) == 1:
      self.input_length = self.layers[0].input_length
    self.output_length = self.layers[-1].output_length

  def _hashfunction(self, input : np.array):
    if input.shape[1] != self.input_length:
      raise Exception("Array dimensions don't match")
    old = input
    for ct, layer in enumerate(self.layers):
      if ct == 0:
        new = np.array([layer.hash(item) for item in old])
      else:
        n = len(old)
        if self.layers[ct-1].output_length != layer.input_length:
          new = np.array([layer.hash(old[k - layer.input_length : k]) for k in range(layer.input_length,n+1, layer.input_length)])
        else:
          new = np.array([layer.hash(k) for k in old])
      if layer.output_length == 1:
        new = new.flatten()
      old = new
    return old
