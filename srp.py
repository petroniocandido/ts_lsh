import numpy as np
from ts_lsh.base import LSH

class SignedRandomProjectionLSH(LSH):
  def __init__(self, **kwargs):
    super(SignedRandomProjectionLSH, self).__init__(**kwargs)

    scale = kwargs.get("scale", 1.0)

    dist = kwargs.get('dist','normal')
    if dist == 'normal':
      self.weights = np.random.randn(self.input_length) * scale
    elif dist == 'unif':
      self.weights = (np.random.rand(self.input_length) * 2 * scale) - scale

  def _hashfunction(self, input : np.array, **kwargs):
    if len(input) != self.input_length:
      raise Exception("Input length is wrong!")
    return np.dot(self.weights, input)
