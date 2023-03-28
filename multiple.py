import numpy as np
from ts_lsh.base import LSH
from ts_lsh.srp import SignedRandomProjectionLSH

class MultipleLSH(LSH):
  def __init__(self, **kwargs):
    super(MultipleLSH, self).__init__(**kwargs) 
    
    self.output_length = kwargs.get("num_components", 2)

    self.scale = kwargs.get("scale", 1.0)
    self.dist = kwargs.get('dist','normal')
    if self.dist == 'normal':
      self.weights = np.random.randn(self.output_length, self.input_length) * self.scale
    elif self.dist == 'unif':
      self.weights = (np.random.rand(self.output_length, self.input_length) * 2 * self.scale) - self.scale

  def _hashfunction(self, input : np.array, **kwargs):
    return np.dot(self.weights, input)

  
class MultipleRandomSampledLSH(LSH):
  def __init__(self, **kwargs):
    super(MultipleRandomSampledLSH, self).__init__(**kwargs)
    
    self.output_length = kwargs.get("num_components", 2)

    self.sample_size : int = kwargs.get("sample_size", self.input_length // 2) # Sample must be lower than length

    self.scale = kwargs.get("scale", 1.0)
    self.dist = kwargs.get('dist','normal')
    if self.dist == 'normal':
      self.weights = np.random.randn(self.output_length, self.sample_size) * self.scale
    elif self.dist == 'unif':
      self.weights = (np.random.rand(self.output_length, self.sample_size) * 2 * self.scale) - self.scale

    self.sample_indexes = []

    for k in range(self.output_length):
      self.sample_indexes.append( np.random.choice(self.input_length, self.sample_size, replace = False) )

  def _hashfunction(self, input : np.array, **kwargs):
    ret = []
    for k in range(self.output_length):
      i = input[[self.sample_indexes[k]]]
      ret.append(np.dot(self.weights[k,:], i))
    return np.array(ret)
