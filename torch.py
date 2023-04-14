import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

class LSH(nn.Module):
  
  def __init__(self, **kwargs):
    super().__init__()
    self.batch : bool = kwargs.get("batch", False)
    self.name : str = kwargs.get("name","LSH")
    self.input_length : int = kwargs.get("input_length",0)
    self.output_length : int = kwargs.get("output_length",1)
    self.hashtype : str = kwargs.get("hashtype","real")
    self.width : int = kwargs.get("width",None)

    if self.hashtype == "binary":
      self.nbins = 2
    
    if self.hashtype == "integer":
      if self.width is None:
        raise Exception("The parameter width must be informed!")

      self.b = np.random.randint(0, self.width, 1)

  def hash(self, input : np.array, **kwargs):
    return self.forward(torch.from_numpy(input), **kwargs)
  
  
  
class SignedRandomProjectionLSH(LSH):
  def __init__(self, **kwargs):
    super(SignedRandomProjectionLSH, self).__init__(**kwargs)

    scale = kwargs.get("scale", 1.0)

    dist = kwargs.get('dist','unif')
    if dist == 'normal':
      self.weights = nn.Parameter(torch.randn(self.input_length) * scale, requires_grad=False)
    elif dist == 'unif':
      self.weights = nn.Parameter((torch.rand(self.input_length) * (scale * 2)) - scale, requires_grad=False)

  def forward(self, input):
    return torch.dot(self.weights, input)


  class MultipleLSH(LSH):
  def __init__(self, **kwargs):
    super(MultipleLSH, self).__init__(**kwargs) 
    
    # It is important to keep the number of components and the output length as separated attributes
    self.output_length = self.num_components = kwargs.get("num_components", 2)

    self.scale = kwargs.get("scale", 1.0)
    self.dist = kwargs.get('dist','unif')
    if self.dist == 'normal':
      self.weights = nn.Parameter(torch.randn(self.num_components, self.input_length) * self.scale, requires_grad=False)
    elif self.dist == 'unif':
      self.weights = nn.Parameter((torch.rand(self.num_components, self.input_length) * (self.scale * 2)) - self.scale, requires_grad=False)

  def forward(self, input):
    return np.dot(self.weights, input)

  
class MultipleRandomSampledLSH(LSH):
  def __init__(self, **kwargs):
    super(MultipleRandomSampledLSH, self).__init__(**kwargs)
    
    self.output_length = self.num_components = kwargs.get("num_components", 2)

    self.sample_size : int = kwargs.get("sample_size", self.input_length // 2) # Sample must be lower than length

    self.scale = kwargs.get("scale", 1.0)
    self.dist = kwargs.get('dist','unif')
    if self.dist == 'normal':
      self.weights = nn.Parameter(torch.randn(self.num_components, self.sample_size) * self.scale, requires_grad=False)
    elif self.dist == 'unif':
      self.weights = nn.Parameter((torch.rand(self.num_components, self.sample_size) * (self.scale * 2)) - self.scale, requires_grad=False)

    self.sample_indexes = []

    for k in range(self.output_length):
      self.sample_indexes.append( [int(k) for k in np.random.choice(self.input_length, self.sample_size, replace = False)] )

  def forward(self, input):
    ret = torch.zeros(self.num_components)
    for k in range(self.num_components):
      i = input[[self.sample_indexes[k]]]
      ret[k] = torch.dot(self.weights[k,:], i)
    return ret
  
  
