import numpy as np
from ts_lsh.common import sigmoid

class LSH(object):
  
  def __init__(self, **kwargs):
    self.name : str = kwargs.get("name","LSH")
    self.input_length : int = kwargs.get("input_length",0)
    self.output_length : int = kwargs.get("output_length",1)
    self.hashtype : str = kwargs.get("hashtype","real")
    self.width : int = kwargs.get("width",None)
    self.activation : str = kwargs.get("activation", "sign")

    if self.hashtype == "binary":
      self.nbins = 2
    
    if self.hashtype == "integer":
      if self.width is None:
        raise Exception("The parameter width must be informed!")

      self.b = np.random.randint(0, self.width, 1)

  def hash(self, input : np.array, **kwargs):
    if self.hashtype == "binary":
      if self.activation == "sign":
        return 1 if self._hashfunction(input, **kwargs) > 0 else 0
      elif self.activation == "sigmoid":
        return sigmoid(self._hashfunction(input, **kwargs))
    elif self.hashtype == "integer":
      return np.round((self._hashfunction(input, **kwargs) + self.b)/self.width, 0)
    else:
      if self.activation == "exp":
        return np.exp(self._hashfunction(input, **kwargs))
      elif self.activation == "scale":
        factor = kwargs.get('factor', np.pi)
        return self._hashfunction(input, **kwargs) * factor
      else:
        return self._hashfunction(input, **kwargs)

  def _hashfunction(self, input : np.array, **kwargs):
    pass  
