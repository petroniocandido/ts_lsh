from ts-lsh.base import LSH
from ts-lsh.srp import SignedRandomProjectionLSH

class StackedLSH(LSH):
  def __init__(self, **kwargs):
    super(StackedLSH, self).__init__(**kwargs)
    self.layers = []

  def append(self, length, **kwargs):
    _type = kwargs.get("type", SignedRandomProjectionLSH)
    self.layers.append(_type(length=length, **kwargs))

  def _hashfunction(self, input : np.array):
    if input.shape[1] != self.length:
      raise Exception("Array dimensions don't match")
    old = input
    for ct, layer in enumerate(self.layers):
      if ct == 0:
        new = np.array([layer.hash(item) for item in old]).flatten()
      else:
        n = len(old)
        new = np.array([layer.hash(old[k - layer.length : k]) for k in range(layer.length,n, layer.length)]).flatten()
      old = new
    return old
