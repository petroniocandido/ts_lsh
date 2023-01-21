from ts-lsh.base import LSH
from ts-lsh.srp import SignedRandomProjectionLSH

class MultipleLSH(LSH):
  def __init__(self, **kwargs):
    super(MultipleLSH, self).__init__(**kwargs)
    
    self.num : int = kwargs.get("number",2)

    self.scale = kwargs.get("scale", 1.0)
    self.dist = kwargs.get('dist','normal')
    if self.dist == 'normal':
      self.weights = np.random.randn(self.num, self.length) * self.scale
    elif self.dist == 'unif':
      self.weights = (np.random.rand(self.num, self.length) * 2 * self.scale) - self.scale

  def _hashfunction(self, input : np.array, **kwargs):
    return np.dot(self.weights, input)
