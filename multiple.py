from ts-lsh.base import LSH
from ts-lsh.srp import SignedRandomProjectionLSH

class MultipleLSH(LSH):
  def __init__(self, **kwargs):
    super(MultipleLSH, self).__init__(**kwargs)
    
    self.num : int = kwargs.get("number",2)

    scale = kwargs.get("scale", 1.0)

    dist = kwargs.get('dist','normal')
    if dist == 'normal':
      self.weights = np.random.randn(self.num, self.length) * scale
    elif dist == 'unif':
      self.weights = (np.random.rand(self.num, self.length) * 2 * scale) - scale

  def _hashfunction(self, input : np.array, **kwargs):
    return np.dot(self.weights, input)
