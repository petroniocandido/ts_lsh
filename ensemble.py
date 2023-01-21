from ts-lsh.multiple import MultipleLSH
from ts-lsh.common import owa, get_owa_weights

class EnsembleLSH(MultipleLSH):
  def __init__(self, **kwargs):
    super(EnsembleLSH, self).__init__(**kwargs)
    
    self.aggregation = kwargs.get("aggregation", "mean")

    self.aggregation = kwargs.get("aggregation", "srp")

    self.aggregation_weights = kwargs.get("aggregation_weights",None)

    if self.aggregation == "srp":
      self.scale = kwargs.get("scale", 1.0)

      self.dist = kwargs.get('dist','normal')
      if self.dist == 'normal':
        self.aggregation_weights = np.random.randn(self.num) * self.scale
      elif self.dist == 'unif':
        self.aggregation_weights = (np.random.rand(self.num) * 2 * self.scale) - self.scale

  def _hashfunction(self, input : np.array, **kwargs):
    hashes = super(EnsembleLSH, self)._hashfunction(input)
    if self.aggregation == "owa":
      if self.aggregation_weights is None:
        raise Exception("Ordered Weight Aggregation weights are not set!")
      return owa(hashes, self.aggregation_weights)
    elif self.aggregation == "srp":
      return np.dot(self.aggregation_weights, hashes)
    else:
      if self.aggregation_weights is None:
        self.aggregation_weights = get_owa_weights(self.aggregation,input)
      return owa(hashes, self.aggregation_weights)
