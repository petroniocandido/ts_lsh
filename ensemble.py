from ts-lsh.multiple import MultipleLSH
from ts-lsh.common import owa, get_owa_weights

class EnsembleLSH(MultipleLSH):
  def __init__(self, **kwargs):
    super(EnsembleLSH, self).__init__(**kwargs)
    
    self.aggregation = kwargs.get("aggregation", "mean")

    self.owa_weights = kwargs.get("owa_weights",None)

  def _hashfunction(self, input : np.array, **kwargs):
    hashes = super(EnsembleLSH, self)._hashfunction(input)
    if self.aggregation == "owa" and self.owa_weigts is not None:
      owa(hashes, self.owa_weights)
    else:
      w = get_owa_weights(self.aggregation,input)
      return owa(input, w)
