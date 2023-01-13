from ts-lsh.base import LSH
from ts-lsh.srp import SignedRandomProjectionLSH

class MultipleLSH(LSH):
  def __init__(self, **kwargs):
    super(MultipleLSH, self).__init__(**kwargs)
    
    self.num : int = kwargs.get("number",2)

    self.hashes = [SignedRandomProjectionLSH(**kwargs) for k in range(self.num)]

  def _hashfunction(self, input : np.array, **kwargs):
    return [lsh.hash(input) for lsh in self.hashes]
