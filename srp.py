from ts-lsh.base import LSH

class SignedRandomProjectionLSH(LSH):
  def __init__(self, **kwargs):
    super(SignedRandomProjectionLSH, self).__init__(**kwargs)

    self.weights = np.random.randn(self.length)

  def _hashfunction(self, input : np.array, **kwargs):
    if len(input) != self.length:
      raise Exception("Input length is wrong!")
    return np.dot(self.weights, input)
