import math
import numpy as np

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def normalization(data : np.array ) -> np.array :
  rows, cols = data.shape
  new = np.zeros((rows,cols))

  for col in range(cols):
    values = data[:,col]
    _avg = np.mean(values)
    _std = np.std(values)
    values = values - _avg
    values = values / _std
    new[:, col] = values

  return new

def standardization(data : np.array) -> np.array :
  rows, cols = data.shape
  new = np.zeros((rows,cols))

  for col in range(cols):
    values = data[:,col]
    _min = np.min(values)
    _max = np.max(values)
    rng = _max - _min
    values = values - _min
    values = values / rng
    new[:, col] = values

  return new

def differentiation(data):
  rows, cols = data.shape
  new = np.zeros((rows,cols))

  for col in range(cols):
    values = data[:,col]
    diff = np.array([values[row-1] - values[row] for row in range(1,rows)])
    new[1:, col] = diff

  return new

def owa(values : np.array, weights : np.array) -> np.float :
  if len(values) != len(weights):
    raise Exception("Array dimensions don't match")

  return np.dot(np.sort(values), weights)

def get_owa_weights(method : str, values : np.array ) -> np.array :
  n = len(values)
  if method == 'mean':
    return np.ones(n) * 1/n
  elif method == 'median':
    w = np.zeros(n)
    m = n % 2
    w[n-1] = 1
    return w
  elif method == 'max':
    w = np.zeros(n)
    w[-1] = 1
    return w
  elif method == 'min':
    w = np.zeros(n)
    w[0] = 1
    return w
  elif method == 'sum':
    return np.ones(n)
