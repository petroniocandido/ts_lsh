import math
import numpy as np

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def euclidean(a : np.array, b : np.array):
  return np.sqrt(np.sum((a-b)**2))

def distance_matrix(data : np.array, fn_distance) -> np.array:
  if len(data.shape) > 1:
    l,_ = data.shape 
  else:
    l = data.shape[0] 
  matrix = np.zeros((l,l))
  for i in range(l):
    for j in range(i+1,l):
      d = fn_distance(data[i,],data[j,])
      matrix[i,j] = d
      matrix[j,i] = d
  return matrix

def matrix_histogram(mat : np.array, nbins : int):
  m = np.min(mat[~np.eye(mat.shape[0],dtype=bool)]) # all elements except the diagonal
  M = np.max(mat)
  bins = np.linspace(m,M,nbins)
  freq, bins = np.histogram(mat, bins=bins)
  freq = freq.astype(np.float64)
  freq /= np.float64(2.)
  freq /= np.sum(freq)
  return bins, freq

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
