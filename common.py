import math
import numpy as np

def gram_schmidt(A):
  # From: https://zerobone.net/blog/cs/gram-schmidt-orthogonalization/
  (n, m) = A.shape
  for i in range(m):        
    q = A[:, i] # i-th column of A
        
    for j in range(i):
      q = q - np.dot(A[:, j], A[:, i]) * A[:, j]
        
    if np.array_equal(q, np.zeros(q.shape)):
      raise np.linalg.LinAlgError("The column vectors are not linearly independent")
        
    # normalize q
    q = q / np.sqrt(np.dot(q, q))
        
    # write the vector back in the matrix
    A[:, i] = q

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

def data_vs_hash_distance_matrices(dataset : np.array, lsh, fn_distance = euclidean, nbins=20):
  if not lsh.batch:
    hash = np.array([lsh.hash(k) for k in dataset])
  else:
    hash = lsh.hash(dataset)
  mat = distance_matrix(dataset, fn_distance)
  mat_hash = distance_matrix(hash, fn_distance)
  return mat, mat_hash

def DKL(mat : np.array, mat_hash : np.array, nbins=20):

  bins_d, freqs_d = matrix_histogram(mat, nbins)
  bins_h, freqs_h = matrix_histogram(mat_hash, nbins)

  return np.sum([(pd * np.log(1/ph)) - (pd * np.log(1/pd)) \
                 for pd, ph in zip(freqs_d, freqs_h) \
                 if pd > 0 and ph > 0])


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
