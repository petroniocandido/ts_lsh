import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ts_lsh.common import euclidean, distance_matrix, matrix_histogram, data_vs_hash_distance_matrices

def distance_distribution(mat, nbins=20):
  fig = plt.figure(layout='constrained', figsize=(15, 5))
  subfigs = fig.subfigures(1, 2, wspace=0.07)
  ax = subfigs[0].subplots(1,1)
  cols = ax.matshow(mat)
  subfigs[0].colorbar(cols)
  ax.set_title("Distance Matrix")
  m = np.min(mat[~np.eye(mat.shape[0],dtype=bool)]) # Todos os elementos exceto a diagnonal principal
  M = np.max(mat)
  bins, freq = matrix_histogram(mat, nbins)
  ax = subfigs[1].subplots(1,1)
  ax.bar(bins[:-1], freq)
  ax.set_title("Distance Distribution")

  
def data_vs_hash_distance_distribution(**kwargs):
  mat : np.array = kwargs.get('mat', None)
  mat_hash : np.array = kwargs.get('mat_hash', None)
  nbins : int = kwargs.get('nbins', 20)
  
  if mat is None and mat_hash is None:
    dataset : np.array = kwargs.get('dataset',None)
    lsh = kwargs.get('lsh',None)
    fn_distance = kwargs.get('fn_distance', euclidean)
    mat, mat_hash = data_vs_hash_distance_matrices(dataset, lsh, fn_distance, nbins)
  
  fig = plt.figure(layout='constrained', figsize=(10, 5))
  subfigs = fig.subfigures(1, 2, wspace=0.07)
  
  ax_mat = subfigs[0].subplots(2,1)
  cols = ax_mat[0].matshow(mat)
  cols = ax_mat[1].matshow(mat_hash)
  ax_mat[0].set_title("Distance Matrix")
  subfigs[0].colorbar(cols, ax=ax_mat)
  
  ax_dist = subfigs[1].subplots(2,1)

  bins, freq = matrix_histogram(mat, nbins)
  ax_dist[0].bar(bins[:-1], freq)
  ax_dist[0].set_title("Original Distance Distribution")

  bins_hash, freq_hash = matrix_histogram(mat_hash, nbins)
  ax_dist[1].bar(bins_hash[:-1], freq_hash)
  ax_dist[1].set_title("Embedding Distance Distribution")
  return mat, math_hash


def conditional_distance_distribution(dataset : np.array, lsh, fn_distance = euclidean, nbins=20, mat=None, mat_hash=None):
  mat : np.array = kwargs.get('mat', None)
  mat_hash : np.array = kwargs.get('mat_hash', None)
  nbins : int = kwargs.get('nbins', 20)
  
  if mat is None and mat_hash is None:
    dataset : np.array = kwargs.get('dataset',None)
    lsh = kwargs.get('lsh',None)
    fn_distance = kwargs.get('fn_distance', euclidean)
    mat, mat_hash = data_vs_hash_distance_matrices(dataset, lsh, fn_distance, nbins)
  
  fig = plt.figure(layout='constrained', figsize=(10, 5))
  subfigs = fig.subfigures(1, 2, wspace=0.07)
  
  bins, freqs = matrix_histogram(mat, nbins)

  dist = {bin : [] for bin in bins}
  n,m = mat.shape

  mat_ix = np.digitize(mat, bins)

  for i in range(n-2):
    for j in range(i+1,m-1):
      f = bins[mat_ix[i,j]-1]
      dist[f].append(mat_hash[i,j])
  
  fig, ax = plt.subplots(1,2, figsize=(10,5))

  _ = ax[0].barh(bins[:-1], freqs)
  ax[0].set_title("Original Distance distribution")
  ax[0].set_xlabel("Distance Frequency")
  ax[0].set_ylabel("Distance")
  _ = ax[1].boxplot([h for h in dist.values()][:-1],vert=False, \
                    positions=[round(k,2) for k in bins[:-1]])
  ax[1].set_title("Embedding Related Distances")
  ax[1].set_xlabel("Embedded Distances")
