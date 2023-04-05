import numpy as np
import pandas as pd
from ts_lsh.common import euclidean, distance_matrix, matrix_histogram

def hist_global(mat, nbins=20):
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


def original_vs_hash(dataset : np.array, lsh, fn_distance = euclidian, nbins=20):
  if not lsh.batch:
    hash = np.array([lsh.hash(k) for k in dataset])
  else:
    hash = lsh.hash(dataset)
  mat = distance_matrix(dataset, fn_distance)
  mat_hash = distance_matrix(hash, fn_distance)

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

def compare_distributions(dataset : np.array, lsh, fn_distance = euclidian, nbins=20):
  if not lsh.batch:
    hash = np.array([lsh.hash(k) for k in dataset])
  else:
    hash = lsh.hash(dataset)
  mat = distance_matrix(dataset, fn_distance)
  mat_hash = distance_matrix(hash, fn_distance)

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
