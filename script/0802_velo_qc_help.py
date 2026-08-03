import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml
from scipy.sparse import issparse
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def calculate_spearman(high_dim, low_dim, sample_size=1000):
    """Calculates Spearman CPD like in Kobbak (2019) using a random subset of 1.000 cells."""
    # Copied code from the papers repository
    # Sample data for pairwise distance correlation calculation
    subset = np.random.choice(high_dim.shape[0], size=sample_size, replace=False)

    # Extrace indices and handle sparse matrices in case they are sparse
    hd_subset = high_dim[subset, :]
    ld_subset = low_dim[subset, :]

    if issparse(hd_subset):
        hd_subset = hd_subset.toarray()
    if issparse(ld_subset):
        ld_subset = ld_subset.toarray()

    # Compute pairwise distances in original and embedded spaces
    d1 = pdist(hd_subset)  # Pairwise distances in high-dimensional space
    d2 = pdist(ld_subset)  # Pairwise distances in low-dimensional space

    # Calculate Spearman correlation between the pairwise distances
    rho = spearmanr(d1[:, None], d2[:, None]).correlation

    return rho

def calculate_knn(high_dim, low_dim, k=10):
    """Calculates KNN accoriding to Kobbak(2019)"""
    # Copied code from the papers repository
    # Adapt for spare matrices
    if issparse(high_dim):
        high_dim = high_dim.tocsr()
    if issparse(low_dim):
        low_dim = low_dim.tocsr()
    # Compute local nearest-neighbor indices for high-dimensional data
    nbrs1 = NearestNeighbors(n_neighbors=k).fit(high_dim)
    ind1 = nbrs1.kneighbors(return_distance=False)  # Nearest neighbor indices for X

    # Compute local nearest-neighbor indices for low-dimensional embedding
    nbrs2 = NearestNeighbors(n_neighbors=k).fit(low_dim)
    ind2 = nbrs2.kneighbors(return_distance=False)  # Nearest neighbor indices for Z

    # Calculate mean nearest-neighbor overlap (local)
    intersections = 0.0
    for i in range(high_dim.shape[0]):
        # Count intersection of nearest neighbors in original and embedded spaces
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn = intersections / high_dim.shape[0] / k  # Normalize by total samples and neighbors
        
    return mnn

def calculate_knc(high_dim, low_dim, labels, k = 4):
    """Calculates KNC accoriding to Kobbak(2019)"""
    # Copied code from the papers repository
    # Adapt for spare matrices
    if issparse(high_dim):
        high_dim = high_dim.tocsr()
    if issparse(low_dim):
        low_dim = low_dim.tocsr()

    # Compute class centroids for high-dimensional and embedded spaces
    cl, cl_inv = np.unique(labels, return_inverse=True)  # Unique class labels and their indices
    C = cl.size  # Number of unique classes
    # Ensure that k is not to big
    k = min(k, C -1)
    mu1 = np.zeros((C, high_dim.shape[1]))  # Centroids for X
    mu2 = np.zeros((C, low_dim.shape[1]))  # Centroids for Z
    for c in range(C):
        # Calculate centroids for each class in X and Z
        mu1[c, :] = np.mean(high_dim[cl_inv == c, :], axis=0)
        mu2[c, :] = np.mean(low_dim[cl_inv == c, :], axis=0)

    # Compute global nearest-neighbor indices based on class centroids
    nbrs1 = NearestNeighbors(n_neighbors=k).fit(mu1)
    ind1 = nbrs1.kneighbors(return_distance=False)  # Nearest neighbor indices for class centroids in X
    nbrs2 = NearestNeighbors(n_neighbors=k).fit(mu2)
    ind2 = nbrs2.kneighbors(return_distance=False)  # Nearest neighbor indices for class centroids in Z

    # Calculate mean nearest-neighbor overlap (global, class-based)
    intersections = 0.0
    for i in range(C):
        # Count intersection of nearest neighbors in class centroid space
        intersections += len(set(ind1[i]) & set(ind2[i]))
    knc = intersections / C / k  # Normalize by number of classes and neighbors

    return knc

def calculate_trustworthiness(high, low):
    return trustworthiness(high, low, n_neighbors=10)

def calculate_continuity(high, low):
    return trustworthiness(low, high, n_neighbors=10)