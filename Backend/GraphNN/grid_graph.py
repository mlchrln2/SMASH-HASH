import sklearn
import sklearn.metrics
import scipy.sparse, scipy.sparse.linalg  # scipy.spatial.distance
import numpy as np
import torch

def grid_graph(grid_side,number_edges,metric):
    """Generate graph of a grid"""
    z = grid(grid_side)
    dist, idx = distance_sklearn_metrics(z, k=number_edges, metric=metric)
    A = adjacency(dist, idx)
    # print("nb edges: ",A.nnz)
    return A


def grid(m, dtype=torch.float32):
    """Return coordinates of grid points"""
    M = m**2
    x = torch.linspace(0,1,m, dtype=dtype, requires_grad=False)
    y = torch.linspace(0,1,m, dtype=dtype, requires_grad=False)
    xx, yy = torch.meshgrid(x, y)
    z = torch.empty((M,2), dtype=dtype, requires_grad=False)
    z[:,0] = xx.reshape(M)
    z[:,1] = yy.reshape(M)
    return z


def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute pairwise distances"""
    d = sklearn.metrics.pairwise.pairwise_distances(z, metric=metric, n_jobs=-2)
    # k-NN
    idx = np.argsort(d)[:,1:k+1]
    d.sort()
    d = d[:,1:k+1]
    return d, idx


def adjacency(dist, idx):
    """Return adjacency matrix of a kNN graph"""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0
    assert dist.max() <= 1

    # Pairwise distances
    sigma2 = np.mean(dist[:,-1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix
    I = torch.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections
    W.setdiag(0)

    # Undirected graph
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W


    