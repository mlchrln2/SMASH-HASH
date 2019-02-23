import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
import pdb
import collections
import time
import numpy as np

from grid_graph import grid_graph
from coarsening import coarsen
from coarsening import lmax_L
from coarsening import perm_data
from coarsening import rescale_L

if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

# Construct graph
t_start = time.time()
grid_side = 28
number_edges = 8
metric = 'euclidean'
A = grid_graph(grid_side,number_edges,metric) # create graph of Euclidean grid

# Compute coarsened graphs
coarsening_levels = 4
L, perm = coarsen(A, coarsening_levels)

# Compute max eigenvalue of graph Laplacians
lmax = []
for i in range(coarsening_levels):
    lmax.append(lmax_L(L[i]))
print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

# # Reindex nodes to satisfy a binary tree structure
# train_data = perm_data(train_data, perm)
# val_data = perm_data(val_data, perm)
# test_data = perm_data(test_data, perm)

# print(train_data.shape)
# print(val_data.shape)
# print(test_data.shape)
