import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
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

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
	
batch_size=128
dataset=MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
img,truth=0,0

# Construct graph
t_start = time.time()
grid_side = 28
number_edges = 8
metric = 'euclidean'
A = grid_graph(grid_side,number_edges,metric) # create graph of Euclidean grid

# Compute coarsened graphs
coarsening_levels = 2
L, perm = coarsen(A, coarsening_levels)

# Compute max eigenvalue of graph Laplacians
lmax = []
for i in range(coarsening_levels):
    lmax.append(lmax_L(L[i]))
print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

train_data=[]
train_labels=[]
for i in dataloader:
	train_data,train_labels=i
	break

print(train_data.shape)
# pdb.set_trace()
# Reindex nodes to satisfy a binary tree structure
train_data = perm_data(train_data.reshape(128,-1), perm)

print(train_data)
print(train_data.shape)
