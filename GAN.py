import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
# import dataloader
from torch_geometric.loader import DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MUTAG dataset
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

dataloader = DataLoader(dataset, batch_size=188)

data = next(iter(dataloader))
print(data)

print(data[0].num_nodes)
print(data[0].num_edges)

