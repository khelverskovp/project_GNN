# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# %% Interactive plots
plt.ion() # Enable interactive plotting
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

# %% Device
device = 'cpu'

# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)

# %% compute graph statistics

# compute histograms of node degree in the dataset
degrees = []
for data in dataset:
    degrees.append(data.num_edges)
plt.hist(degrees, bins=20)
plt.xlabel('Node degree')
plt.ylabel('Number of nodes')
plt.title('Histogram of node degrees in the dataset')
plt.show()
# %%
