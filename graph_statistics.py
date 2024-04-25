# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import networkx as nx

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


# compute histograms of average clustering coefficient in the dataset for each graph
clustering_coefficients = []
for data in dataset:
    # compute clustering coefficient
    G = nx.Graph()
    G.add_edges_from(data.edge_index.T.cpu().numpy())
    clustering_coefficients.append(nx.average_clustering(G))
plt.hist(clustering_coefficients, bins=20)
plt.xlabel('Average clustering coefficient')
plt.ylabel('Number of graphs')
plt.title('Histogram of average clustering coefficient in the dataset')

# %%
