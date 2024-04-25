# %% Imports
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import numpy as np
from scipy.stats import rv_discrete

# %% Load the MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG')
n_graphs = 188
data_loader = DataLoader(dataset, batch_size=n_graphs)
data = next(iter(data_loader))

# %% Calculate emperical node count distribution
graph_node_count_histogram = np.zeros(30)
graph_node_count = np.zeros(n_graphs)
for i in range(n_graphs):
    n_nodes = sum(data["batch"] == i)
    graph_node_count[i] = n_nodes
    graph_node_count_histogram[n_nodes] += 1

normalized_graph_node_count_histogram = graph_node_count_histogram / len(data["y"])

# %% Calculate emperical graph density distribution
graph_link_counts = np.zeros(188)
for link in data["edge_index"][0]:
    grapp_number = data["batch"][link]
    graph_link_counts[grapp_number] += 1

graph_possible_links = graph_node_count * (graph_node_count - 1) / 2

graph_density = graph_link_counts / graph_possible_links

# What is needed to sample graphs:
N_distribution = rv_discrete(values=(range(30), normalized_graph_node_count_histogram))
link_probability = np.mean(graph_density)

if __name__ == '__main__':
    sampled_N = N_distribution.rvs(size=1000)