# %% Imports
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import numpy as np
from scipy.stats import rv_discrete
import networkx as nx
import matplotlib.pyplot as plt

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

# %% What is needed to sample graphs:
N_distribution = rv_discrete(values=(range(30), normalized_graph_node_count_histogram))

#link_probability = np.mean(graph_density)
link_probability = []
for i in range(10, 28 + 1):
    link_probability.append(np.mean(graph_density[graph_node_count == i]))

# %% Sample a graph
def generate_erdos_graphs(n_graphs):
    global link_probability
    global N_distribution

    graphs = []
    N_samples = N_distribution.rvs(size=n_graphs)
    for N in N_samples:
        G = nx.erdos_renyi_graph(N, link_probability[N-10])
        graphs.append(G)
    return graphs

if __name__ == '__main__':
    graphs = generate_erdos_graphs(1)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for G in graphs:
        nx.draw(G, with_labels=True, ax=ax1)
        ax1.set_title('Graph')

        adj_matrix = nx.adjacency_matrix(G)

        # Convert the adjacency matrix to a numpy array
        adj_matrix_array = adj_matrix.toarray()

        # Plot the adjacency matrix
        ax2.imshow(adj_matrix_array, cmap='gray')
        ax2.set_title('Adjacency Matrix')

        plt.savefig('erdos_graph.png')

        plt.show()