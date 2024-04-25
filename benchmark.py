# %% Import
from torch_geometric.datasets import TUDataset
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# %% load dataset
dataset = TUDataset(root='.', name='MUTAG', use_node_attr=True)

# %% Convert dataset to NetworkX graphs and hash it with weisfeiler_lehman_graph_hash
training_graphs = []
for data in dataset:
    # Extract graph information
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    x = data.x

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes with attributes
    for i in range(x.size(0)):
        G.add_node(i, attr_dict={'feat': x[i].tolist()})

    # Add edges with attributes
    for j in range(edge_index.size(1)):
        src, dst = edge_index[0][j].item(), edge_index[1][j].item()
        attr = {'feat': edge_attr[j].tolist()} if edge_attr is not None else {}
        G.add_edge(src, dst, attr_dict=attr)

    training_graphs.append(G)

training_data_graph_hash = [nx.weisfeiler_lehman_graph_hash(G) for G in training_graphs]

# %% Define a function to compute novelty and uniqueness
def novelty_and_uniqueness(generated_graphs):
    global training_data_graph_hash

    generated_data_graph_hash = [nx.weisfeiler_lehman_graph_hash(G) for G in generated_graphs]

    # Compute novelty and uniqueness
    novel = [hash for hash in generated_data_graph_hash if not hash in training_data_graph_hash]
    precent_novel = len(novel)/len(generated_data_graph_hash)

    unique = list(set(generated_data_graph_hash))
    pecent_unique = len(unique)/len(generated_data_graph_hash)

    novel_and_unique = [hash for hash in unique if not hash in training_data_graph_hash]
    pecent_novel_and_unique = len(novel_and_unique)/len(generated_data_graph_hash)

    return precent_novel, pecent_unique, pecent_novel_and_unique

def node_degree_histogram(graphs):
    degrees = []
    for G in graphs:
        degrees.extend([d for n, d in G.degree()])
    plt.hist(degrees, bins=20)
    plt.xlabel('Node degree')
    plt.ylabel('Number of nodes')
    plt.title('Histogram of node degrees in the dataset')
    plt.show()


def clustering_coefficient_histogram(graphs):
    clustering_coefficients = []
    for G in graphs:
        clustering_coefficients.append(nx.average_clustering(G))
    plt.hist(clustering_coefficients, bins=20)
    plt.xlabel('Average clustering coefficient')
    plt.ylabel('Number of graphs')
    plt.title('Histogram of average clustering coefficient in the dataset')
    plt.show()

def eigenvector_centrality_histogram(graphs):
    eigenvector_centralities = []
    for G in graphs:
        eigenvector_centralities.extend(list(nx.eigenvector_centrality(G).values()))
    plt.hist(eigenvector_centralities, bins=20)
    plt.xlabel('Eigenvector centrality')
    plt.ylabel('Number of nodes')
    plt.title('Histogram of eigenvector centralities in the dataset')
    plt.show()
    
if __name__ == "__main__":
    from ErdösRényi import generate_erdos_graphs
    generated_graphs = generate_erdos_graphs(1000)

    node_degree_histogram(training_graphs)
    clustering_coefficient_histogram(training_graphs)
    eigenvector_centrality_histogram(training_graphs)


    #precent_novel, pecent_unique, pecent_novel_and_unique = novelty_and_uniqueness(generated_graphs)

    #print(f"Novel: {precent_novel}")
    #print(f"Unique: {pecent_unique}")
    #print(f"Novel and Unique: {pecent_novel_and_unique}")


# %%
