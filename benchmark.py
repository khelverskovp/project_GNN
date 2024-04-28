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
    novel = [hash for hash in generated_data_graph_hash if hash not in training_data_graph_hash]
    precent_novel = len(novel)/len(generated_data_graph_hash)

    unique = list(set(generated_data_graph_hash))
    pecent_unique = len(unique)/len(generated_data_graph_hash)

    novel_and_unique = [hash for hash in unique if hash not in training_data_graph_hash]
    pecent_novel_and_unique = len(novel_and_unique)/len(generated_data_graph_hash)

    return precent_novel, pecent_unique, pecent_novel_and_unique

def node_degree_histogram(ax, graphs):
    degrees = []
    for G in graphs:
        degrees.extend([d for n, d in G.degree()])
    ax.hist(degrees, bins=range(min(degrees), max(degrees) + 2), density=True, width=.8)
    ax.set_xlabel('Node degree')
    ax.set_ylabel('Number of nodes')
    ax.set_title('Histogram of node degrees in the dataset')
    


def clustering_coefficient_histogram(ax, graphs):
    clustering_coefficients = []
    for G in graphs:
        clustering_coefficients.extend(list(nx.clustering(G).values()))
    ax.hist(clustering_coefficients, bins=20, density=True)
    ax.set_xlabel('Average clustering coefficient')
    ax.set_ylabel('Number of graphs')
    ax.set_title('Histogram of average clustering coefficient in the dataset')
    

def eigenvector_centrality_histogram(ax, graphs):
    eigenvector_centralities = []
    for i, G in enumerate(graphs):
        try:
            eigenvector_centralities.extend(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
        except:
            print(f"Graph {i} is not connected, skipping eigenvector centrality calculation")
            

    ax.hist(eigenvector_centralities, bins=20, density=True)
    ax.set_xlabel('Eigenvector centrality')
    ax.set_ylabel('Number of nodes')
    ax.set_title('Histogram of eigenvector centralities in the dataset')
    
if __name__ == "__main__":
    from ErdösRényi import generate_erdos_graphs
    generated_erdos_graphs = generate_erdos_graphs(1000)

    precent_novel, pecent_unique, pecent_novel_and_unique = novelty_and_uniqueness(generated_erdos_graphs)

    print("Erdös-Rényi method")
    print(f"Novel: {precent_novel}")
    print(f"Unique: {pecent_unique}")
    print(f"Novel and Unique: {pecent_novel_and_unique}")
    print("\n")

    from load_model import model

    generated_GVAE_graphs = model.sample(1000)
    precent_novel, pecent_unique, pecent_novel_and_unique = novelty_and_uniqueness(generated_GVAE_graphs)

    print("GNN method")
    print(f"Novel: {precent_novel}")
    print(f"Unique: {pecent_unique}")
    print(f"Novel and Unique: {pecent_novel_and_unique}")

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    graphs_sets = [training_graphs, generated_erdos_graphs, generated_GVAE_graphs]
    
    for i in range(3):
        node_degree_histogram(axs[i, 0], graphs_sets[i])
        clustering_coefficient_histogram(axs[i, 1], graphs_sets[i])
        eigenvector_centrality_histogram(axs[i, 2], graphs_sets[i])

    # Add general column titles
    row_titles = ['Molecules', 'Erdös-Rényi', 'GVAE']
    #TODO: Add titles to the rows
    
    plt.tight_layout()
    plt.savefig('benchmark.png')