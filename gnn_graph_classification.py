import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as td
from tqdm import tqdm
import networkx as nx
from ErdösRényi import N_distribution

# Interactive plots
plt.ion() # Enable interactive plotting
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()




class GaussianPrior(torch.nn.Module):
    def __init__(self, M):
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = torch.nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(M), requires_grad=False)
        
    def forward(self):
        return td.Independent(td.Normal(loc = self.mean, scale = self.std), 1)

class BernoulliDecoder(torch.nn.Module):
    def __init__(self, decoder_net):
        
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        
    def forward(self, zu, zv):
       
        # Concat the latent variables which are a single scaler tensor for each node
        #zu = zu.unsqueeze(0)  # Adds an extra dimension, making zu a 1-dimensional tensor
        #zv = zv.unsqueeze(0)  # Adds an extra dimension, making zv a 1-dimensional tensor

        concat = zu * zv 
        
        logits = self.decoder_net(concat)
        
        return td.Independent(td.Bernoulli(logits=logits), 0)

class GraphVAE(torch.nn.Module):
    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, encoder, decoder_net, prior):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        
        # Encoder 
        # It will output a mean and a log-variance for each graph
        self.encoder = encoder
        
        
        self.decoder = BernoulliDecoder(decoder_net=decoder_net)
        
        # Initialize prior distribution of latent space
        self.prior = prior

    # Function for sampling in the latent space
    def sample_latent(self, mu, logvar):
        """Sample from the latent space."""
        eps = torch.randn_like(logvar)
        Z = mu +  eps*logvar
        return Z
    
    def elbo(self, x):
 
        
        q = self.encoder(x)
        z = q.rsample()

        
        # True edges in the graphs
        true_log_probs = []
        target = torch.tensor(1.0)
        for edge in x.edge_index[:,]:
            zu = z[edge[0]]
            zv = z[edge[1]]
            log_prob = self.decoder(zu, zv).log_prob(target)
            true_log_probs.append(log_prob)
            
        false_log_probs = []
        for graph_idx in torch.unique(x["batch"]):
            graph_nodes = torch.where(x["batch"] == graph_idx)[0]

            all_graph_node_combinations = torch.combinations(graph_nodes, 2)
            random_permutation = torch.randperm(all_graph_node_combinations.size(0))
            all_graph_node_combinations = all_graph_node_combinations[random_permutation]# random suffle

            for combo in all_graph_node_combinations:
                
                # Skip if the edge exists
                if torch.any(torch.all(torch.unsqueeze(combo, dim=1) == x["edge_index"], dim=0)):
                    continue
                
                
                if len(false_log_probs) == len(true_log_probs):
                    break
                
                zu = z[combo[0]]
                zv = z[combo[1]]
                logp = self.decoder(zu, zv).log_prob(target)
                log_prob = torch.log(1 - torch.exp(logp)) # Remember we look for the probability of the edge not existing
                false_log_probs.append(log_prob)


        # Gather loss
        recon_loss = torch.mean(torch.stack(true_log_probs + false_log_probs))
        
       
        kl_div = td.kl_divergence(q, self.prior()).mean()
        
        return recon_loss - kl_div
    
    
    def forward(self, x):
        return -self.elbo(x)
    
    def sample(self, n = 1):
        
        sampled_graphs = []
        N_list = N_distribution.rvs(size=n)
        
        for N in N_list:
            sampled_nodes = self.prior().rsample((N,))

            all_samples_node_combinations = torch.combinations(torch.arange(N), 2)

            nodesu = sampled_nodes[all_samples_node_combinations[:, 0]]
            nodesv = sampled_nodes[all_samples_node_combinations[:, 1]]

            sampled_edges_bool_idx = torch.squeeze(self.decoder(nodesu, nodesv).sample()).bool().numpy()

            sampled_edges = all_samples_node_combinations[sampled_edges_bool_idx]

            A = torch.zeros((N, N)) # Adjacency matrix
            A[sampled_edges[:, 0], sampled_edges[:, 1]] = 1
            A[sampled_edges[:, 1], sampled_edges[:, 0]] = 1

            G = nx.from_numpy_array(A.numpy())

            sampled_graphs.append(G)

        return sampled_graphs    
    

def train(model, optimizer, dataloader, epochs, device):
    
    
    model.train()
    
    total_steps = len(dataloader)*epochs
    progress_bar = tqdm(range(total_steps), desc='Training')

    
    for epoch in range(epochs):
        data_iter = iter(dataloader)
        
        for x in data_iter:
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())



class SimpleGNN(torch.nn.Module):
    """Simple graph neural network for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        state_dim : Dimension of the node states
        num_message_passing_rounds : Number of message passing rounds
    """

    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, output_dim=1):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.output_dim = output_dim

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU()
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])


        # State output network
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.state_dim, self.output_dim*2)
        )

    def forward(self, batch_data):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """

        x = batch_data.x
        edge_index = batch_data.edge_index
        batch = batch_data.batch
        
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        state = self.input_net(x)

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            state = state + self.update_net[r](aggregated)

        out = self.output_net(state)

        mean, logvar = torch.chunk(out, 2, dim=-1)
        
        return td.Independent(td.Normal(mean, torch.exp(logvar)), 1)
    
# %% Set up the model, loss, and optimizer etc.
# Instantiate the model


if __name__ == '__main__':
    
    device = 'cpu'

    # Load data
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)
    node_feature_dim = 7

    # Split into training and validation
    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

    # Create dataloader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=100)
    validation_loader = DataLoader(validation_dataset, batch_size=44)
    test_loader = DataLoader(test_dataset, batch_size=44)

    
        
    state_dim = 32
    num_message_passing_rounds = 8
    node_feature_dim = 7

    latent_dim = 2
    

    # Encoder
    encoder = SimpleGNN(node_feature_dim = node_feature_dim, 
                        state_dim=state_dim, 
                        num_message_passing_rounds = num_message_passing_rounds, 
                        output_dim=latent_dim)

    # Decoder Network
    decoder_net = torch.nn.Sequential(torch.nn.Linear(latent_dim, latent_dim), 
                                    torch.nn.ReLU(), 
                                    torch.nn.Linear(latent_dim, 1),
                                    torch.nn.Sigmoid())
    
    # Define the prior
    prior = GaussianPrior(latent_dim)

    # Define model
    model = GraphVAE(node_feature_dim=node_feature_dim, 
                     state_dim=state_dim, 
                     num_message_passing_rounds=num_message_passing_rounds, 
                     decoder_net=decoder_net, 
                     prior = prior, 
                     encoder = encoder).to(device)
    
    #state_dict = torch.load('model.pt')
    #model.load_state_dict(state_dict)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#%%
    # Train the model
    train(model, optimizer, train_loader, epochs=1000, device=device)

    # %% Save final model.

    torch.save(model.state_dict(), 'model.pt')
    
    
    # %%
