from gnn_graph_classification import SimpleGNN, GraphVAE, GaussianPrior
import torch
import matplotlib.pyplot as plt
import networkx as nx


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
                    encoder = encoder)

state_dict = torch.load('model.pt')
model.load_state_dict(state_dict)



