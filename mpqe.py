import argparse
import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.data as geom_data
import numpy as np
import copy

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch_scatter import scatter, scatter_add
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from custom_query import query_descriptions, Query, Query_desc


### Description of the entire graph data ###
edge_index = torch.tensor(
            [[0, 1, 2, 2, 5, 4, 5, 9 , 9, 9 , 8 , 13, 14],
             [1, 2, 3, 5, 4, 6, 7, 12, 8, 11, 10, 14, 15]
            ])


edge_type = torch.tensor(
            [ 1, 3, 1, 3, 5, 4, 3, 1 , 1, 1 , 4 , 1 , 2]
            )

num_nodes = 16
num_relations = 6


### Utility functions ###

## To get uniform embedding weights of custom range
def uniform_embeddings(num_nodes, emb_dim, device=None):
    uniform_distribution = Uniform(torch.tensor([-100.0]), torch.tensor([100.0]))

    # Generate random center between -100 and 100
    node_embeddings = uniform_distribution.sample((num_nodes, int(emb_dim))).squeeze(-1)
    if device:
        node_embeddings = node_embeddings.to(device)
    node_embeddings.requires_grad = True

    return node_embeddings


# Gets edge index and edge types from a query description
def process_query(query_description):
    query_copy = copy.deepcopy(query_description)
    edge_type = []
    
    for x in range(len(query_copy)):
        edge_type.append(query_copy[x].pop(1))
      
    edge_index = query_copy
    return edge_index, edge_type

    



# Model parameters
embedding_dim = 4
num_bases = 4

# Create initial embedding = 2D tensor of shape (num_nodes + 2,embedding_dim) 
# "+2" representing variables and target
# final shape = (18, 4) with index 16 representing variables & index 17 representing target
untrained_embedding = uniform_embeddings(num_nodes, embedding_dim)




# Processing query descriptions into processed queries
queries = []
for query in query_descriptions:
    edge_index, edge_type = process_query(query.edges)
    queries.append(Query(edge_index, edge_type, query.target))

print(queries)

# Initializing batch_ids randomly for testing:
batch_ids = torch.from_numpy(np.random.randint(4, size=num_nodes))


# Neural network with 2 RGCNConv layers, input = node embeddings
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.node_embeddings = []
        
        self.conv1 = RGCNConv(in_channels=embedding_dim, out_channels=embedding_dim, num_relations=num_relations,
                              num_bases=num_bases)
        self.conv2 = RGCNConv(in_channels=embedding_dim, out_channels=embedding_dim, num_relations=num_relations,
                              num_bases=num_bases)
        

    def forward(self, embedding, edge_index, edge_type, batch_ids = None):
        x = F.relu(self.conv1(embedding, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)   
        
        # For producing embeddings
        # Here we save node embeddings for all nodes = shape [23606,2]
        self.node_embeddings = x
        
        # Pooling using torch_scatter sum
        if(batch_ids is not None):
            self.node_embeddings = scatter(x, batch_ids, dim=0, reduce="sum")
        
        # For classification
        x = F.log_softmax(x, dim=1)
        
        
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)




# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(untrained_embedding, data.edge_index, data.edge_type, batch_ids)
#     loss = F.nll_loss(out[data.train_idx], data.train_y)
#     loss.backward()
#     optimizer.step()
#     return loss.item(), out


# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(untrained_embedding, data.edge_index, data.edge_type, batch_ids).argmax(dim=-1)
#     train_acc = pred[data.train_idx].eq(data.train_y).to(torch.float).mean()
#     test_acc = pred[data.test_idx].eq(data.test_y).to(torch.float).mean()
#     return train_acc.item(), test_acc.item()

# # Train for 50 epochs to get trained node embedding of size (data.num_nodes, embedding_dim)
# for epoch in range(1, 10):
#     loss, embedding = train()
#     train_acc, test_acc = test()
#     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
#           f'Test: {test_acc:.4f}')
    
# print(model.node_embeddings)
# print(model.node_embeddings.shape)