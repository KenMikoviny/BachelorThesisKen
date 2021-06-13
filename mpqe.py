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
from custom_query import query_descriptions, Query, Query_desc, query_batch


### Description of the entire graph data ###
edge_index = torch.tensor(
            [[0, 1, 2, 2, 5, 4, 5, 9 , 9, 9 , 8 , 13, 14],
             [1, 2, 3, 5, 4, 6, 7, 12, 8, 11, 10, 14, 15]
            ])


edge_type = torch.tensor(
            [ 1, 3, 1, 3, 5, 4, 3, 1 , 1, 1 , 4 , 1 , 2]
            )

# +1 for variable and +1 for target
num_nodes = 16 + 2
num_relations = 6


### Utility functions ###########################

## To get uniform embedding weights of custom range
def uniform_embeddings(num_nodes, emb_dim, device=None):
    uniform_distribution = Uniform(torch.tensor([-20.0]), torch.tensor([20.0]))

    # Generate random center between -20 and 20
    node_embeddings = uniform_distribution.sample((num_nodes, int(emb_dim))).squeeze(-1)
    if device:
        node_embeddings = node_embeddings.to(device)
    node_embeddings.requires_grad = True

    return node_embeddings

## Select node embeddings by index
def select_embeddings_by_index(indices):
    embeddings = torch.empty(size=(len(indices),embedding_dim))
    for i in range(len(embeddings)):
        embeddings[i] = node_embeddings[indices[i]]
    
    return embeddings


#################################################   

# Model parameters
embedding_dim = 4
num_bases = 4

# Create initial embedding = 2D tensor of shape (num_nodes + 2,embedding_dim) 
# "+2" representing variables and target
# final shape = (18, 4) with index 16 representing variables & index 17 representing target
node_embeddings = uniform_embeddings(num_nodes, embedding_dim)
initial_node_embeddings = copy.deepcopy(node_embeddings)

class QueryEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_relations, num_bases):
        super(QueryEmbeddingModel, self).__init__()
        
        self.rgcn = RGCNConv(in_channels=embedding_dim, out_channels=embedding_dim, num_relations=num_relations,
                              num_bases=num_bases)
        

    def forward(self, query_node_embeddings, edge_index, edge_type, batch_ids = None):
        # Run the embeddings through RGCNConv
        query_node_embeddings = self.rgcn(query_node_embeddings, edge_index, edge_type)   

        # Pooling the nodes in each query by summing 
        query_embeddings = scatter(query_node_embeddings, batch_ids, dim=0, reduce="sum")
        print("\n \n Pooled query embeddings at the end of forward(): \n", query_embeddings)
        return query_embeddings




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = QueryEmbeddingModel(embedding_dim, num_relations, num_bases)

# Adding node embeddings to model parameters so weights can get trained
model_parameters = list(model.parameters())
model_parameters.append(node_embeddings)
optimizer = torch.optim.Adam(model_parameters, lr=0.01, weight_decay=0.0005)
score_function = nn.CosineSimilarity(dim=0)

def train():

    #For each batch of queries do:
    query_node_embeddings = select_embeddings_by_index(query_batch.global_entitiy_ids)

    # This is hardcoded because it is done in QueryGraphBatch (collate_query_data())
    # https://github.com/HyperQueryEmbedding/hqe/blob/ee20f71fd7cfb32ea82daf99b8004b3f223e5111/src/mphrqe/data/loader.py#L354
    remapped_edge_indices = torch.tensor(
                            [[0, 8, 3, 8, 6, 7, 8],
                             [8, 9, 8, 9, 8, 8, 9]])
    # This is an illustration of the remapping: https://drive.google.com/drive/u/0/folders/1CKz9xZIkgqylX0BsK1JJtIjx08rw6Vmj

    model.train()
    optimizer.zero_grad()
    print("\n\n Input to the forward() function is:")
    print("query_node_embeddings:\n",query_node_embeddings)
    print("remmaped_edge_indices:\n",remapped_edge_indices)
    print("query_batch.edge_type:\n",query_batch.edge_type)
    print("query_batch.batch_ids:\n",query_batch.batch_ids)
    out = model(query_node_embeddings, remapped_edge_indices, query_batch.edge_type, query_batch.batch_ids)


    query_targets = select_embeddings_by_index(query_batch.targets)
    # We calculate loss by comparing query embedding to the embedding of the target
    query_score = score_function(out, query_targets)

    # (This will also have negative samples added later)
    loss = torch.clamp(1 - query_score, min=0)

    # Not sure what operation should be here, using mean() for now
    loss = loss.mean()

    print("\n\nloss: ", loss)
    loss.backward()
    optimizer.step()
    # return loss.item(), out
    return loss


train()


# Uncomment the block below to see:
# if we run train for multiple epochs the loss gets minimized so everything should be working for now


#Train for 100 epochs to get trained node embedding of size (data.num_nodes, embedding_dim)
# for epoch in range(1, 101):
#     loss = train()
# print("\n\nnode_embeddings after training: ", node_embeddings)
# print("initial_node_embeddings: ", initial_node_embeddings)





########################### Ignore below:  #####################################
# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(node_embeddings, data.edge_index, data.edge_type, batch_ids).argmax(dim=-1)
#     train_acc = pred[data.train_idx].eq(data.train_y).to(torch.float).mean()
#     test_acc = pred[data.test_idx].eq(data.test_y).to(torch.float).mean()
#     return train_acc.item(), test_acc.item()

# Train for 50 epochs to get trained node embedding of size (data.num_nodes, embedding_dim)
# for epoch in range(1, 51):
#     loss = train()
#   train_acc, test_acc = test()
#   print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
#         f'Test: {test_acc:.4f}')
# print("\n\nnode_embeddings after training: ", node_embeddings)
# print("initial_node_embeddings: ", initial_node_embeddings)