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
import pathlib
import random
from mphrqe.data import loader
from mphrqe import similarity, evaluation, loss

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch_scatter import scatter, scatter_add
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from custom_query import query_descriptions, Query, Query_desc, Query_batch

# Remove later:
import pprint

#from custom_query import query_batch //Add this if data from custom_query is needed instead of aifb
debug_statements = False
################################# Load Data #################################

# datasets = loader.get_query_datasets(data_root=pathlib.Path("/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mphrqe/data/binaryQueries"), 
# train=[loader.Sample("/1hop/*", 1000)], 
# validation=[loader.Sample("/1hop/*", 167)], 
# test = [loader.Sample("/1hop/*", 1000)])


# Loading binary queries directly as data loaders
dataloaders = loader.get_query_data_loaders(batch_size=32, 
                                            num_workers=2, 
                                            data_root=pathlib.Path("/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mphrqe/data/binaryQueries"),
                                            train=[loader.Sample("/1hop/", 1000)], 
                                            validation=[loader.Sample("/1hop/*", 167)], 
                                            test = [loader.Sample("/1hop/*", 1000)])

loaders = dataloaders[0]
# print("Number of train batches: ", len(loaders["train"]))
# print("Number of train batches: ", len(loaders["test"]))
# print("Number of train batches: ", len(loaders["validation"]))

single_train_batch = next(iter(loaders["train"]))
#single_train_batch = loaders["train"].__getitem__(2)
if debug_statements: print(single_train_batch)

query_batch = Query_batch(
                global_entitiy_ids = single_train_batch.entity_ids,
                edge_index = single_train_batch.edge_index, 
                edge_type = single_train_batch.edge_type, 
                targets = single_train_batch.targets, 
                batch_ids = single_train_batch.graph_ids
                )

#############################################################################

### Description of the entire graph data ###
edge_index = torch.tensor(
            [[0, 1, 2, 2, 5, 4, 5, 9 , 9, 9 , 8 , 13, 14],
             [1, 2, 3, 5, 4, 6, 7, 12, 8, 11, 10, 14, 15]
            ])


edge_type = torch.tensor(
            [ 1, 3, 1, 3, 5, 4, 3, 1 , 1, 1 , 4 , 1 , 2]
            )

# For AIFB:
# 2601 total nodes (from entoid)
# 19 total relations (reltoid)

# For test custom_query:
# 16 total nodes
# 6 total relations

# +1 for variable and +1 for target
num_nodes = 2639 + 2
num_relations = 18


### Utility functions ###########################

## To get uniform embedding weights of custom range
def uniform_embeddings(num_nodes, emb_dim, device=None):
    uniform_distribution = Uniform(torch.tensor([-200.0]), torch.tensor([200.0]))

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

def initialize_embeddings(num_nodes, emb_dim) -> nn.Parameter:
    embeddings = nn.Parameter(torch.empty(num_nodes,emb_dim))
    stdv = 100
    nn.init.xavier_normal_(embeddings.data)
    return embeddings
#################################################   

# Model parameters
embedding_dim = 128
num_bases = None

# Create initial embedding = 2D tensor of shape (num_nodes + 2,embedding_dim) 
# "+2" representing variables and target
# final shape = (18, 4) with index 16 representing variables & index 17 representing target
node_embeddings = uniform_embeddings(num_nodes, embedding_dim)

class QueryEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_relations, num_bases):
        super(QueryEmbeddingModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.num_bases = num_bases if num_bases else num_relations
        
        self.rgcn = RGCNConv(in_channels=self.embedding_dim, out_channels=self.embedding_dim, num_relations=self.num_relations,
                              num_bases=None)
        self.rgcn2 = RGCNConv(in_channels=self.embedding_dim, out_channels=self.embedding_dim, num_relations=self.num_relations,
                              num_bases=None)
        self.node_embeddings = initialize_embeddings(num_nodes, embedding_dim)
        

    def forward(self, edge_index, edge_type, entity_ids, batch_ids = None):
        selected_node_embeddings: torch.float64
        selected_node_embeddings = self.node_embeddings[entity_ids.type(torch.long)]

        # Run the embeddings through RGCNConv
        query_node_embeddings = self.rgcn(selected_node_embeddings, edge_index, edge_type) 
        query_node_embeddings = self.rgcn2(query_node_embeddings, edge_index, edge_type)

        if debug_statements: print("Query node embedding shape: ", query_node_embeddings.shape)
        # Pooling the nodes in each query by summing 
        query_embeddings = scatter(query_node_embeddings, batch_ids, dim=0, reduce="sum")
        return query_embeddings


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = QueryEmbeddingModel(embedding_dim, num_relations, num_bases)

# Optimizer + loss functions 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#score_function = nn.CosineSimilarity(dim=0)
similarity_function = similarity.DotProductSimilarity() # from mphrqe
loss_function = loss.BCEQueryEmbeddingLoss() # from mphrqe


# To get single batch for testing
single_batch = next(iter(loaders["train"]))

def train(use_single_batch):
    model.train()
    train_evaluator = evaluation.RankingMetricAggregator()
    epoch_loss = torch.zeros(size=tuple())

    # This is hardcoded because it is done in QueryGraphBatch (collate_query_data())
    # https://github.com/HyperQueryEmbedding/hqe/blob/ee20f71fd7cfb32ea82daf99b8004b3f223e5111/src/mphrqe/data/loader.py#L354
    # remapped_edge_indices = torch.tensor(
    #                         [[0, 8, 3, 8, 6, 7, 8],
    #                          [8, 9, 8, 9, 8, 8, 9]])
    # This is an illustration of the remapping: https://drive.google.com/drive/u/0/folders/1CKz9xZIkgqylX0BsK1JJtIjx08rw6Vmj
    if use_single_batch:
        optimizer.zero_grad()

        out = model(single_batch.edge_index, single_batch.edge_type, single_batch.entity_ids, single_batch.graph_ids)

        # Input to this should be [32,4] (number of queries,embedding_dim) 
        scores = similarity_function(out, model.node_embeddings)

        #query_targets = select_embeddings_by_index(query_batch.targets)
        # We calculate loss by comparing query embedding to the embedding of the target
        #query_score = similarity_function(x=out, y=query_targets)

        loss = loss_function(scores, single_batch.targets)

        # (This will also have negative samples added later)
        #loss = torch.clamp(1 - test_scores, min=0)

        # Not sure what operation should be here, using mean() for now
        #loss = loss.mean()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach() * scores.shape[0]
        train_evaluator.process_scores_(scores=scores, targets=single_batch.targets)
        #print(evaluator.finalize())

        if debug_statements: 
            print("\nRGCN Gradient after backward:\n", model.rgcn.weight.grad[:10]) 
            print("\nGradient after backward shape:\n", model.rgcn.weight.grad.shape) 
            print("\nnode_embeddings Gradient after backward:\n", node_embeddings.grad[:10]) 

        return dict(
            loss=epoch_loss.item(),
            **train_evaluator.finalize(),
        )
    else:
        #For each batch of queries do:
        for batch in loaders["train"]:
            optimizer.zero_grad()

            out = model(batch.edge_index, batch.edge_type, batch.entity_ids, batch.graph_ids)
            scores = similarity_function(out, model.node_embeddings)

            loss = loss_function(scores, batch.targets)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * scores.shape[0]
            train_evaluator.process_scores_(scores=scores, targets=batch.targets)

        return dict(
            loss=epoch_loss.item() / len(loaders["train"]),
            **train_evaluator.finalize(),
        )


def train_for_epochs(epochs, use_single_batch):

    initial_loss = train(use_single_batch)
    for epoch in range(0, epochs):
        loss = train(use_single_batch)
        print(f'Epoch: {epoch:03d}, Loss: {loss["loss"]:.5f}')
        #pprint.pprint(loss)
    loss = train(use_single_batch)
    print("\nAfter training 100 epochs stats:")
    pprint.pprint(loss)
    print("\nInitial stats:")
    pprint.pprint(initial_loss)



#evaluation.evaluate(loaders["train"],model,similarity_function,loss_function)
#print(train(True))

def custom_loss(scores, targets):
    query_id, target_entity_id = targets
    negative_target_entity_id = torch.empty(len(query_id), dtype=torch.long)
    
    # Get random negative entity ids for each query
    for i in range(len(query_id)):
        x = random.randint(0, len(model.node_embeddings)-1)
        while (x == target_entity_id[i]):
            x = random.randint(0, len(model.node_embeddings)-1) 

        negative_target_entity_id[i] = x


    positive_scores = scores[query_id, target_entity_id]
    negative_scores = scores[query_id, negative_target_entity_id]
    # (This will also have negative samples added later)
    loss = 1 - positive_scores.sum() + negative_scores.sum()
    loss = torch.clamp(loss , min=0)
    return loss

initial_node_embeddings = copy.deepcopy(model.node_embeddings)
train_for_epochs(100, False)

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