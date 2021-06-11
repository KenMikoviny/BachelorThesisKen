
import torch
import numpy as np


# Custom query description class (unprocessed query)
class Query_desc:

    def __init__(self, edges, target):
        self.edges = edges
        self.target = target

# Custom query class
class Query:

    def __init__(self, edge_index, edge_type, target):
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.target = target

# Custom query_batch class (only necessary for the test dataset)
class Query_batch:

    def __init__(self, global_entitiy_ids, edge_index, edge_type, targets, batch_ids):
        self.global_entitiy_ids = global_entitiy_ids
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.targets = targets
        self.batch_ids = batch_ids



# Creating 3 custom queries
# index 16 represents variables and index 17 represents target!


### Query 0 ### 
### 3 nodes, 2 edges ###
### node 0 being the root, node 1 being variable, node 2 = target ###

query0_description = Query_desc(
                        edges =
                        torch.tensor(
                            [[0 , 1, 16],
                            [16, 3, 17]]),
                        target = torch.tensor(2))


### Query 1 ### 
### 3 nodes, 2 edges ###
### node 5 being the root, node 4 being variable, node 6 = target ###

query1_description =  Query_desc(
                        edges =
                        torch.tensor(
                            [[5 , 5, 16],
                            [16, 4, 17]]),
                        target = torch.tensor(6))
    

### Query 2 ### 
### 4 nodes, 3 edges ###
### node 12 and 11 being the root, node 9 being variable, node 8 = target ###

query2_description =  Query_desc(
                        edges =
                        torch.tensor(
                            [[12, 1, 16],
                            [11, 1, 16],
                            [16,  1, 17]]),
                        target = torch.tensor(8))

query_descriptions = (query0_description,query1_description,query2_description)






############################# Utility functions ###########################

# Gets edge index and edge types a from a query description
def process_query_edges(query_description): 
    return query_description[:,0::2], query_description[:,1]

    
# Creates batch ids from a list of queries for batching
def get_batch_ids(query_list): 
    batch_ids = []

    # For each unique node in a query append 1x the index of the specific query (index in the given list of queries)
    for query_index in range(len(query_list)):
        for unique_node in torch.unique(query_list[query_index].edge_index):
            batch_ids.append(query_index)
    return batch_ids




#############################################################################






#### Processing query descriptions into processed queries (Query class),this is only necessary when working with the test data #####
queries = []
for query in query_descriptions:
    edge_index, edge_type = process_query_edges(query.edges)
    queries.append(Query(edge_index, edge_type, query.target))


# Getting batch ids of the query batch
query_batch_ids = get_batch_ids(queries)

# Getting targets of all queries in the batch as a tensor
flat_targets = []
for query in queries:
    flat_targets.append(query.target)

# Constructing the edge_index of the current query batch
all_query_edges = [edge for query in queries for edge in query.edge_index]
query_edge_index = torch.empty(size=(2, len(all_query_edges)))
for i in range(len(all_query_edges)):
    query_edge_index[0][i] = all_query_edges[i][0]
    query_edge_index[1][i] = all_query_edges[i][1]

query_batch = Query_batch(
                global_entitiy_ids = torch.tensor([node for query in queries for node in torch.unique(query.edge_index)]),
                edge_index = query_edge_index, 
                edge_type = torch.tensor([edge_type for query in queries for edge_type in query.edge_type]), 
                targets = torch.tensor(flat_targets), 
                batch_ids = torch.tensor(query_batch_ids)
                )
# Note for myself: edge_index = [edge for query in queries for edge in query.edge_index]
# Corresponds to:
# edge_index = []
# for query in queries:
#     for edge in query.edge_index:
#         edge_index.append(edge)


attrs = vars(query_batch)
# {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
# now dump this in some way or another
print("\nquery batch info: \n",',\n \n '.join("%s: %s " % item for item in attrs.items()))