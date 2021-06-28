"""
For testing on a single batch, put in run_model.py 
"""

# test_batch = next(iter(loaders["train"]))
# test_edge_index = test_batch.edge_index.detach().clone()

# type_edge_index = test_batch.edge_index.detach().clone()
# type_targets = test_batch.targets.detach().clone()
# type_entity_ids = test_batch.entity_ids.detach().clone()

# print(type_entity_ids)
# print(type_edge_index)
# print(test_batch.edge_type)
# print(type_targets)

# for i in range(len(type_edge_index[0])):
#     type_edge_index[0][i] = entity_type_ids[type_edge_index[0][i]]
#     type_edge_index[1][i] = entity_type_ids[type_edge_index[1][i]]

# for i in range(len(type_targets[1])):
#     type_targets[1][i] = entity_type_ids[type_targets[1][i]]


# for i in range(len(type_entity_ids)):
#     #target
#     if(type_entity_ids[i] == torch.tensor(2639)):
#         type_entity_ids[i] = 7
#     #variable
#     elif(type_entity_ids[i] == torch.tensor(2640)):
#         type_entity_ids[i] = 6
#     else:
#         type_entity_ids[i] = entity_type_ids[type_entity_ids[i]]
# print("xxxxxxxxxx")
# print(type_entity_ids)
# print(type_edge_index)
# print(test_batch.edge_type)
# print(type_targets)

# for i in range(len(test_edge_index[0])):
#     test_edge_index[0][i] = entity_type_ids[test_edge_index[0][i]]
#     test_edge_index[1][i] = entity_type_ids[test_edge_index[1][i]]
# test_targets = test_batch.targets.detach().clone()
# for i in range(len(test_targets[1])):
#     test_targets[1][i] = entity_type_ids[test_targets[1][i]]