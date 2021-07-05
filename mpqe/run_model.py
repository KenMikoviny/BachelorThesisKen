import torch
import torch.nn as nn
import pprint
import pathlib

from mphrqe import similarity, evaluation, loss
from mphrqe.data import loader, mapping

from utility_methods import get_entity_type_ids, save_obj, load_obj, transfer_model_parameters
from mpqe_model import mpqe
from training import train_for_epochs
from evaluation import evaluate

################################# Load Data #################################

# #NOTE: when changing dataset also change entoid and reltoid in mphrqe/data/mappings, this is due to how mapping.py retrieves highest entity index

# train_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"), loader.resolve_sample("/2hop/*:atmost1000"), 
#                 loader.resolve_sample("/2i/*:atmost1000"), loader.resolve_sample("/2i-1hop/*:atmost1000"),
#                 loader.resolve_sample("/3hop/*:atmost1000"), loader.resolve_sample("/3i/*:atmost1000"),
#                 ]
# valid_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"), loader.resolve_sample("/2hop/*:atmost1000"), loader.resolve_sample("/3hop/*:atmost1000"), 
#                 loader.resolve_sample("/3i/*:atmost1000"),
#                 ]
# test_samples = [loader.resolve_sample("/1hop/*:atmost1000"), loader.resolve_sample("/1hop-2i/*:atmost1000"), loader.resolve_sample("/2hop/*:atmost1000"), loader.resolve_sample("/3hop/*:atmost1000"), 
#                loader.resolve_sample("/3i/*:atmost1000"),
#                ]


# # Loading binary queries directly as data loaders
# dataloaders = loader.get_query_data_loaders(batch_size=16, 
#                                             num_workers=2, 
#                                             data_root=pathlib.Path("/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mpqe/data/binary_queries/binary_queries_AIFB"),
#                                             train=train_samples, 
#                                             validation=valid_samples, 
#                                             test=test_samples)

# 3 loaders, loaders["train"], loaders["test"], loaders["validation"]
dataloaders = load_obj("dataloaders")
loaders = dataloaders[0]

aifb_entity_id_path = "/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mpqe/data/triple_split_AIFB/entity_id_typing.txt"
am_entity_id_path = "/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mpqe/data/triple_split_AM/entity_id_typing.txt"
mutag_entity_id_path = "/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mpqe/data/triple_split_MUTAG/entity_id_typing.txt"


# Get list of entity type id's by converting the string representation to an int, order corresponds to global ids
entity_type_ids = get_entity_type_ids(aifb_entity_id_path)


#############################################################################

############################## Model parameters #############################
# For AIFB:
# 2601 total nodes (from entoid)
# 19 total relations (reltoid)
num_nodes = mapping.get_entity_mapper().highest_entity_index + 3
num_relations = mapping.get_relation_mapper().get_largest_forward_relation_id() - 2

embedding_dim = 128
num_bases = None
learning_rate = 0.0001

training_epochs = 10

#############################################################################

################## Optimizer, Similarity, Loss, Evaluator ###################

similarity_function = similarity.DotProductSimilarity() # from mphrqe

loss_function = loss.BCEQueryEmbeddingLoss() # from mphrqe

#############################################################################


model_instance = mpqe(embedding_dim, num_relations, num_nodes, num_bases)
optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)

#Training
training_results = train_for_epochs(
    epochs=training_epochs,
    data_loader=loaders["train"],
    model_instance=model_instance,
    evaluation_model_instance=None,
    loss_function=loss_function,
    similarity_function=similarity_function,
    optimizer_instance=optimizer,
    )

# Evaluation
for loader in loaders:
    if loader == "train":
        continue

    results = evaluate(
    data_loader=loaders[loader],
    model_instance=model_instance,
    loss_function=loss_function,
    similarity_function=similarity_function,
    )

    if loader == "validation":
        save_obj(results, "validation_results_entities")
        print("\nValidation set results:\n")
        pprint.pprint(results)
    if loader == "test":
        save_obj(results, "test_results_entities")
        print("\nTest set results:\n")
        pprint.pprint(results)

save_obj(training_results, "training_results")

#############################################################################
################ MPQE Model that trains on type embeddings ##################
#############################################################################

# +1 for variable and +1 for target
num_types = len(torch.unique(entity_type_ids)) + 2 

similarity_function = similarity.DotProductSimilarity() # from mphrqe

loss_function = loss.BCEQueryEmbeddingLoss() # from mphrqe

#############################################################################

type_embedding_model_instance = mpqe(
    embedding_dim=embedding_dim, 
    num_relations=num_relations, 
    num_nodes=num_types, 
    num_bases=num_bases, 
    entity_type_ids=entity_type_ids,
    )

# Used for evaluation using entities instead of types
evaluation_mpqe_model_instance = mpqe(
    embedding_dim=embedding_dim, 
    num_relations=num_relations, 
    num_nodes=num_nodes, 
    num_bases=num_bases, 
    entity_type_ids=entity_type_ids,
    )

optimizer = torch.optim.Adam(type_embedding_model_instance.parameters(), lr=learning_rate)

#Training
training_results = train_for_epochs(
    epochs=10,
    data_loader=loaders["train"],
    model_instance=type_embedding_model_instance,
    evaluation_model_instance=evaluation_mpqe_model_instance,
    loss_function=loss_function,
    similarity_function=similarity_function,
    optimizer_instance=optimizer,
    is_type_embeddings_model=True,
    )

# # Evaluation
# for loader in loaders:
#     if loader == "train":
#         continue

#     results = evaluate(
#     data_loader=loaders[loader],
#     model_instance=type_embedding_model,
#     loss_function=loss_function,
#     similarity_function=similarity_function,
#     )

#     if loader == "validation":
#         print("\nValidation set results:\n")
#         pprint.pprint(results)
#     if loader == "test":
#         print("\nTest set results:\n")
#         pprint.pprint(results)


################################## Transfer weights + train ######################################
print("\nTransfering weights and starting training again")

combined_model_instance = mpqe(
    embedding_dim=embedding_dim, 
    num_relations=num_relations, 
    num_nodes=num_nodes, 
    num_bases=num_bases, 
    )

# Transfering weights from mpqe that was trained on types
transfer_model_parameters(
    from_model=type_embedding_model_instance, 
    to_model=combined_model_instance)

#Training
combined_model_training_results = train_for_epochs(
    epochs=training_epochs,
    data_loader=loaders["train"],
    model_instance=combined_model_instance,
    evaluation_model_instance=None,
    loss_function=loss_function,
    similarity_function=similarity_function,
    optimizer_instance=optimizer,
    )

save_obj(combined_model_training_results, "combined_model_training_results")