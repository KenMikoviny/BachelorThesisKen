import torch
import torch.nn as nn
import pprint
import pathlib

from mphrqe import similarity, evaluation, loss
from mphrqe.data import loader

from utility_methods import get_entity_type_ids
from mpqe_model import mpqe
from training import train_for_epochs
from evaluation import evaluate

################################# Load Data #################################

# Loading binary queries directly as data loaders
dataloaders = loader.get_query_data_loaders(batch_size=32, 
                                            num_workers=2, 
                                            data_root=pathlib.Path("/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mphrqe/data/binaryQueries"),
                                            train=[loader.Sample("*", 2000)], 
                                            validation=[loader.Sample("*", 2000)], 
                                            test = [loader.Sample("*", 2000)])

# 3 loaders, loaders["train"], loaders["test"], loaders["validation"]
loaders = dataloaders[0]

# Path used to save model that has been trained on type embeddings
model_save_path = "/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mpqe/data/saved_models/state_dict_type_model.pt"

aifb_entity_id_path = "/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mpqe/data/triple_split_AIFB/entity_id_typing.txt"
#TODO remove later
test_path = "/mnt/c/Users/Sorys/Desktop/Thesis/BachelorThesisKen/mpqe/data/triple_split_AIFB/test.txt"

# Get list of entity type id's by converting the string representation to an int, order corresponds to global ids
entity_type_ids = get_entity_type_ids(aifb_entity_id_path)

#############################################################################

############################## Model parameters #############################
# For AIFB:
# 2601 total nodes (from entoid)
# 19 total relations (reltoid)

# TODO get these from files somehow
# apparently 2639=target, 2640=variable?
num_nodes = 2641 
num_relations = 19

embedding_dim = 128
num_bases = None
learning_rate = 0.0001

training_epochs = 1000

#############################################################################

################## Optimizer, Similarity, Loss, Evaluator ###################

similarity_function = similarity.DotProductSimilarity() # from mphrqe

loss_function = loss.BCEQueryEmbeddingLoss() # from mphrqe

#############################################################################


model_instance = mpqe(embedding_dim, num_relations, num_nodes, num_bases)
optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)

# #Training
# training_results = train_for_epochs(
#     epochs=1000,
#     data_loader=loaders["train"],
#     model_instance=model_instance,
#     loss_function=loss_function,
#     similarity_function=similarity_function,
#     optimizer_instance=optimizer,
#     )

# # Evaluation
# for loader in loaders:
#     if loader == "train":
#         continue

#     results = evaluate(
#     data_loader=loaders[loader],
#     model_instance=model_instance,
#     loss_function=loss_function,
#     similarity_function=similarity_function,
#     )

#     if loader == "validation":
#         print("\nValidation set results:\n")
#         pprint.pprint(results)
#     if loader == "test":
#         print("\nTest set results:\n")
#         pprint.pprint(results)

##############################################################################
################# MPQE Model that trains on type embeddings ##################
##############################################################################

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
    epochs=training_epochs,
    data_loader=loaders["train"],
    model_instance=type_embedding_model_instance,
    evaluation_model_instance=None,
    loss_function=loss_function,
    similarity_function=similarity_function,
    optimizer_instance=optimizer,
    is_type_embeddings_model=True,
    save_path=model_save_path,
    )

print("\nFirst epoch:\n")
#pprint.pprint(training_results[0])
print("\nLast epoch:\n")
#pprint.pprint(training_results[training_epochs-1])

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
