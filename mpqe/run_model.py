import torch
import torch.nn as nn
import pprint
import pathlib
import sys
import copy
sys.path.append("..")
from BachelorThesisKen.hqe.src.mphrqe import similarity, loss
from BachelorThesisKen.hqe.src.mphrqe.data import mapping

print("First imports done", flush=True)
from utility_methods import get_entity_type_ids, save_obj, load_obj, transfer_model_parameters
from mpqe_model import mpqe
from training import train_for_epochs
from evaluation import evaluate
from config import aifb_entity_id_path, save_embeddings, save_results, get_new_dataloaders, tsne_seed
from get_dataloaders import get_dataloaders

################################# Load Data #################################
print("Loading data", flush=True)
# Generate new dataloaders and samples if needed, else load new ones
get_dataloaders() if get_new_dataloaders else None

# 3 loaders, loaders["train"], loaders["test"], loaders["validation"]
dataloaders = load_obj("dataloaders")
loaders = dataloaders[0]

# Get list of entity type id's by converting the string representation to an int, order corresponds to global ids
entity_type_ids = get_entity_type_ids(aifb_entity_id_path)
#############################################################################




############################## Model & parameters #############################
# For AIFB:
# 2601 total nodes (from entoid)
# 19 total relations (reltoid)
num_nodes = mapping.get_entity_mapper().highest_entity_index + 3 
num_relations = mapping.get_relation_mapper().get_largest_forward_relation_id() - 2 

embedding_dim = 128
num_bases = None
learning_rate = 0.0001

training_epochs = 3

entity_model_instance = mpqe(embedding_dim, num_relations, num_nodes, num_bases)
#############################################################################


####################### Optimizer, Similarity, Loss #########################

similarity_function = similarity.DotProductSimilarity() # from mphrqe

loss_function = loss.BCEQueryEmbeddingLoss() # from mphrqe

optimizer = torch.optim.Adam(entity_model_instance.parameters(), lr=learning_rate)
#############################################################################


################################ Training ###################################
print("Saving visualization", flush=True)
# Save embeddings and labels in a dictionary for visualization
save_dict = {'node_embeddings':entity_model_instance.node_embeddings, 'labels':entity_type_ids}
save_obj(save_dict, "entity_embeddings_before") if save_embeddings else None

print("Starting training", flush=True)
training_results = train_for_epochs(
    epochs=training_epochs,
    data_loader=loaders["train"],
    model_instance=entity_model_instance,
    evaluation_model_instance=None,
    loss_function=loss_function,
    similarity_function=similarity_function,
    optimizer_instance=optimizer,
    )

# Save embeddings and labels in a dictionary for visualization
save_dict = {'node_embeddings':entity_model_instance.node_embeddings, 'labels':entity_type_ids}
save_obj(save_dict, "entity_embeddings_after") if save_embeddings else None

# Save results of training
save_obj(training_results, "training_results") if save_results else None

# Evaluation
for loader in loaders:
    if loader == "train":
        continue

    results = evaluate(
    data_loader=loaders[loader],
    model_instance=entity_model_instance,
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
#############################################################################




#############################################################################
################# MPQE Model that trains on type embeddings #################
#############################################################################


###################### Models & Parameters & Optimizer ######################

# From entoid + 1 for target and 1 for variable
num_types = len(torch.unique(entity_type_ids))

# Used for training on type embeddings
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

# Used for training on entities after transfering parameters from type_embedding_model_instance
combined_model_instance = mpqe(
    embedding_dim=embedding_dim, 
    num_relations=num_relations, 
    num_nodes=num_nodes, 
    num_bases=num_bases, 
    )

type_model_optimizer = torch.optim.Adam(type_embedding_model_instance.parameters(), lr=learning_rate)
combined_model_optimizer = torch.optim.Adam(combined_model_instance.parameters(), lr=learning_rate)
#############################################################################


############################# Training on Types #############################

save_dict = {'node_embeddings':type_embedding_model_instance.node_embeddings, 'labels':list(range(num_types))}
save_obj(save_dict, "type_embeddings_before") if save_embeddings else None

training_results = train_for_epochs(
    epochs=100,
    data_loader=loaders["train"],
    model_instance=type_embedding_model_instance,
    evaluation_model_instance=evaluation_mpqe_model_instance,
    loss_function=loss_function,
    similarity_function=similarity_function,
    optimizer_instance=type_model_optimizer,
    )

save_dict = {'node_embeddings':type_embedding_model_instance.node_embeddings, 'labels':list(range(num_types))}
save_obj(save_dict, "type_embeddings_after") if save_embeddings else None

#############################################################################


################### Transfer weights + Train on Entities ####################
print("\nTransfering weights and starting training again", flush=True)

# Transfering weights from mpqe that was trained on types
transfer_model_parameters(
    from_model=type_embedding_model_instance, 
    to_model=combined_model_instance)


save_dict = {'node_embeddings':combined_model_instance.node_embeddings, 'labels':entity_type_ids}
save_obj(save_dict, "combined_embeddings_before") if save_embeddings else None

combined_model_training_results = train_for_epochs(
    epochs=training_epochs,
    data_loader=loaders["train"],
    model_instance=combined_model_instance,
    evaluation_model_instance=None,
    loss_function=loss_function,
    similarity_function=similarity_function,
    optimizer_instance=combined_model_optimizer,
    )

save_dict = {'node_embeddings':combined_model_instance.node_embeddings, 'labels':entity_type_ids}
save_obj(save_dict, "combined_embeddings_after") if save_embeddings else None

save_obj(combined_model_training_results, "combined_model_training_results") if save_results else None
#############################################################################