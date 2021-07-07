################################# Utility functions #################################
import os
import torch
import torch.nn as nn
import pickle
import torch
import copy

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

def get_entity_type_ids(file_path):
    """
    Returns a list of entity type ids, order corresponds to global entity id's

    :param file_path: path to the entity id typing.txt file

    :return: list of entity type ids
    """ 
    entity_type_ids = []
    string_entity_types = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip(os.linesep)
            string_entity_types.append(line.split()[1])

    # Get unique ID's as strings by converting to set, then convert the unique strings to integer id's with list(range(len()))
    unique_entity_types = sorted(set(string_entity_types))
    unique_entity_type_ids = list(range(len(unique_entity_types)))
    type_id_dict = dict(zip(unique_entity_types,unique_entity_type_ids))

    for entity in string_entity_types: 
        entity_type_ids.append(type_id_dict[entity])

    return torch.tensor(entity_type_ids)


def select_embeddings_by_index(embeddings, indices):
    """
    Returns embeddings from all node embeddings chosen by global id

    :param indices: list of global ids

    :return: selected node embeddings, shape (len(indices, embedding_dim)
    """ 
    selected_embeddings = torch.empty(size=(len(indices),embedding_dim))
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[indices[i]]

    return selected_embeddings


def initialize_embeddings(num_nodes, emb_dim) -> nn.Parameter:
    """
    Used to randomly initialize embeddings before training is done, uses nn.init.xavier_normal_

    :num_nodes: total number of nodes
    :emb_dim: embedding dimension

    :return: node embeddings, shape(num_nodes,emb_dim)
    """ 
    embeddings = nn.Parameter(torch.empty(num_nodes,emb_dim))
    nn.init.xavier_normal_(embeddings.data)
    return embeddings

# TODO maybe find better name for this
def transfer_model_parameters(
    from_model: torch.nn.Module, 
    to_model: torch.nn.Module, 
) -> torch.nn.Module:

    # Path used to save model that has been trained on type embeddings
    save_path = "/mnt/c/Users/Sorys/OneDrive/VU am/Thesis/BachelorThesisKen/mpqe/data/saved_models/state_dict_type_model.pt"

    # Initialize empty node embeddings
    entity_node_embeddings = torch.empty(to_model.node_embeddings.shape)

    # Save trained model except node embeddings
    model_dict = copy.deepcopy(from_model.state_dict())
    del model_dict['node_embeddings']
    torch.save(model_dict,save_path)

    # Load saved model params 
    pretrained_dict = torch.load(save_path)

    trained_type_embeddings = copy.deepcopy(from_model.node_embeddings)

    # Fill entity_node_embeddings with pretrained ones according to entity types
    # shape goes from [num_unique_classes + 2, emb_dim] -> [num_entities + 2, emb_dim]
    for i in range(len(from_model.entity_type_ids)):
        entity_node_embeddings[i] = trained_type_embeddings[from_model.entity_type_ids[i]]

    # Add the entity_node_embeddings to the dictionary before loading the weights
    pretrained_dict['node_embeddings'] = entity_node_embeddings
    to_model.load_state_dict(pretrained_dict)

def save_obj(obj, name ):
    with open('/mnt/c/Users/Sorys/OneDrive/VU am/Thesis/BachelorThesisKen/mpqe/data/saved_data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/mnt/c/Users/Sorys/OneDrive/VU am/Thesis/BachelorThesisKen/mpqe/data/saved_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#####################################################################################  