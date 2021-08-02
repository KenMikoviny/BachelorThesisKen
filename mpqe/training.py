""" Code used for training the model """
import torch
import copy
import sys
sys.path.append("..")
from mphrqe import similarity, evaluation, loss
from mphrqe.data import QueryGraphBatch, mapping

from mpqe_model import mpqe
from evaluation import evaluate
from utility_methods import transfer_model_parameters, save_obj

def train(
    data_loaders: torch.utils.data.DataLoader[QueryGraphBatch],
    model: mpqe,
    eval_model: mpqe,
    loss_function: loss.QueryEmbeddingLoss,
    similarity_function: similarity.Similarity,
    optimizer: torch.optim.Optimizer,
    train_on_types: bool,
    save_obj_name: str = None,
):
    """ A single training loop iteration, handles training on entities as well as training on types """
    model.train()
    train_evaluator = evaluation.RankingMetricAggregator()
    epoch_loss = torch.zeros(size=tuple())

    # If the model trains on types instead of entities:
    if train_on_types:
        #For each batch of queries do:
        for batch in data_loaders["train"]:
            # Zero out the gradients
            optimizer.zero_grad()

            # Convert ids in edge_index and targets to type ids 
            entity_type_ids = copy.deepcopy(model.entity_type_ids)
            type_edge_index = batch.edge_index.detach().clone()
            type_targets = batch.targets.detach().clone()
            type_entity_ids = batch.entity_ids.detach().clone()
            for i in range(len(type_edge_index[0])):
                type_edge_index[0][i] = entity_type_ids[type_edge_index[0][i]]
                type_edge_index[1][i] = entity_type_ids[type_edge_index[1][i]]
            
            for i in range(len(type_targets[1])):
                type_targets[1][i] = entity_type_ids[type_targets[1][i]]


            for i in range(len(type_entity_ids)):
                #target
                if(type_entity_ids[i] == torch.tensor(target_id)):
                    type_entity_ids[i] = type_target_id
                #variable
                elif(type_entity_ids[i] == torch.tensor(variable_id)):
                    type_entity_ids[i] = type_variable_id
                else:
                    type_entity_ids[i] = entity_type_ids[type_entity_ids[i]]

            # Get query embeddings using remapped batch attributes
            out = model(type_edge_index, batch.edge_type, type_entity_ids, batch.graph_ids)

            # Compute pair-wise scores between individual queries and all other nodes
            scores = similarity_function(out, model.node_embeddings)
            
            # Filter duplicate targets
            type_targets = torch.unique(type_targets, dim=1)

            # Compute loss based on scores
            loss = loss_function(scores, type_targets)

            # Backpropagate & update weights
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach() * scores.shape[0]
            train_evaluator.process_scores_(scores=scores, targets=type_targets)

            return dict(
                loss=epoch_loss.item() / len(data_loaders["train"]),
                **train_evaluator.finalize(),
            )

    # If the model trains on entities:
    else:
        #For each batch of queries do:
        for batch in data_loaders["train"]:

            # Zero out the gradients
            optimizer.zero_grad()

            # Get query embeddings
            out = model(batch.edge_index, batch.edge_type, batch.entity_ids, batch.graph_ids)

            # Compute pair-wise scores between individual queries and all other nodes
            scores = similarity_function(out, model.node_embeddings)

            # Filter duplicate targets
            targets = torch.unique(batch.targets, dim=1)

            # Compute loss based on scores
            loss = loss_function(scores, targets)

            # Backpropagate & update weights
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach() * scores.shape[0]
            train_evaluator.process_scores_(scores=scores, targets=targets)

        ## Evaluation
        transfer_model_parameters(model, eval_model, True)
        # Evaluate on entities with weights + parameters from the model that was trained on entities
        eval_model_loss = evaluate(
            data_loader=data_loaders["test"],
            model_instance=eval_model,
            loss_function=loss_function,
            similarity_function=similarity_function,  
        )

        return dict(
            loss=epoch_loss.item() / len(data_loaders["train"]),
            **train_evaluator.finalize(),
        ), eval_model_loss

# Training on type embeddings and evaluating on entities each epoch 
def train_on_types_and_evaluate_on_entities(
    epochs: int,
    data_loaders: torch.utils.data.DataLoader[QueryGraphBatch],
    model_instance: mpqe,
    evaluation_model_instance: mpqe,
    loss_function: loss.QueryEmbeddingLoss,
    similarity_function: similarity.Similarity,
    optimizer_instance: torch.optim.Optimizer,
):
    """
    Trains a model instance on types while outputing evaluation loss using an entity model instance

    :param epochs: number of training epochs
    :param data_loader: data loader
    :param model_instance: model instance that gets trained on type embeddings
    :param evaluation_model_instance: model instance that gets used for evaluating on entity embeddings
    :param loss_function: loss function
    :param similarity_function: similarity function
    :param optimizer_instance: optimizer instance
    """ 

    print("Training on type embeddings and evaluating on entity embeddings each epoch")

    global target_id
    global variable_id
    global type_variable_id
    global type_target_id

    target_id = mapping.get_entity_mapper().highest_entity_index + 1
    variable_id = mapping.get_entity_mapper().highest_entity_index + 2
    type_variable_id = len(torch.unique(model_instance.entity_type_ids)) -1 
    type_target_id = len(torch.unique(model_instance.entity_type_ids)) - 2

    results_dict = {}
    entity_results_dict = {}

    for epoch in range(1, epochs + 1):
        type_loss = train(
            data_loaders = data_loaders,
            model = model_instance,
            eval_model = evaluation_model_instance,
            loss_function = loss_function,
            similarity_function = similarity_function,
            optimizer = optimizer_instance,
            train_on_types=True
            )

        results_dict[epoch] = type_loss

        transfer_model_parameters(model_instance, evaluation_model_instance, False)
        
        # Evaluate on entities with weights + parameters from the model that was trained on entities
        entity_model_loss = evaluate(
            data_loader=data_loaders["test"],
            model_instance=evaluation_model_instance,
            loss_function=loss_function,
            similarity_function=similarity_function,  
        )
        entity_results_dict[epoch] = entity_model_loss

        print(f'Epoch: {epoch:03d}, Type embedding model loss: {type_loss["loss"]:.5f}, Entity embedding model loss: {entity_model_loss["loss"]:.5f}')
    
    # Save entity_model_loss during training on types
    save_obj(entity_results_dict, "transfered_entity_results_dict")
    return results_dict

def train_for_epochs(
    epochs: int,
    data_loaders: torch.utils.data.DataLoader[QueryGraphBatch],
    model_instance: mpqe,
    evaluation_model_instance: mpqe,
    loss_function: loss.QueryEmbeddingLoss,
    similarity_function: similarity.Similarity,
    optimizer_instance: torch.optim.Optimizer,
    train_on_types: bool = False,
    save_obj_name: str = None,
):
    """
    Trains a model instance on types while outputing evaluation loss using an entity model instance

    :param epochs: number of training epochs
    :param data_loaders: data loaders
    :param model_instance: model instance that gets trained on type embeddings
    :param evaluation_model_instance: model instance that gets used for evaluating on entity embeddings
    :param loss_function: loss function
    :param similarity_function: similarity function
    :param optimizer_instance: optimizer instance
    :param save_obj_name: name of the saved object on disk
    """ 
    
    results_dict_train = {}
    results_dict_eval = {}


    # If the model trains on type embeddings + we have a entity evaluation model instance then: (see train_on_types_and_evaluate_on_entities())
    if(evaluation_model_instance is not None and train_on_types):
        results_dict = train_on_types_and_evaluate_on_entities(
            epochs=epochs,
            data_loaders=data_loaders,
            model_instance=model_instance,
            evaluation_model_instance=evaluation_model_instance,
            loss_function=loss_function,
            similarity_function=similarity_function,
            optimizer_instance=optimizer_instance,
        )

    else:
        for epoch in range(1, epochs + 1): 
            train_loss, eval_loss = train(
                data_loaders = data_loaders,
                model = model_instance,
                eval_model = evaluation_model_instance,
                loss_function = loss_function,
                similarity_function = similarity_function,
                optimizer = optimizer_instance,
                train_on_types=False,
                save_obj_name=save_obj_name,
                )
            results_dict_train[epoch] = train_loss
            results_dict_eval[epoch] = eval_loss
            if(epoch % 100 == 0):
                save_obj(results_dict_train, str(save_obj_name) + "_train") if save_obj_name else None 
                save_obj(results_dict_eval, str(save_obj_name) + "_eval") if save_obj_name else None 
            if(epoch == 1000):
                save_obj(results_dict_eval, str(save_obj_name) + "_1000") if save_obj_name else None 

            print(f'Epoch: {epoch:03d}, Training loss: {train_loss["loss"]:.5f}, Evaluation loss: {eval_loss["loss"]:.5f}') 

    return


