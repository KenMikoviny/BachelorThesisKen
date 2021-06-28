import torch
import pprint

from mphrqe import similarity, evaluation, loss
from mphrqe.data.loader import QueryGraphBatch

from mpqe_model import mpqe
from evaluation import evaluate

def train(
    data_loader: torch.utils.data.DataLoader[QueryGraphBatch],
    model: mpqe,
    loss_function: loss.QueryEmbeddingLoss,
    similarity_function: similarity.Similarity,
    optimizer: torch.optim.Optimizer,
    train_on_types:bool,
    save_path:None,
):

    model.train()
    train_evaluator = evaluation.RankingMetricAggregator()
    epoch_loss = torch.zeros(size=tuple())

    # If the model trains on types instead of entities:
    if train_on_types:
        #For each batch of queries do:
        for batch in data_loader:

            # Zero out the gradients
            optimizer.zero_grad()

             # Convert ids in edge_index and targets to type ids 
            entity_type_ids = model.entity_type_ids
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
                if(type_entity_ids[i] == torch.tensor(2639)):
                    type_entity_ids[i] = 7
                #variable
                elif(type_entity_ids[i] == torch.tensor(2640)):
                    type_entity_ids[i] = 6
                else:
                    type_entity_ids[i] = entity_type_ids[type_entity_ids[i]]
            # Get query embeddings
            out = model(type_edge_index, batch.edge_type, type_entity_ids, batch.graph_ids)

            # Compute pair-wise scores between individual queries and all other nodes
            scores = similarity_function(out, model.node_embeddings)

            # Compute loss based on scores
            loss = loss_function(scores, type_targets)

            # Backpropagate & update weights
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach() * scores.shape[0]
            train_evaluator.process_scores_(scores=scores, targets=type_targets)

        return dict(
            loss=epoch_loss.item() / len(data_loader),
            **train_evaluator.finalize(),
        )

    # If the model trains on entities:
    else:
        #For each batch of queries do:
        for batch in data_loader:

            # Zero out the gradients
            optimizer.zero_grad()

            # Get query embeddings
            out = model(batch.edge_index, batch.edge_type, batch.entity_ids, batch.graph_ids)

            # Compute pair-wise scores between individual queries and all other nodes
            scores = similarity_function(out, model.node_embeddings)

            # Compute loss based on scores
            loss = loss_function(scores, batch.targets)

            # Backpropagate & update weights
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach() * scores.shape[0]
            train_evaluator.process_scores_(scores=scores, targets=batch.targets)

        return dict(
            loss=epoch_loss.item() / len(data_loader),
            **train_evaluator.finalize(),
        )

def train_and_evaluate_on_entities(
    epochs: int,
    data_loader: torch.utils.data.DataLoader[QueryGraphBatch],
    model_instance: mpqe,
    evaluation_model_instance: mpqe,
    loss_function: loss.QueryEmbeddingLoss,
    similarity_function: similarity.Similarity,
    optimizer_instance: torch.optim.Optimizer,
    is_type_embeddings_model=False,
    save_path=None,
):

    print("Training on type embeddings and evaluating on entity embeddings each epoch")
    results_dict = {}
    # Initialize empty node embeddings
    entity_node_embeddings = torch.empty(evaluation_model_instance.node_embeddings.shape)

    for epoch in range(0, epochs):
        type_loss = train(
            data_loader = data_loader,
            model = model_instance,
            loss_function = loss_function,
            similarity_function = similarity_function,
            optimizer = optimizer_instance,
            train_on_types=is_type_embeddings_model,
            save_path=save_path,
            )
        results_dict[epoch] = type_loss

        # Save trained model except node embeddings
        model_dict = model_instance.state_dict()
        del model_dict['node_embeddings']
        torch.save(model_dict,save_path)

        # Load saved model params 
        pretrained_dict = torch.load(save_path)

        trained_type_embeddings = model_instance.node_embeddings 

        # #for testing:
        # print("\nEntity type ids: ", model_instance.entity_type_ids)
        # print("\ntrained node emb:",trained_type_embeddings[:100])
        # print("\nEntity node embeddings before fill: ", entity_node_embeddings[:10])


        # Fill entity_node_embeddings with pretrained ones according to entity types
        # shape goes from [num_unique_classes + 2, emb_dim] -> [num_entities + 2, emb_dim]
        for i in range(len(model_instance.entity_type_ids)):
            entity_node_embeddings[i] = trained_type_embeddings[model_instance.entity_type_ids[i]]

        # Add the entity_node_embeddings to the dictionary before loading the weights
        pretrained_dict['node_embeddings'] = entity_node_embeddings
        evaluation_model_instance.load_state_dict(pretrained_dict)
        
        # #for testing
        # print("\nEntity node embeddings after fill: ", entity_node_embeddings[:100])

        entity_model_loss = evaluate(
            data_loader=data_loader,
            model_instance=evaluation_model_instance,
            loss_function=loss_function,
            similarity_function=similarity_function,  
        )

        print(f'Epoch: {epoch:03d}, Type embedding model loss: {type_loss["loss"]:.5f}, Entity embedding model loss: {entity_model_loss["loss"]:.5f}')
    return results_dict

def train_for_epochs(
    epochs: int,
    data_loader: torch.utils.data.DataLoader[QueryGraphBatch],
    model_instance: mpqe,
    evaluation_model_instance: mpqe,
    loss_function: loss.QueryEmbeddingLoss,
    similarity_function: similarity.Similarity,
    optimizer_instance: torch.optim.Optimizer,
    is_type_embeddings_model=False,
    save_path=None,
):
    
    results_dict = {}

    # If the model trains on type embeddings + we have a entity evaluation model instance then: (see train_and_evaluate_on_entities())
    if(is_type_embeddings_model and evaluation_model_instance):
        results_dict = train_and_evaluate_on_entities(
            epochs=epochs,
            data_loader=data_loader,
            model_instance=model_instance,
            evaluation_model_instance=evaluation_model_instance,
            loss_function=loss_function,
            similarity_function=similarity_function,
            optimizer_instance=optimizer_instance,
            is_type_embeddings_model=is_type_embeddings_model,
            save_path=save_path,
        )

    else:
        for epoch in range(0, epochs): 
            loss = train(
                data_loader = data_loader,
                model = model_instance,
                loss_function = loss_function,
                similarity_function = similarity_function,
                optimizer = optimizer_instance,
                train_on_types=is_type_embeddings_model,
                save_path=save_path,
                )
            results_dict[epoch] = loss
            print(f'Epoch: {epoch:03d}, Loss: {loss["loss"]:.5f}') 

    return results_dict