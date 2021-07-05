import torch
import pprint

from mphrqe import similarity, evaluation, loss
from mphrqe.data.loader import QueryGraphBatch

from mpqe_model import mpqe

def evaluate(
    data_loader: torch.utils.data.DataLoader[QueryGraphBatch],
    model_instance: mpqe,
    loss_function: loss.QueryEmbeddingLoss,
    similarity_function: similarity.Similarity,
):

    model_instance.eval()
    evaluator = evaluation.RankingMetricAggregator()
    val_loss = torch.zeros(size=tuple())

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        #For each batch of queries do:
        for batch in data_loader:
            # Get query embeddings
            out = model_instance(batch.edge_index, batch.edge_type, batch.entity_ids, batch.graph_ids)
            
            # Compute pair-wise scores between individual queries and all other nodes
            scores = similarity_function(out, model_instance.node_embeddings)

            # Filter duplicate targets
            targets = torch.unique(batch.targets, dim=1)

            # Compute loss based on scores
            loss = loss_function(scores, targets)
            val_loss += loss * scores.shape[0]
            evaluator.process_scores_(scores=scores, targets=targets)
        
        return dict(
            loss=val_loss.item() / len(data_loader),
            **evaluator.finalize(),
        )