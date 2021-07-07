import torch
from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch_scatter import scatter

from utility_methods import initialize_embeddings


class mpqe(torch.nn.Module):
    def __init__(self, embedding_dim, num_relations, num_nodes, num_bases, entity_type_ids=None):
        super(mpqe, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.num_bases = num_bases if num_bases else num_relations
        
        self.rgcn = RGCNConv(in_channels=self.embedding_dim, out_channels=self.embedding_dim, num_relations=self.num_relations,
                              num_bases=None)
        self.rgcn2 = RGCNConv(in_channels=self.embedding_dim, out_channels=self.embedding_dim, num_relations=self.num_relations,
                              num_bases=None)
        self.node_embeddings = initialize_embeddings(num_nodes, embedding_dim)

        # For training with entity type ids
        if(entity_type_ids is not None):
            self.entity_type_ids = entity_type_ids.type(torch.long)
        

    def forward(self, edge_index, edge_type, entity_ids, batch_ids = None):
        selected_node_embeddings: torch.float64
        selected_node_embeddings = self.node_embeddings[entity_ids]

        # Run the embeddings through RGCNConv
        query_node_embeddings = self.rgcn(selected_node_embeddings, edge_index, edge_type) 
        query_node_embeddings = self.rgcn2(query_node_embeddings, edge_index, edge_type)
        
        # Pooling the nodes in each query by summing 
        query_embeddings = scatter(query_node_embeddings, batch_ids, dim=0, reduce="sum")

        return query_embeddings