#!/usr/bin/env python3
"""
GNN encoders for query graphs with transfer learning support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from typing import Optional, Tuple

class GINEncoder(nn.Module):
    """Graph Isomorphism Network encoder"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        nn_first = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn_first))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            nn_hidden = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_hidden))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, edge_attr=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class SAGEEncoder(nn.Module):
    """GraphSAGE encoder"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # SAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, edge_attr=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class JoinOrderRanker(nn.Module):
    """
    Complete model for ranking join orders.
    Combines GNN encoder with order-specific scoring.
    """
    
    def __init__(self,
                 encoder_type: str = 'gin',
                 node_features: int = 128,
                 edge_features: Optional[int] = None,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 order_encoding_dim: int = 32,
                 dropout: float = 0.2,
                 freeze_encoder: bool = False):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder
        
        # Choose encoder
        if encoder_type == 'gin':
            self.encoder = GINEncoder(node_features, hidden_dim, num_layers, dropout)
        elif encoder_type == 'sage':
            self.encoder = SAGEEncoder(node_features, hidden_dim, num_layers, dropout)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Edge feature processing (if needed)
        self.edge_encoder = None
        if edge_features:
            self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        
        # Graph-level pooling
        self.pool = GlobalPooling(hidden_dim)
        
        # Join order encoding
        self.order_encoder = nn.Sequential(
            nn.Linear(order_encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final scoring head
        # Takes concatenated graph embedding and order embedding
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim, hidden_dim),  # 3 for pooling, 1 for order
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters for transfer learning"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
    
    def forward(self, x, edge_index, edge_attr, orders, batch):
        """
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features] (optional)
            orders: Join order encodings [batch_size, num_orders, order_encoding_dim]
            batch: Batch assignment for nodes [num_nodes]
        
        Returns:
            scores: Scores for each join order [batch_size, num_orders]
        """
        # Encode graph structure
        node_embeddings = self.encoder(x, edge_index, edge_attr)
        
        # Pool to graph-level representation
        graph_embedding = self.pool(node_embeddings, batch)  # [batch_size, hidden_dim * 3]
        
        # Score each join order
        batch_size, num_orders, order_dim = orders.shape
        scores = []
        
        for i in range(num_orders):
            # Get order encoding for this join order
            order_enc = orders[:, i, :]  # [batch_size, order_encoding_dim]
            order_emb = self.order_encoder(order_enc)  # [batch_size, hidden_dim]
            
            # Concatenate graph and order embeddings
            combined = torch.cat([graph_embedding, order_emb], dim=-1)
            
            # Compute score
            score = self.score_head(combined)  # [batch_size, 1]
            scores.append(score)
        
        # Stack scores
        scores = torch.cat(scores, dim=-1)  # [batch_size, num_orders]
        
        return scores
    
    def get_graph_embedding(self, x, edge_index, edge_attr, batch):
        """Get graph embedding without scoring (useful for analysis)"""
        node_embeddings = self.encoder(x, edge_index, edge_attr)
        graph_embedding = self.pool(node_embeddings, batch)
        return graph_embedding

class GlobalPooling(nn.Module):
    """Global pooling layer combining multiple aggregations"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, x, batch):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Pooled features [batch_size, hidden_dim * 3]
        """
        # Multiple pooling strategies
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        add_pool = global_add_pool(x, batch)
        
        # Concatenate
        pooled = torch.cat([mean_pool, max_pool, add_pool], dim=-1)
        
        return pooled

class TransferAdapter(nn.Module):
    """
    Adapter module for few-shot transfer learning.
    Adds a small trainable module while keeping the main encoder frozen.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Initialize near identity
        with torch.no_grad():
            self.adapter[-1].weight.data = torch.eye(hidden_dim)[:hidden_dim // 4]
            self.adapter[-1].bias.data.zero_()
    
    def forward(self, x):
        return x + self.adapter(x)

class JoinOrderRankerWithAdapter(nn.Module):
    """
    Join order ranker with adapter modules for transfer learning
    """
    
    def __init__(self, base_model: JoinOrderRanker, num_adapters: int = 3):
        super().__init__()
        
        self.base_model = base_model
        self.base_model._freeze_encoder()
        
        # Add adapters after each GNN layer
        self.adapters = nn.ModuleList([
            TransferAdapter(base_model.hidden_dim)
            for _ in range(num_adapters)
        ])
    
    def forward(self, x, edge_index, edge_attr, orders, batch):
        """Forward with adapters inserted"""
        # This is simplified - in practice you'd insert adapters between layers
        # For now, just apply adapter to final graph embedding
        
        # Get base graph embedding
        graph_emb = self.base_model.get_graph_embedding(x, edge_index, edge_attr, batch)
        
        # Apply adapter
        adapted_emb = self.adapters[0](graph_emb[:, :self.base_model.hidden_dim])
        
        # Continue with scoring using adapted embedding
        # (This is a simplified implementation)
        batch_size, num_orders, order_dim = orders.shape
        scores = []
        
        for i in range(num_orders):
            order_enc = orders[:, i, :]
            order_emb = self.base_model.order_encoder(order_enc)
            
            # Use adapted embedding
            combined = torch.cat([
                adapted_emb,
                graph_emb[:, self.base_model.hidden_dim:],  # Keep other pooling
                order_emb
            ], dim=-1)
            
            score = self.base_model.score_head(combined)
            scores.append(score)
        
        scores = torch.cat(scores, dim=-1)
        return scores

if __name__ == "__main__":
    print("Testing JoinOrderRanker model...")
    
    # Model parameters
    encoder_type = "gin"  # Use 'gin' or 'sage'
    node_features = 64
    hidden_dim = 128
    num_layers = 4
    order_encoding_dim = 32
    dropout = 0.1
    
    model = JoinOrderRanker(
        encoder_type=encoder_type,
        node_features=node_features,
        edge_features=None,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        order_encoding_dim=order_encoding_dim,
        dropout=dropout
    )
    
    # Create sample data for 2 graphs in a batch
    num_nodes_graph1 = 25
    num_nodes_graph2 = 25
    total_nodes = num_nodes_graph1 + num_nodes_graph2
    
    x = torch.randn(total_nodes, node_features)
    
    # Create edges (simple chain for each graph)
    edge_index_1 = torch.tensor([[i, i+1] for i in range(num_nodes_graph1-1)]).t()
    edge_index_2 = torch.tensor([[i+num_nodes_graph1, i+num_nodes_graph1+1] 
                                  for i in range(num_nodes_graph2-1)]).t()
    edge_index = torch.cat([edge_index_1, edge_index_2], dim=1)
    
    edge_attr = None
    
    # Batch tensor: indicates which graph each node belongs to
    batch = torch.cat([
        torch.zeros(num_nodes_graph1, dtype=torch.long),
        torch.ones(num_nodes_graph2, dtype=torch.long)
    ])
    
    # Join orders for each graph (10 joins each)
    orders = torch.randn(2, 10, order_encoding_dim)
    
    print(f"Node features shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Batch tensor shape: {batch.shape}")
    print(f"Orders shape: {orders.shape}")
    
    # Forward pass
    scores = model(x, edge_index, edge_attr, orders, batch)
    print(f"Output scores shape: {scores.shape}")
    print(f"Sample scores: {scores}")
    print("\nTest completed successfully!")