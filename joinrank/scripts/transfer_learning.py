#!/usr/bin/env python3
"""
Transfer Learning Strategies for Few-Shot Join Order Selection
Implements Head-Only, Adapters, and LoRA fine-tuning strategies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer
    Decomposes weight updates into low-rank matrices: ΔW = BA
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.1)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # ΔW = B @ A, then x @ ΔW.T = x @ A.T @ B.T
        return x @ self.lora_A.T @ self.lora_B.T * self.scaling

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""

    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_layer = original_layer

        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Add LoRA adaptation
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        return self.original_layer(x) + self.lora(x)

class AdapterLayer(nn.Module):
    """
    Adapter module for transfer learning
    Small bottleneck network inserted between frozen layers
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = None, dropout: float = 0.1):
        super().__init__()

        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim // 4

        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim)
        )

        # Initialize adapter to near-identity transformation
        with torch.no_grad():
            # Initialize the last layer to output small values
            self.adapter[-1].weight.data.normal_(0, 0.01)
            self.adapter[-1].bias.data.zero_()

    def forward(self, x):
        # Residual connection
        return x + self.adapter(x)

class GNNEncoderWithAdapters(nn.Module):
    """
    GNN encoder with adapter modules inserted between layers
    """

    def __init__(self, base_encoder, adapter_positions: list = None, bottleneck_dim: int = None):
        super().__init__()
        self.base_encoder = base_encoder

        # Freeze base encoder
        for param in self.base_encoder.parameters():
            param.requires_grad = False

        # Add adapters at specified positions
        if adapter_positions is None:
            adapter_positions = list(range(len(base_encoder.convs)))

        self.adapters = nn.ModuleDict()
        # Get the actual hidden dimension from the encoder
        actual_hidden_dim = getattr(base_encoder, 'hidden_dim', 256)

        for pos in adapter_positions:
            self.adapters[f'adapter_{pos}'] = AdapterLayer(
                actual_hidden_dim,
                bottleneck_dim
            )

        self.adapter_positions = adapter_positions

    def forward(self, x, edge_index, edge_attr=None):
        # Forward through base encoder with adapters
        for i in range(self.base_encoder.num_layers):
            x = self.base_encoder.convs[i](x, edge_index)
            x = self.base_encoder.batch_norms[i](x)

            # Apply adapter if at this position
            if i in self.adapter_positions:
                x = self.adapters[f'adapter_{i}'](x)

            if i < self.base_encoder.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.base_encoder.dropout, training=self.training)

        return x

class GNNEncoderWithLoRA(nn.Module):
    """
    GNN encoder with LoRA adaptation applied to linear layers
    """

    def __init__(self, base_encoder, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_encoder = base_encoder

        # Freeze base encoder
        for param in self.base_encoder.parameters():
            param.requires_grad = False

        # Replace linear layers in GIN convolutions with LoRA versions
        self._apply_lora_to_convs(rank, alpha)

    def _apply_lora_to_convs(self, rank: int, alpha: float):
        """Apply LoRA to linear layers in GIN convolutions"""
        for i, conv in enumerate(self.base_encoder.convs):
            if hasattr(conv, 'nn'):  # GINConv has a 'nn' sequential module
                new_layers = []
                for layer in conv.nn:
                    if isinstance(layer, nn.Linear):
                        new_layers.append(LoRALinear(layer, rank=rank, alpha=alpha))
                    else:
                        new_layers.append(layer)
                conv.nn = nn.Sequential(*new_layers)

    def forward(self, x, edge_index, edge_attr=None):
        return self.base_encoder(x, edge_index, edge_attr)

class TransferLearningRanker(nn.Module):
    """
    Join order ranker with configurable transfer learning strategies
    """

    def __init__(self,
                 pretrained_model_path: str,
                 strategy: str = 'head_only',
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0,
                 adapter_bottleneck_dim: int = None,
                 device: str = 'cuda'):
        super().__init__()

        self.strategy = strategy
        self.device = device

        # Load pretrained model (weights_only=False for compatibility with older checkpoints)
        checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Get model config from args if available
            if 'args' in checkpoint:
                args = checkpoint['args']
                hidden_dim = getattr(args, 'hidden_dim', 256)
                num_layers = getattr(args, 'num_layers', 3)
                dropout = getattr(args, 'dropout', 0.2)
            else:
                hidden_dim = 256
                num_layers = 3
                dropout = 0.2
        else:
            state_dict = checkpoint
            hidden_dim = 256
            num_layers = 3
            dropout = 0.2

        # Import the base model architecture
        from encoders import JoinOrderRanker

        # Reconstruct model with same architecture as pretraining
        self.base_model = JoinOrderRanker(
            encoder_type='gin',  # GIN encoder as used in pretraining
            node_features=128,   # TPC-H used 128 features
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            order_encoding_dim=32,  # Standard order encoding size
            dropout=dropout
        )

        # Load pretrained weights
        self.base_model.load_state_dict(state_dict, strict=False)

        # Add feature projection layers for domain adaptation
        # IMDB has 64-dim features, TPC-H pretrained model expects 128-dim
        self.feature_projection = nn.Linear(64, 128)

        # IMDB order encoder outputs 128-dim, TPC-H model expects 32-dim
        self.order_projection = nn.Linear(128, 32)

        # Initialize projections to preserve information
        with torch.no_grad():
            # Node feature projection: 64->128 by duplicating and adding small noise
            self.feature_projection.weight[:64, :].copy_(torch.eye(64))
            self.feature_projection.weight[64:, :].copy_(torch.eye(64) * 0.1)
            self.feature_projection.bias.zero_()

            # Order projection: 128->32 by taking first 32 dimensions
            self.order_projection.weight.copy_(torch.eye(32, 128))
            self.order_projection.bias.zero_()

        # Apply transfer learning strategy
        self._setup_transfer_strategy(lora_rank, lora_alpha, adapter_bottleneck_dim)

    def _setup_transfer_strategy(self, lora_rank: int, lora_alpha: float, adapter_bottleneck_dim: int):
        """Setup the specific transfer learning strategy"""

        if self.strategy == 'head_only':
            # Freeze everything except the final scoring head and feature projection
            for name, param in self.base_model.named_parameters():
                if 'score_head' not in name:
                    param.requires_grad = False
            # Projection layers are always trainable
            for param in self.feature_projection.parameters():
                param.requires_grad = True
            for param in self.order_projection.parameters():
                param.requires_grad = True

        elif self.strategy == 'adapters':
            # Replace encoder with adapter version
            self.base_model.encoder = GNNEncoderWithAdapters(
                self.base_model.encoder,
                bottleneck_dim=adapter_bottleneck_dim
            )

            # Freeze everything except adapters and head
            for name, param in self.base_model.named_parameters():
                if 'adapter' not in name and 'score_head' not in name:
                    param.requires_grad = False
            # Projection layers are always trainable
            for param in self.feature_projection.parameters():
                param.requires_grad = True
            for param in self.order_projection.parameters():
                param.requires_grad = True

        elif self.strategy == 'lora':
            # Replace encoder with LoRA version
            self.base_model.encoder = GNNEncoderWithLoRA(
                self.base_model.encoder,
                rank=lora_rank,
                alpha=lora_alpha
            )

            # Only LoRA parameters and head are trainable
            for name, param in self.base_model.named_parameters():
                if 'lora' not in name and 'score_head' not in name:
                    param.requires_grad = False
            # Projection layers are always trainable
            for param in self.feature_projection.parameters():
                param.requires_grad = True
            for param in self.order_projection.parameters():
                param.requires_grad = True

        elif self.strategy == 'full_finetune':
            # All parameters are trainable
            pass

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def forward(self, x, edge_index, edge_attr, orders, batch):
        # Project node features from 64-dim to 128-dim
        x_projected = self.feature_projection(x)

        # Project order encodings from 128-dim to 32-dim
        batch_size, num_orders, order_dim = orders.shape
        orders_reshaped = orders.view(-1, order_dim)  # [batch_size * num_orders, 128]
        orders_projected = self.order_projection(orders_reshaped)  # [batch_size * num_orders, 32]
        orders_projected = orders_projected.view(batch_size, num_orders, 32)  # [batch_size, num_orders, 32]

        return self.base_model(x_projected, edge_index, edge_attr, orders_projected, batch)

    def get_trainable_parameters(self):
        """Get only the trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]

    def get_parameter_count(self):
        """Get count of total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'trainable_ratio': trainable_params / total_params
        }

def create_transfer_model(pretrained_path: str,
                         strategy: str,
                         config: Dict[str, Any] = None) -> TransferLearningRanker:
    """
    Factory function to create transfer learning models

    Args:
        pretrained_path: Path to pretrained model
        strategy: 'head_only', 'adapters', 'lora', or 'full_finetune'
        config: Configuration dictionary for strategy-specific parameters
    """
    if config is None:
        config = {}

    return TransferLearningRanker(
        pretrained_model_path=pretrained_path,
        strategy=strategy,
        lora_rank=config.get('lora_rank', 8),
        lora_alpha=config.get('lora_alpha', 16.0),
        adapter_bottleneck_dim=config.get('adapter_bottleneck_dim', 32),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )

if __name__ == "__main__":
    # Test the transfer learning implementations
    print("Testing Transfer Learning Strategies...")

    # Create a dummy pretrained model for testing
    from encoders import JoinOrderRanker

    dummy_model = JoinOrderRanker(
        encoder_type='gin',
        node_features=64,
        hidden_dim=128,
        num_layers=3,
        order_encoding_dim=32
    )

    # Save dummy model
    torch.save(dummy_model.state_dict(), 'dummy_pretrained.pt')

    # Test each strategy
    strategies = ['head_only', 'adapters', 'lora', 'full_finetune']

    for strategy in strategies:
        print(f"\n--- Testing {strategy} strategy ---")

        model = create_transfer_model(
            'dummy_pretrained.pt',
            strategy,
            config={'device': 'cpu'}
        )

        param_counts = model.get_parameter_count()
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")
        print(f"Trainable ratio: {param_counts['trainable_ratio']:.4f}")

        # Test forward pass
        batch_size = 2
        num_nodes = 10
        num_orders = 5

        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, 15))
        edge_attr = None
        orders = torch.randn(batch_size, num_orders, 32)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        try:
            with torch.no_grad():
                scores = model(x, edge_index, edge_attr, orders, batch)
            print(f"Forward pass successful: {scores.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")

    # Cleanup
    import os
    os.remove('dummy_pretrained.pt')

    print("\nTransfer learning module test completed!")