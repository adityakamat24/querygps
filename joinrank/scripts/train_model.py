#!/usr/bin/env python3
"""
Train GNN model for join order selection
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import torch_geometric.nn as pyg_nn

class QueryDataset(Dataset):
    """Dataset for query graphs"""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load metadata (list format)
        with open(self.data_dir / f'{split}_data.json', 'r') as f:
            self.metadata_list = json.load(f)
        
        # Convert list to dictionary if needed
        if isinstance(self.metadata_list, list):
            self.metadata = {item['query_id']: item for item in self.metadata_list}
        else:
            self.metadata = self.metadata_list
        
        # Load table statistics
        with open(self.data_dir / 'table_stats.json', 'r') as f:
            self.table_stats = json.load(f)
        
        self.graphs_dir = self.data_dir / 'graphs'
        self.query_ids = list(self.metadata.keys())
        
        # Normalize runtimes (log scale)
        self.runtimes = []
        for qid in self.query_ids:
            query_data = self.metadata[qid]
            # Handle different data formats
            if 'runtimes' in query_data:
                runtimes = query_data['runtimes']
                if isinstance(runtimes, dict):
                    best_runtime = min(runtimes.values())
                elif isinstance(runtimes, list):
                    best_runtime = min(runtimes)
                else:
                    best_runtime = runtimes
            elif 'runtime' in query_data:
                best_runtime = query_data['runtime']
            elif 'best_runtime' in query_data:
                best_runtime = query_data['best_runtime']
            else:
                # Default fallback
                best_runtime = 1.0
            
            self.runtimes.append(np.log(best_runtime + 1))
    
    def __len__(self):
        return len(self.query_ids)
    
    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        
        # Graph files have _graph.pt suffix
        graph_path = self.graphs_dir / f'{query_id}_graph.pt'
        
        if not graph_path.exists():
            # Try without _graph suffix as fallback
            graph_path = self.graphs_dir / f'{query_id}.pt'
            if not graph_path.exists():
                available_files = list(self.graphs_dir.glob("*.pt"))
                print(f"ERROR: Could not find graph for query_id: {query_id}")
                print(f"Tried: {query_id}_graph.pt and {query_id}.pt")
                print(f"Available files: {[f.name for f in available_files[:3]]}...")
                raise FileNotFoundError(f"Graph file not found for query {query_id}")
        
        # Load graph with weights_only=False for torch_geometric objects
        graph = torch.load(graph_path, weights_only=False)
        
        # Add target runtime
        graph.y = torch.tensor([self.runtimes[idx]], dtype=torch.float32)
        
        return graph

class GNNModel(nn.Module):
    """Graph Neural Network for query cost estimation"""
    
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 1,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Readout layers
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for mean+max pool
        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        # GNN layers with residual connections
        for i in range(self.num_layers):
            x_prev = x if i == 0 else x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            
            # Residual connection (except first layer)
            if i > 0 and x_prev.shape == x.shape:
                x = x + x_prev
        
        # Global pooling (both mean and max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Final layers
        x = F.relu(self.linear1(x))
        x = self.dropout_layer(x)
        x = F.relu(self.linear2(x))
        x = self.dropout_layer(x)
        x = self.linear3(x)
        
        return x

def collate_batch(batch):
    """Custom collate function for batching graphs"""
    return Batch.from_data_list(batch)

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(pred.squeeze(), batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(model, loader, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(pred.squeeze(), batch.y)
            
            total_loss += loss.item()
            num_batches += 1
            
            predictions.extend(pred.squeeze().cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    # Correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    return total_loss / num_batches, mape, correlation

def main():
    parser = argparse.ArgumentParser(description='Train GNN model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory with processed dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    print(f"Loading dataset from {args.data_dir}...")
    
    # Create datasets
    train_dataset = QueryDataset(args.data_dir, 'train')
    val_dataset = QueryDataset(args.data_dir, 'val')
    test_dataset = QueryDataset(args.data_dir, 'test')
    
    print(f"Train: {len(train_dataset)} queries")
    print(f"Val: {len(val_dataset)} queries")
    print(f"Test: {len(test_dataset)} queries")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_batch)
    
    # Get input dimension from first sample
    sample_graph = train_dataset[0]
    input_dim = sample_graph.x.shape[1]
    
    # Initialize model
    print(f"\nInitializing GNN model...")
    model = GNNModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print(f"\nStarting training...")
    print("-" * 80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        val_loss, val_mape, val_corr = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAPE: {val_mape:.2f}% | "
              f"Val Corr: {val_corr:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mape': val_mape,
                'val_corr': val_corr,
                'args': args
            }, os.path.join(args.output_dir, 'best_model.pt'))
            
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model and test
    print("\n" + "="*80)
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    test_loss, test_mape, test_corr = evaluate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
    print(f"  Correlation: {test_corr:.3f}")
    
    # Save final results
    results = {
        'best_epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'val_mape': checkpoint['val_mape'],
        'val_corr': checkpoint['val_corr'],
        'test_loss': test_loss,
        'test_mape': test_mape,
        'test_corr': test_corr,
        'num_parameters': num_params
    }
    # Convert numpy / torch scalars to native Python floats
    results = {k: float(v) if isinstance(v, (np.generic, torch.Tensor)) else v for k, v in results.items()}

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {args.output_dir}")
    print(f"Results saved to {args.output_dir}/results.json")

if __name__ == '__main__':
    main()