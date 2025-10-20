#!/usr/bin/env python3
"""
Pretraining loop for GNN encoder on TPC-H
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from encoders import JoinOrderRanker
from losses import PairwiseHingeLoss, ListwiseLoss

class JoinOrderDataset(Dataset):
    """Dataset for join order ranking"""
    
    def __init__(self, data_path: str, split: str = 'train'):
        """
        Args:
            data_path: Path to processed data directory
            split: 'train', 'val', or 'test'
        """
        self.data_path = Path(data_path)
        self.split = split
        
        # Load data
        with open(self.data_path / f"{split}_data.json") as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} examples for {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a query with multiple join orders and their runtimes
        """
        item = self.data[idx]
        
        # Load graph data (PyG Data object saved as .pt)
        graph_file = self.data_path / item['graph_file']
        graph_data = torch.load(graph_file, weights_only=False)
        
        # Load join orders and runtimes
        orders = item['orders']  # List of order encodings
        runtimes = torch.tensor(item['runtimes'], dtype=torch.float32)
        
        # Convert runtimes to scores (negative log runtime)
        # Lower runtime = higher score
        scores = -torch.log(runtimes + 1.0)
        
        return {
            'graph': graph_data,
            'orders': torch.tensor(orders, dtype=torch.float32),
            'scores': scores,
            'query_id': item['query_id']
        }

def collate_fn(batch):
    """Custom collate function for batching graphs"""
    graphs = [item['graph'] for item in batch]
    batch_graphs = Batch.from_data_list(graphs)
    
    # Stack orders and scores
    orders = torch.stack([item['orders'] for item in batch])
    scores = torch.stack([item['scores'] for item in batch])
    
    return {
        'graph': batch_graphs,
        'orders': orders,
        'scores': scores
    }

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        graph = batch['graph'].to(device)
        orders = batch['orders'].to(device)
        scores = batch['scores'].to(device)
        
        # Forward pass
        pred_scores = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr if hasattr(graph, 'edge_attr') else None,
            orders,
            graph.batch
        )
        
        # Compute loss
        loss = criterion(pred_scores, scores)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_ndcg = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            graph = batch['graph'].to(device)
            orders = batch['orders'].to(device)
            scores = batch['scores'].to(device)
            
            # Forward pass
            pred_scores = model(
                graph.x,
                graph.edge_index,
                graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                orders,
                graph.batch
            )
            
            # Compute loss
            loss = criterion(pred_scores, scores)
            total_loss += loss.item()
            
            # Compute NDCG@k
            for i in range(pred_scores.size(0)):
                ndcg = compute_ndcg(pred_scores[i], scores[i], k=10)
                all_ndcg.append(ndcg)
    
    avg_loss = total_loss / len(dataloader)
    avg_ndcg = np.mean(all_ndcg)
    
    return avg_loss, avg_ndcg

def compute_ndcg(pred_scores, true_scores, k=10):
    """Compute NDCG@k"""
    # Get top-k indices by predicted scores
    _, pred_indices = torch.topk(pred_scores, min(k, len(pred_scores)))
    
    # Get ideal ranking by true scores
    _, ideal_indices = torch.topk(true_scores, min(k, len(true_scores)))
    
    # Compute DCG
    dcg = 0.0
    for i, idx in enumerate(pred_indices):
        relevance = true_scores[idx].item()
        dcg += relevance / np.log2(i + 2)
    
    # Compute IDCG
    idcg = 0.0
    for i, idx in enumerate(ideal_indices):
        relevance = true_scores[idx].item()
        idcg += relevance / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to processed data')
    parser.add_argument('--encoder', default='gin', choices=['gin', 'sage'])
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='results/tpch_pretrain')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = JoinOrderDataset(args.data_dir, split='train')
    val_dataset = JoinOrderDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    model = JoinOrderRanker(
        encoder_type=args.encoder,
        node_features=4,
        edge_features=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        order_encoding_dim=32,
        dropout=args.dropout
    ).to(args.device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = PairwiseHingeLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop
    best_ndcg = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_ndcg': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        
        # Evaluate
        val_loss, val_ndcg = evaluate(model, val_loader, criterion, args.device)
        
        # Log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val NDCG@10: {val_ndcg:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ndcg'].append(val_ndcg)
        
        # Save best model
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ndcg': val_ndcg,
                'args': vars(args)
            }, save_dir / 'best_model.pt')
            print(f"Saved new best model (NDCG@10: {val_ndcg:.4f})")
        
        # Learning rate scheduling
        scheduler.step(val_ndcg)
    
    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best Val NDCG@10: {best_ndcg:.4f}")

if __name__ == "__main__":
    main()