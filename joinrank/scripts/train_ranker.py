#!/usr/bin/env python3
"""
Complete training script for join order ranking model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

# Add scripts directory to path
sys.path.append('scripts')

from encoders import JoinOrderRanker
from ranking_dataset import JoinOrderRankingDataset, collate_ranking_batch

class RankingLoss(nn.Module):
    """
    Pairwise ranking loss
    Ensures better order gets lower predicted cost than worse order
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, better_scores, worse_scores, speedups):
        """
        Args:
            better_scores: Predicted costs for better orders [batch_size]
            worse_scores: Predicted costs for worse orders [batch_size]
            speedups: Actual speedup ratios [batch_size]
        
        Returns:
            loss: Ranking loss
        """
        # Hinge loss: max(0, margin - (worse_score - better_score))
        # We want worse_score > better_score by at least margin
        diff = worse_scores - better_scores
        loss = F.relu(self.margin - diff)
        
        # Weight by speedup - larger speedups should have stronger signal
        weights = torch.log(1 + speedups)
        weighted_loss = loss * weights
        
        return weighted_loss.mean()

class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.correct = 0
        self.total = 0
    
    def update(self, loss, better_scores, worse_scores):
        self.losses.append(loss.item())
        
        # Count how many times we correctly rank (better < worse)
        correct = (better_scores < worse_scores).sum().item()
        self.correct += correct
        self.total += len(better_scores)
    
    def get_metrics(self):
        return {
            'loss': np.mean(self.losses),
            'accuracy': self.correct / max(self.total, 1)
        }

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Skip invalid batches
        if batch is None:
            continue
            
        # Move to device
        better_batch = batch['better_batch'].to(device)
        worse_batch = batch['worse_batch'].to(device)
        better_order_encs = batch['better_order_encs'].to(device)
        worse_order_encs = batch['worse_order_encs'].to(device)
        speedups = batch['speedups'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass for both orders
        better_scores = model(
            better_batch.x,
            better_batch.edge_index,
            better_batch.edge_attr,
            better_order_encs.unsqueeze(1),  # [batch_size, 1, order_dim]
            better_batch.batch
        ).squeeze(-1)  # [batch_size]
        
        worse_scores = model(
            worse_batch.x,
            worse_batch.edge_index,
            worse_batch.edge_attr,
            worse_order_encs.unsqueeze(1),
            worse_batch.batch
        ).squeeze(-1)
        
        # Compute loss
        loss = criterion(better_scores, worse_scores, speedups)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        metrics.update(loss, better_scores.detach(), worse_scores.detach())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': metrics.correct / max(metrics.total, 1)})
    
    return metrics.get_metrics()

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    metrics = MetricsTracker()
    
    for batch in tqdm(dataloader, desc="Validating"):
        # Skip invalid batches
        if batch is None:
            continue
            
        better_batch = batch['better_batch'].to(device)
        worse_batch = batch['worse_batch'].to(device)
        better_order_encs = batch['better_order_encs'].to(device)
        worse_order_encs = batch['worse_order_encs'].to(device)
        speedups = batch['speedups'].to(device)
        
        # Forward pass
        better_scores = model(
            better_batch.x,
            better_batch.edge_index,
            better_batch.edge_attr,
            better_order_encs.unsqueeze(1),
            better_batch.batch
        ).squeeze(-1)
        
        worse_scores = model(
            worse_batch.x,
            worse_batch.edge_index,
            worse_batch.edge_attr,
            worse_order_encs.unsqueeze(1),
            worse_batch.batch
        ).squeeze(-1)
        
        loss = criterion(better_scores, worse_scores, speedups)
        metrics.update(loss, better_scores, worse_scores)
    
    return metrics.get_metrics()

def main():
    parser = argparse.ArgumentParser(description='Train join order ranking model')
    parser.add_argument('--data-dir', required=True, help='Directory with ranking data')
    parser.add_argument('--output-dir', default='models/ranker', help='Where to save model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Ranking loss margin')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache directory for graphs
    cache_dir = data_dir / 'graph_cache'
    cache_dir.mkdir(exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = JoinOrderRankingDataset(
        data_dir / 'train_data.json',
        data_dir / 'table_stats.json',
        cache_dir=cache_dir
    )
    
    val_dataset = JoinOrderRankingDataset(
        data_dir / 'val_data.json',
        data_dir / 'table_stats.json',
        cache_dir=cache_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ranking_batch,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = JoinOrderRanker(
        encoder_type='gin',
        node_features=64,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        order_encoding_dim=128,  # Updated to match new join tree encoding
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")
    
    # Loss and optimizer
    criterion = RankingLoss(margin=args.margin)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            print(f"✓ New best model! Saving to {output_dir}/best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping after {epoch + 1} epochs (no improvement for {max_patience} epochs)")
            break
    
    # Save final model and history
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved to: {output_dir}")

if __name__ == "__main__":
    main()