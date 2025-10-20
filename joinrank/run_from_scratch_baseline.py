#!/usr/bin/env python3
"""
Train baseline model from scratch on IMDB without transfer learning
This provides comparison against transfer learning approaches
"""
import sys
import os
sys.path.append('scripts')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import numpy as np
from pathlib import Path

# Import core modules
from fewshot_dataset import FewShotRankingDataset, collate_fewshot_batch
from encoders import JoinOrderRanker
from losses import PairwiseHingeLoss

def compute_metrics(predictions, targets):
    """Compute evaluation metrics"""
    predictions = np.array(predictions)
    targets = np.array(targets)

    metrics = {}

    # NDCG calculation
    def ndcg_score(y_true, y_score, k=5):
        if len(y_true) < k:
            k = len(y_true)

        # Sort by predicted scores
        order = np.argsort(y_score)[::-1]
        y_true_sorted = np.take(y_true, order[:k])

        # DCG
        dcg = y_true_sorted[0]
        for i in range(1, len(y_true_sorted)):
            dcg += y_true_sorted[i] / np.log2(i + 2)

        # IDCG
        y_true_ideal = np.sort(y_true)[::-1][:k]
        idcg = y_true_ideal[0]
        for i in range(1, len(y_true_ideal)):
            idcg += y_true_ideal[i] / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    # Spearman correlation
    def spearman_correlation(x, y):
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))
        return np.corrcoef(rank_x, rank_y)[0, 1]

    # Compute metrics for each query
    if len(predictions.shape) == 1:
        # Single query
        metrics['ndcg@5'] = ndcg_score(targets, predictions, k=5)
        metrics['ndcg@3'] = ndcg_score(targets, predictions, k=3)
        metrics['ndcg@1'] = ndcg_score(targets, predictions, k=1)
        try:
            metrics['spearman'] = spearman_correlation(predictions, targets)
        except:
            metrics['spearman'] = 0.0
    else:
        # Multiple queries
        ndcg5_scores = []
        ndcg3_scores = []
        ndcg1_scores = []
        spearman_scores = []

        for i in range(predictions.shape[0]):
            ndcg5_scores.append(ndcg_score(targets[i], predictions[i], k=5))
            ndcg3_scores.append(ndcg_score(targets[i], predictions[i], k=3))
            ndcg1_scores.append(ndcg_score(targets[i], predictions[i], k=1))
            try:
                spear = spearman_correlation(predictions[i], targets[i])
                if not np.isnan(spear):
                    spearman_scores.append(spear)
            except:
                pass

        metrics['ndcg@5'] = np.mean(ndcg5_scores) if ndcg5_scores else 0.0
        metrics['ndcg@3'] = np.mean(ndcg3_scores) if ndcg3_scores else 0.0
        metrics['ndcg@1'] = np.mean(ndcg1_scores) if ndcg1_scores else 0.0
        metrics['spearman'] = np.mean(spearman_scores) if spearman_scores else 0.0

        metrics['ndcg@5_std'] = np.std(ndcg5_scores) if ndcg5_scores else 0.0
        metrics['spearman_std'] = np.std(spearman_scores) if spearman_scores else 0.0

    return metrics

def train_from_scratch(sample_size: int, config: dict) -> dict:
    """Train model from scratch on IMDB"""
    print(f"\n=== TRAINING FROM SCRATCH ({sample_size} SAMPLES) ===")

    # Create datasets
    train_dataset = FewShotRankingDataset(
        'data/imdb/processed',
        split=f'custom_{sample_size}',
        max_orders_per_query=5
    )
    val_dataset = FewShotRankingDataset(
        'data/imdb/processed',
        split='val',
        max_orders_per_query=5
    )

    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Val dataset: {len(val_dataset)} examples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fewshot_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fewshot_batch
    )

    # Create model from scratch
    model = JoinOrderRanker(
        encoder_type='gin',
        node_features=64,  # IMDB has 64-dim features
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 3),
        order_encoding_dim=128,  # IMDB order encoder outputs 128-dim
        dropout=config.get('dropout', 0.2),
        freeze_encoder=False  # Train everything
    ).to(config['device'])

    # Setup training
    criterion = PairwiseHingeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    start_time = time.time()

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            if batch is None:
                continue

            # Move batch to device
            batch = batch.to(config['device'])

            optimizer.zero_grad()

            scores = model(
                batch.x, batch.edge_index,
                batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                batch.orders, batch.batch
            )

            loss = criterion(scores, batch.targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                # Move batch to device
                batch = batch.to(config['device'])

                scores = model(
                    batch.x, batch.edge_index,
                    batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                    batch.orders, batch.batch
                )

                loss = criterion(scores, batch.targets)
                val_loss += loss.item()
                val_batches += 1

                all_predictions.extend(scores.cpu().numpy())
                all_targets.extend(batch.targets.cpu().numpy())

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)

        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)

        print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"NDCG@5: {metrics['ndcg@5']:.4f}, Spearman: {metrics['spearman']:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/from_scratch_{sample_size}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    training_time = time.time() - start_time

    # Load best model for final evaluation
    model.load_state_dict(torch.load(f'models/from_scratch_{sample_size}_best.pt'))
    model.eval()

    # Final evaluation
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            # Move batch to device
            batch = batch.to(config['device'])

            scores = model(
                batch.x, batch.edge_index,
                batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                batch.orders, batch.batch
            )

            all_predictions.extend(scores.cpu().numpy())
            all_targets.extend(batch.targets.cpu().numpy())

    final_metrics = compute_metrics(all_predictions, all_targets)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    results = {
        'sample_size': int(sample_size),
        'training_time': float(training_time),
        'parameter_counts': {
            'total': int(total_params),
            'trainable': int(trainable_params),
            'trainable_ratio': 1.0  # All parameters are trainable
        },
        'best_val_loss': float(best_val_loss),
        'final_metrics': convert_to_python_types(final_metrics)
    }

    print(f"\n[FINAL RESULTS] From Scratch with {sample_size} samples:")
    print(f"  NDCG@5: {final_metrics['ndcg@5']:.4f} +/- {final_metrics.get('ndcg@5_std', 0):.4f}")
    print(f"  Spearman: {final_metrics['spearman']:.4f} +/- {final_metrics.get('spearman_std', 0):.4f}")
    print(f"  Training time: {training_time:.2f}s")

    return results

def main():
    """Run baseline experiments"""
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 8,
        'learning_rate': 0.001,
        'epochs': 30,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.2
    }

    print("="*70)
    print("FROM-SCRATCH BASELINE EXPERIMENTS")
    print("="*70)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")

    # Run experiments for different sample sizes
    sample_sizes = [10, 25, 50]
    all_results = {}

    for sample_size in sample_sizes:
        results = train_from_scratch(sample_size, config)
        all_results[sample_size] = results

    # Save results
    os.makedirs('results/from_scratch_baseline', exist_ok=True)
    with open('results/from_scratch_baseline/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("BASELINE RESULTS SUMMARY")
    print("="*70)
    for sample_size, results in all_results.items():
        metrics = results['final_metrics']
        print(f"\n{sample_size} samples:")
        print(f"  NDCG@5: {metrics['ndcg@5']:.4f}")
        print(f"  Spearman: {metrics['spearman']:.4f}")
        print(f"  Training time: {results['training_time']:.2f}s")

    print(f"\nResults saved to: results/from_scratch_baseline/all_results.json")

if __name__ == '__main__':
    main()
