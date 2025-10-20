#!/usr/bin/env python3
"""
Simplified Advanced Transfer Learning Experiments
Runs comprehensive evaluation without external visualization dependencies
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
from transfer_learning import create_transfer_model
from losses import PairwiseHingeLoss

def compute_enhanced_metrics(predictions, targets):
    """Compute enhanced evaluation metrics"""
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
            dcg += y_true_sorted[i] / np.log2(i + 1)

        # IDCG
        y_true_ideal = np.sort(y_true)[::-1][:k]
        idcg = y_true_ideal[0]
        for i in range(1, len(y_true_ideal)):
            idcg += y_true_ideal[i] / np.log2(i + 1)

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
        metrics['spearman'] = spearman_correlation(predictions, targets)
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

        # Add standard deviations
        metrics['ndcg@5_std'] = np.std(ndcg5_scores) if ndcg5_scores else 0.0
        metrics['spearman_std'] = np.std(spearman_scores) if spearman_scores else 0.0

    # Add confidence intervals (simple approximation)
    metrics['ndcg@5_ci'] = (
        max(0, metrics['ndcg@5'] - 1.96 * metrics.get('ndcg@5_std', 0.01)),
        min(1, metrics['ndcg@5'] + 1.96 * metrics.get('ndcg@5_std', 0.01))
    )

    return metrics

def run_enhanced_experiment(strategy: str, sample_sizes: list, config: dict) -> dict:
    """Run enhanced experiment for a single strategy"""
    print(f"\n=== ENHANCED {strategy.upper()} EXPERIMENT ===")

    results = {}

    for sample_size in sample_sizes:
        print(f"\n--- {strategy} with {sample_size} samples ---")

        try:
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

            # Create model
            model = create_transfer_model(
                'models/tpch_pretrain/best_model.pt',
                strategy=strategy,
                config={'device': config['device']}
            )

            # Setup training
            criterion = PairwiseHingeLoss()

            # Use discriminative learning rates
            if strategy == 'head_only':
                # Only train the head
                trainable_params = model.get_trainable_parameters()
                optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'], weight_decay=0.01)
            else:
                # Different learning rates for different parts
                # Collect all trainable parameters with IDs to avoid duplicates
                param_dict = {id(p): p for p in model.parameters() if p.requires_grad}

                # Separate parameters into groups
                gnn_param_ids = set()
                proj_param_ids = set()
                head_adapter_param_ids = set()

                # 1. GNN encoder parameters (adapters/LoRA modules within encoder)
                for p in model.base_model.encoder.parameters():
                    if p.requires_grad:
                        gnn_param_ids.add(id(p))

                # 2. Projection layer parameters
                if hasattr(model, 'feature_projection'):
                    for p in model.feature_projection.parameters():
                        if p.requires_grad:
                            proj_param_ids.add(id(p))
                if hasattr(model, 'order_projection'):
                    for p in model.order_projection.parameters():
                        if p.requires_grad:
                            proj_param_ids.add(id(p))

                # 3. Score head parameters
                for p in model.base_model.score_head.parameters():
                    if p.requires_grad:
                        head_adapter_param_ids.add(id(p))

                # Build param groups
                param_groups = []
                if gnn_param_ids:
                    param_groups.append({
                        'params': [param_dict[pid] for pid in gnn_param_ids],
                        'lr': config['learning_rate'] * 0.1
                    })
                if proj_param_ids:
                    param_groups.append({
                        'params': [param_dict[pid] for pid in proj_param_ids],
                        'lr': config['learning_rate'] * 0.5
                    })
                if head_adapter_param_ids:
                    param_groups.append({
                        'params': [param_dict[pid] for pid in head_adapter_param_ids],
                        'lr': config['learning_rate']
                    })

                # Add any remaining trainable parameters not yet assigned
                assigned_ids = gnn_param_ids | proj_param_ids | head_adapter_param_ids
                remaining_ids = set(param_dict.keys()) - assigned_ids
                if remaining_ids:
                    param_groups.append({
                        'params': [param_dict[pid] for pid in remaining_ids],
                        'lr': config['learning_rate']
                    })

                optimizer = optim.AdamW(param_groups, weight_decay=0.01)

            # Learning rate scheduler with warmup
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

            print(f"Model parameters: {model.get_parameter_count()}")

            # Training loop
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0

            for epoch in range(config['epochs']):
                # Training phase
                model.train()
                train_loss = 0.0
                train_batches = 0

                for batch in train_loader:
                    if batch is None:
                        continue

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

                # Compute validation metrics
                val_metrics = compute_enhanced_metrics(all_predictions, all_targets)

                # Learning rate scheduling
                scheduler.step()

                print(f"Epoch {epoch + 1:2d}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, NDCG@5: {val_metrics['ndcg@5']:.4f}, "
                      f"Spearman: {val_metrics['spearman']:.4f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f'models/enhanced_{strategy}_{sample_size}_best.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Load best model and final evaluation
            model.load_state_dict(torch.load(f'models/enhanced_{strategy}_{sample_size}_best.pt'))
            model.eval()

            # Final comprehensive evaluation
            final_predictions = []
            final_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue

                    scores = model(
                        batch.x, batch.edge_index,
                        batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                        batch.orders, batch.batch
                    )

                    final_predictions.extend(scores.cpu().numpy())
                    final_targets.extend(batch.targets.cpu().numpy())

            # Compute final metrics
            final_metrics = compute_enhanced_metrics(final_predictions, final_targets)
            final_metrics['best_val_loss'] = best_val_loss
            final_metrics['total_epochs'] = epoch + 1

            results[str(sample_size)] = final_metrics

            print(f"[OK] {strategy} with {sample_size} samples - Final Results:")
            print(f"  NDCG@5: {final_metrics['ndcg@5']:.4f} +/- {final_metrics.get('ndcg@5_std', 0):.4f}")
            print(f"  NDCG@3: {final_metrics['ndcg@3']:.4f}")
            print(f"  NDCG@1: {final_metrics['ndcg@1']:.4f}")
            print(f"  Spearman: {final_metrics['spearman']:.4f} +/- {final_metrics.get('spearman_std', 0):.4f}")
            print(f"  Best Val Loss: {best_val_loss:.4f}")

        except Exception as e:
            print(f"[FAIL] {strategy} with {sample_size} samples failed: {e}")
            import traceback
            traceback.print_exc()
            results[str(sample_size)] = {'error': str(e)}

    return results

def main():
    """Run all enhanced experiments"""
    print("=" * 70)
    print("ENHANCED FEW-SHOT TRANSFER LEARNING EXPERIMENTS")
    print("=" * 70)

    # Configuration
    config = {
        'device': 'cpu',
        'learning_rate': 1e-3,
        'batch_size': 4,
        'epochs': 15,
        'strategies': ['head_only', 'adapters', 'lora'],
        'sample_sizes': [10, 25, 50]
    }

    # Create output directory
    output_dir = Path('results/enhanced_experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = {}
    start_time = time.time()

    for strategy in config['strategies']:
        strategy_results = run_enhanced_experiment(strategy, config['sample_sizes'], config)
        all_results[strategy] = strategy_results

        # Save intermediate results (convert numpy types to Python types)
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        serializable_results = convert_numpy(strategy_results)
        with open(output_dir / f'{strategy}_enhanced_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)

    total_time = time.time() - start_time

    # Save all results
    all_serializable_results = convert_numpy(all_results)
    with open(output_dir / 'all_enhanced_results.json', 'w') as f:
        json.dump(all_serializable_results, f, indent=2)

    print(f"\n" + "=" * 70)
    print("ENHANCED EXPERIMENT RESULTS")
    print("=" * 70)

    # Performance summary
    print("\n[RESULTS] PERFORMANCE SUMMARY:")
    print("=" * 50)

    best_results = []
    for strategy, results in all_results.items():
        print(f"\n>> {strategy.upper()}:")
        for sample_size, metrics in results.items():
            if isinstance(metrics, dict) and 'ndcg@5' in metrics:
                ndcg5 = metrics['ndcg@5']
                ndcg3 = metrics['ndcg@3']
                ndcg1 = metrics['ndcg@1']
                spearman = metrics['spearman']

                print(f"  * {sample_size:2s} samples: NDCG@5={ndcg5:.4f} | NDCG@3={ndcg3:.4f} | NDCG@1={ndcg1:.4f} | Spearman={spearman:.4f}")

                best_results.append({
                    'strategy': strategy,
                    'sample_size': sample_size,
                    'ndcg@5': ndcg5,
                    'ndcg@3': ndcg3,
                    'ndcg@1': ndcg1,
                    'spearman': spearman
                })

    # Find best configurations
    best_results.sort(key=lambda x: x['ndcg@5'], reverse=True)

    print(f"\n[RANKING] TOP 5 CONFIGURATIONS:")
    print("=" * 50)
    for i, result in enumerate(best_results[:5], 1):
        print(f"{i}. {result['strategy'].upper()} ({result['sample_size']} samples): "
              f"NDCG@5={result['ndcg@5']:.4f}, Spearman={result['spearman']:.4f}")

    # Improvement analysis
    print(f"\n[ANALYSIS] IMPROVEMENT ANALYSIS:")
    print("=" * 50)

    for strategy in config['strategies']:
        if strategy in all_results:
            samples_10 = all_results[strategy].get('10', {})
            samples_50 = all_results[strategy].get('50', {})

            if 'ndcg@5' in samples_10 and 'ndcg@5' in samples_50:
                improvement = ((samples_50['ndcg@5'] - samples_10['ndcg@5']) / samples_10['ndcg@5']) * 100
                print(f">> {strategy.upper()}: {improvement:+.1f}% improvement (10->50 samples)")

    # Statistical summary
    print(f"\n[STATS] STATISTICAL SUMMARY:")
    print("=" * 50)

    all_ndcg5_scores = []
    all_spearman_scores = []

    for strategy, results in all_results.items():
        for sample_size, metrics in results.items():
            if isinstance(metrics, dict) and 'ndcg@5' in metrics:
                all_ndcg5_scores.append(metrics['ndcg@5'])
                all_spearman_scores.append(metrics['spearman'])

    if all_ndcg5_scores:
        print(f">> NDCG@5 - Mean: {np.mean(all_ndcg5_scores):.4f}, Std: {np.std(all_ndcg5_scores):.4f}")
        print(f">> NDCG@5 - Range: [{np.min(all_ndcg5_scores):.4f}, {np.max(all_ndcg5_scores):.4f}]")

    if all_spearman_scores:
        print(f">> Spearman - Mean: {np.mean(all_spearman_scores):.4f}, Std: {np.std(all_spearman_scores):.4f}")
        print(f">> Spearman - Range: [{np.min(all_spearman_scores):.4f}, {np.max(all_spearman_scores):.4f}]")

    print(f"\n[COMPLETE] EXPERIMENT COMPLETED!")
    print(f"[SAVE] Results saved to: {output_dir}")
    print(f"[TIME] Total time: {total_time/60:.1f} minutes")

    if best_results:
        best = best_results[0]
        print(f"\n[WINNER] BEST OVERALL RESULT:")
        print(f"   Strategy: {best['strategy'].upper()}")
        print(f"   Sample Size: {best['sample_size']}")
        print(f"   NDCG@5: {best['ndcg@5']:.4f}")
        print(f"   NDCG@3: {best['ndcg@3']:.4f}")
        print(f"   NDCG@1: {best['ndcg@1']:.4f}")
        print(f"   Spearman: {best['spearman']:.4f}")

    return all_results

if __name__ == "__main__":
    results = main()