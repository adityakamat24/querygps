#!/usr/bin/env python3
"""
Comprehensive Few-Shot Transfer Learning Experiment Runner
Runs systematic experiments comparing Head-Only, Adapters, and LoRA strategies
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
import logging
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from transfer_learning import create_transfer_model
from evaluation_metrics import FewShotEvaluator, evaluate_few_shot_sample_efficiency
from fewshot_dataset import FewShotRankingDataset, collate_fewshot_batch
from losses import ListwiseLoss, PairwiseHingeLoss

class FewShotExperimentRunner:
    """
    Manages and runs few-shot transfer learning experiments
    """

    def __init__(self,
                 pretrained_model_path: str,
                 data_dir: str,
                 output_dir: str,
                 device: str = None):
        self.pretrained_model_path = pretrained_model_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize evaluator
        runtime_data_path = self.data_dir.parent / 'results' / 'imdb_runtimes.json'
        self.evaluator = FewShotEvaluator(
            runtime_data_path if runtime_data_path.exists() else None
        )

        self.results = {}

    def create_few_shot_splits(self, sample_sizes: List[int] = [10, 25, 50]):
        """Create few-shot training splits"""
        self.logger.info("Creating few-shot splits...")

        train_data_path = self.data_dir / 'processed' / 'train_data.json'
        few_shot_dir = self.data_dir / 'few_shot_splits'

        if not train_data_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_data_path}")

        # Use the existing script
        from create_few_shot_splits import create_few_shot_splits

        create_few_shot_splits(
            str(train_data_path),
            str(few_shot_dir),
            sample_sizes=sample_sizes,
            seed=42
        )

        self.logger.info(f"Created few-shot splits for sizes: {sample_sizes}")

    def load_dataset(self, split: str, sample_size: Optional[int] = None) -> DataLoader:
        """Load dataset for a specific split"""
        if split == 'train' and sample_size is not None:
            # Load few-shot training data
            data_path = self.data_dir / 'few_shot_splits' / f'train_{sample_size}.json'
        else:
            # Load regular split
            data_path = self.data_dir / 'processed' / f'{split}_data.json'

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Create dataset
        if sample_size is not None:
            # Create custom split name for few-shot data
            split_name = f'custom_{sample_size}'
        else:
            split_name = split

        dataset = FewShotRankingDataset(
            str(self.data_dir / 'processed'),
            split=split_name,
            max_orders_per_query=10,
            cache_dir=str(self.data_dir / 'graph_cache')
        )

        # Create dataloader
        batch_size = min(16, len(dataset)) if len(dataset) < 50 else 16

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            collate_fn=collate_fewshot_batch,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device == 'cuda' else False
        )

        return dataloader

    def train_model(self,
                   model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model with the given configuration"""

        # Setup optimizer
        trainable_params = model.get_trainable_parameters()
        param_counts = model.get_parameter_count()

        self.logger.info(f"Training with {param_counts['trainable']:,} trainable parameters "
                        f"({param_counts['trainable_ratio']:.4f} of total)")

        optimizer = optim.AdamW(
            trainable_params,
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Setup loss function
        loss_type = config.get('loss_type', 'listwise')
        if loss_type == 'listwise':
            criterion = ListwiseLoss()
        elif loss_type == 'pairwise':
            criterion = PairwiseHingeLoss()
        else:
            criterion = nn.MSELoss()

        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Training loop
        num_epochs = config.get('num_epochs', 50)
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        max_patience = config.get('patience', 10)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                # Move batch to device
                x = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_attr = batch.edge_attr.to(self.device) if batch.edge_attr is not None else None
                orders = batch.orders.to(self.device)
                targets = batch.targets.to(self.device)
                batch_tensor = batch.batch.to(self.device)

                # Forward pass
                outputs = model(x, edge_index, edge_attr, orders, batch_tensor)

                # Compute loss
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    x = batch.x.to(self.device)
                    edge_index = batch.edge_index.to(self.device)
                    edge_attr = batch.edge_attr.to(self.device) if batch.edge_attr is not None else None
                    orders = batch.orders.to(self.device)
                    targets = batch.targets.to(self.device)
                    batch_tensor = batch.batch.to(self.device)

                    outputs = model(x, edge_index, edge_attr, orders, batch_tensor)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    num_val_batches += 1

            val_loss /= num_val_batches
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"Epoch {epoch:3d}: Train Loss {train_loss:.6f}, "
                               f"Val Loss {val_loss:.6f}, LR {optimizer.param_groups[0]['lr']:.2e}")

            if patience_counter >= max_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'num_epochs_trained': epoch + 1
        }

    def run_single_experiment(self,
                             strategy: str,
                             sample_size: int,
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single transfer learning experiment"""

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running {strategy} with {sample_size} samples")
        self.logger.info(f"{'='*60}")

        start_time = time.time()

        # Create model
        model = create_transfer_model(
            self.pretrained_model_path,
            strategy,
            config=config
        ).to(self.device)

        # Load datasets
        train_loader = self.load_dataset('train', sample_size)
        val_loader = self.load_dataset('val')
        test_loader = self.load_dataset('test')

        self.logger.info(f"Train samples: {len(train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(val_loader.dataset)}")
        self.logger.info(f"Test samples: {len(test_loader.dataset)}")

        # Train model
        training_results = self.train_model(model, train_loader, val_loader, config)

        # Evaluate model
        self.logger.info("Evaluating model...")
        test_metrics = self.evaluator.evaluate_model(model, test_loader, self.device)

        # Save model
        model_save_path = self.output_dir / f"model_{strategy}_{sample_size}samples.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'strategy': strategy,
            'sample_size': sample_size,
            'test_metrics': test_metrics,
            'training_results': training_results
        }, model_save_path)

        experiment_time = time.time() - start_time

        results = {
            'strategy': strategy,
            'sample_size': sample_size,
            'training_time': experiment_time,
            'parameter_counts': model.get_parameter_count(),
            'training_results': training_results,
            'test_metrics': test_metrics,
            'model_path': str(model_save_path)
        }

        self.logger.info(f"Experiment completed in {experiment_time:.1f}s")
        self.logger.info(f"Test NDCG@5: {test_metrics.get('ndcg@5', 'N/A'):.4f}")
        self.logger.info(f"Test Spearman: {test_metrics.get('spearman_correlation', 'N/A'):.4f}")

        return results

    def run_all_experiments(self,
                           strategies: List[str] = ['head_only', 'adapters', 'lora'],
                           sample_sizes: List[int] = [10, 25, 50],
                           configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run comprehensive experiments across all strategies and sample sizes"""

        if configs is None:
            configs = {
                'head_only': {
                    'learning_rate': 1e-3,
                    'num_epochs': 30,
                    'weight_decay': 1e-5
                },
                'adapters': {
                    'learning_rate': 1e-3,
                    'num_epochs': 40,
                    'weight_decay': 1e-5,
                    'adapter_bottleneck_dim': 32
                },
                'lora': {
                    'learning_rate': 1e-3,
                    'num_epochs': 40,
                    'weight_decay': 1e-5,
                    'lora_rank': 8,
                    'lora_alpha': 16.0
                }
            }

        # Create few-shot splits if they don't exist
        self.create_few_shot_splits(sample_sizes)

        # Run experiments
        all_results = {}

        for strategy in strategies:
            strategy_results = {}

            for sample_size in sample_sizes:
                config = configs.get(strategy, {}).copy()
                config['device'] = self.device

                try:
                    results = self.run_single_experiment(strategy, sample_size, config)
                    strategy_results[sample_size] = results

                    # Save intermediate results
                    intermediate_path = self.output_dir / f'results_{strategy}_{sample_size}.json'
                    with open(intermediate_path, 'w') as f:
                        json.dump(results, f, indent=2)

                except Exception as e:
                    self.logger.error(f"Experiment failed for {strategy} with {sample_size} samples: {e}")
                    strategy_results[sample_size] = {'error': str(e)}

            all_results[strategy] = strategy_results

        # Save all results
        results_path = self.output_dir / 'all_experiments.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Generate analysis
        analysis = self.analyze_results(all_results)

        # Save analysis
        analysis_path = self.output_dir / 'analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        self.logger.info(f"\nAll experiments completed!")
        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info(f"Analysis saved to: {analysis_path}")

        return {
            'results': all_results,
            'analysis': analysis,
            'output_dir': str(self.output_dir)
        }

    def analyze_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare experimental results"""

        self.logger.info("Analyzing experimental results...")

        analysis = {
            'summary': {},
            'strategy_comparison': {},
            'sample_efficiency': {},
            'parameter_efficiency': {}
        }

        # Extract metrics for comparison
        strategy_metrics = {}
        for strategy, strategy_results in all_results.items():
            strategy_metrics[strategy] = {}
            for sample_size, results in strategy_results.items():
                if 'test_metrics' in results:
                    strategy_metrics[strategy][sample_size] = results['test_metrics']

        # Strategy comparison for each sample size
        for sample_size in [10, 25, 50]:
            size_metrics = {}
            for strategy in strategy_metrics:
                if sample_size in strategy_metrics[strategy]:
                    size_metrics[strategy] = strategy_metrics[strategy][sample_size]

            if size_metrics:
                comparison = self.evaluator.compare_strategies(size_metrics)
                analysis['strategy_comparison'][f'{sample_size}_samples'] = comparison

        # Sample efficiency analysis for each strategy
        for strategy in strategy_metrics:
            if strategy_metrics[strategy]:
                efficiency = evaluate_few_shot_sample_efficiency(strategy_metrics[strategy])
                analysis['sample_efficiency'][strategy] = efficiency

        # Parameter efficiency analysis
        param_efficiency = {}
        for strategy, strategy_results in all_results.items():
            for sample_size, results in strategy_results.items():
                if 'parameter_counts' in results and 'test_metrics' in results:
                    key = f"{strategy}_{sample_size}"
                    param_efficiency[key] = {
                        'trainable_params': results['parameter_counts']['trainable'],
                        'trainable_ratio': results['parameter_counts']['trainable_ratio'],
                        'ndcg@5': results['test_metrics'].get('ndcg@5', 0),
                        'spearman_correlation': results['test_metrics'].get('spearman_correlation', 0)
                    }

        analysis['parameter_efficiency'] = param_efficiency

        # Overall summary
        best_strategies = {}
        for sample_size in [10, 25, 50]:
            best_ndcg = 0
            best_strategy = None
            for strategy in all_results:
                if (sample_size in all_results[strategy] and
                    'test_metrics' in all_results[strategy][sample_size]):
                    ndcg = all_results[strategy][sample_size]['test_metrics'].get('ndcg@5', 0)
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_strategy = strategy

            best_strategies[f'{sample_size}_samples'] = {
                'strategy': best_strategy,
                'ndcg@5': best_ndcg
            }

        analysis['summary']['best_strategies_by_sample_size'] = best_strategies

        return analysis

def main():
    parser = argparse.ArgumentParser(description='Run few-shot transfer learning experiments')
    parser.add_argument('--pretrained-model', required=True,
                       help='Path to pretrained model')
    parser.add_argument('--data-dir', required=True,
                       help='Path to IMDB data directory')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--strategies', nargs='+',
                       default=['head_only', 'adapters', 'lora'],
                       help='Transfer learning strategies to test')
    parser.add_argument('--sample-sizes', nargs='+', type=int,
                       default=[10, 25, 50],
                       help='Few-shot sample sizes to test')
    parser.add_argument('--device', default=None,
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Create experiment runner
    runner = FewShotExperimentRunner(
        args.pretrained_model,
        args.data_dir,
        args.output_dir,
        args.device
    )

    # Run experiments
    results = runner.run_all_experiments(
        strategies=args.strategies,
        sample_sizes=args.sample_sizes
    )

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Print key results
    for strategy in args.strategies:
        print(f"\n{strategy.upper()}:")
        for sample_size in args.sample_sizes:
            if (strategy in results['results'] and
                sample_size in results['results'][strategy] and
                'test_metrics' in results['results'][strategy][sample_size]):

                metrics = results['results'][strategy][sample_size]['test_metrics']
                ndcg5 = metrics.get('ndcg@5', 0)
                spearman = metrics.get('spearman_correlation', 0)
                print(f"  {sample_size:2d} samples: NDCG@5={ndcg5:.4f}, Spearman={spearman:.4f}")

if __name__ == "__main__":
    main()