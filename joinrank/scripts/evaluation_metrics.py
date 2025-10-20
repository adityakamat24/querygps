#!/usr/bin/env python3
"""
Evaluation metrics for join order selection
Includes NDCG@k, ranking correlation, and runtime improvement metrics
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.stats import spearmanr, kendalltau
import json
from pathlib import Path

def ndcg_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Compute NDCG@k for ranking evaluation

    Args:
        scores: Predicted scores [batch_size, num_items]
        targets: True relevance scores [batch_size, num_items]
        k: Number of top items to consider

    Returns:
        NDCG@k score
    """
    batch_size = scores.size(0)
    ndcg_scores = []

    for i in range(batch_size):
        # Get top-k predictions
        _, pred_indices = torch.topk(scores[i], k)

        # Get relevance of predicted top-k items
        pred_relevance = targets[i][pred_indices].cpu().numpy()

        # Get ideal top-k (sorted by true relevance)
        _, ideal_indices = torch.topk(targets[i], k)
        ideal_relevance = targets[i][ideal_indices].cpu().numpy()

        # Compute DCG for predictions
        dcg = 0.0
        for j, rel in enumerate(pred_relevance):
            dcg += (2**rel - 1) / np.log2(j + 2)  # j+2 because log2(1) = 0

        # Compute IDCG (ideal DCG)
        idcg = 0.0
        for j, rel in enumerate(ideal_relevance):
            idcg += (2**rel - 1) / np.log2(j + 2)

        # NDCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return np.mean(ndcg_scores)

def ranking_correlation(scores: torch.Tensor, targets: torch.Tensor, method: str = 'spearman') -> float:
    """
    Compute ranking correlation between predicted and true rankings

    Args:
        scores: Predicted scores [batch_size, num_items]
        targets: True scores [batch_size, num_items]
        method: 'spearman' or 'kendall'

    Returns:
        Average correlation across batch
    """
    batch_size = scores.size(0)
    correlations = []

    for i in range(batch_size):
        pred_ranks = torch.argsort(torch.argsort(scores[i], descending=True)).cpu().numpy()
        true_ranks = torch.argsort(torch.argsort(targets[i], descending=True)).cpu().numpy()

        if method == 'spearman':
            corr, _ = spearmanr(pred_ranks, true_ranks)
        elif method == 'kendall':
            corr, _ = kendalltau(pred_ranks, true_ranks)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        if not np.isnan(corr):
            correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0

def mean_average_precision(scores: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Mean Average Precision for ranking

    Args:
        scores: Predicted scores [batch_size, num_items]
        targets: Binary relevance [batch_size, num_items] or continuous (>threshold = relevant)
        threshold: Threshold for binary relevance if targets are continuous

    Returns:
        MAP score
    """
    batch_size = scores.size(0)
    ap_scores = []

    for i in range(batch_size):
        # Sort by predicted scores (descending)
        sorted_indices = torch.argsort(scores[i], descending=True)
        sorted_targets = targets[i][sorted_indices].cpu().numpy()

        # Binarize targets if needed
        if threshold is not None:
            relevant = (sorted_targets > threshold).astype(int)
        else:
            relevant = sorted_targets.astype(int)

        if np.sum(relevant) == 0:
            continue

        # Compute average precision
        ap = 0.0
        num_relevant = 0

        for j, rel in enumerate(relevant):
            if rel == 1:
                num_relevant += 1
                precision_at_j = num_relevant / (j + 1)
                ap += precision_at_j

        ap_scores.append(ap / np.sum(relevant))

    return np.mean(ap_scores) if ap_scores else 0.0

def runtime_improvement_metrics(predicted_runtimes: np.ndarray,
                               optimal_runtimes: np.ndarray,
                               baseline_runtimes: np.ndarray) -> Dict[str, float]:
    """
    Compute runtime improvement metrics

    Args:
        predicted_runtimes: Runtimes of predicted best orders
        optimal_runtimes: Runtimes of optimal orders
        baseline_runtimes: Runtimes of baseline (e.g., DB optimizer) orders

    Returns:
        Dictionary of metrics
    """
    # Relative improvement over baseline
    baseline_improvement = (baseline_runtimes - predicted_runtimes) / baseline_runtimes

    # Relative distance from optimal
    optimality_gap = (predicted_runtimes - optimal_runtimes) / optimal_runtimes

    # Percentage of queries where prediction beats baseline
    beat_baseline_pct = np.mean(predicted_runtimes < baseline_runtimes) * 100

    # Percentage of queries within X% of optimal
    within_5pct_optimal = np.mean(optimality_gap <= 0.05) * 100
    within_10pct_optimal = np.mean(optimality_gap <= 0.10) * 100

    return {
        'mean_baseline_improvement': np.mean(baseline_improvement),
        'median_baseline_improvement': np.median(baseline_improvement),
        'mean_optimality_gap': np.mean(optimality_gap),
        'median_optimality_gap': np.median(optimality_gap),
        'beat_baseline_percentage': beat_baseline_pct,
        'within_5pct_optimal': within_5pct_optimal,
        'within_10pct_optimal': within_10pct_optimal,
        'geometric_mean_speedup': np.exp(np.mean(np.log(baseline_runtimes / predicted_runtimes)))
    }

class FewShotEvaluator:
    """
    Comprehensive evaluator for few-shot transfer learning experiments
    """

    def __init__(self, runtime_data_path: Optional[str] = None):
        self.runtime_data = None
        if runtime_data_path and Path(runtime_data_path).exists():
            with open(runtime_data_path, 'r') as f:
                self.runtime_data = json.load(f)

    def evaluate_model(self,
                      model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      device: str = 'cuda',
                      k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a model

        Args:
            model: Trained model to evaluate
            dataloader: Test data loader
            device: Device to run evaluation on
            k_values: List of k values for NDCG@k computation

        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()

        all_scores = []
        all_targets = []
        all_query_ids = []

        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch (assuming it contains x, edge_index, edge_attr, orders, targets, batch, query_ids)
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                edge_attr = batch.edge_attr.to(device) if batch.edge_attr is not None else None
                orders = batch.orders.to(device)
                targets = batch.targets.to(device)
                batch_tensor = batch.batch.to(device)

                # Forward pass
                scores = model(x, edge_index, edge_attr, orders, batch_tensor)

                all_scores.append(scores.cpu())
                all_targets.append(targets.cpu())

                if hasattr(batch, 'query_ids'):
                    all_query_ids.extend(batch.query_ids)

        # Concatenate all results
        all_scores = torch.cat(all_scores, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute ranking metrics
        metrics = {}

        # NDCG@k for different k values
        for k in k_values:
            if k <= all_scores.size(1):
                metrics[f'ndcg@{k}'] = ndcg_at_k(all_scores, all_targets, k)

        # Ranking correlations
        metrics['spearman_correlation'] = ranking_correlation(all_scores, all_targets, 'spearman')
        metrics['kendall_correlation'] = ranking_correlation(all_scores, all_targets, 'kendall')

        # MAP
        metrics['map'] = mean_average_precision(all_scores, all_targets)

        # Top-1 accuracy (best predicted order matches best true order)
        pred_best = torch.argmax(all_scores, dim=1)
        true_best = torch.argmax(all_targets, dim=1)
        metrics['top1_accuracy'] = (pred_best == true_best).float().mean().item()

        # Runtime metrics (if runtime data available)
        if self.runtime_data and all_query_ids:
            runtime_metrics = self._compute_runtime_metrics(all_scores, all_query_ids)
            metrics.update(runtime_metrics)

        return metrics

    def _compute_runtime_metrics(self, scores: torch.Tensor, query_ids: List[str]) -> Dict[str, float]:
        """Compute runtime-based metrics using actual execution times"""
        predicted_runtimes = []
        optimal_runtimes = []
        baseline_runtimes = []

        for i, query_id in enumerate(query_ids):
            if query_id not in self.runtime_data:
                continue

            query_runtimes = self.runtime_data[query_id]

            # Get predicted best order
            pred_best_idx = torch.argmax(scores[i]).item()

            # Get runtimes
            if pred_best_idx < len(query_runtimes['runtimes']):
                predicted_runtimes.append(query_runtimes['runtimes'][pred_best_idx])
                optimal_runtimes.append(min(query_runtimes['runtimes']))

                # Assuming baseline is the first runtime (original order)
                baseline_runtimes.append(query_runtimes['runtimes'][0])

        if predicted_runtimes:
            return runtime_improvement_metrics(
                np.array(predicted_runtimes),
                np.array(optimal_runtimes),
                np.array(baseline_runtimes)
            )
        else:
            return {}

    def compare_strategies(self,
                          strategy_results: Dict[str, Dict[str, float]],
                          save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare results from different transfer learning strategies

        Args:
            strategy_results: Dict mapping strategy names to their metrics
            save_path: Optional path to save comparison results

        Returns:
            Comparison analysis
        """
        comparison = {
            'strategies': list(strategy_results.keys()),
            'metrics': {},
            'best_strategy': {},
            'summary': {}
        }

        # For each metric, find the best strategy
        all_metrics = set()
        for results in strategy_results.values():
            all_metrics.update(results.keys())

        for metric in all_metrics:
            values = {}
            for strategy, results in strategy_results.items():
                if metric in results:
                    values[strategy] = results[metric]

            if values:
                comparison['metrics'][metric] = values

                # Find best strategy for this metric (higher is usually better)
                best_strategy = max(values.keys(), key=lambda k: values[k])
                comparison['best_strategy'][metric] = {
                    'strategy': best_strategy,
                    'value': values[best_strategy]
                }

        # Summary statistics
        strategy_wins = {strategy: 0 for strategy in strategy_results.keys()}
        for best_info in comparison['best_strategy'].values():
            strategy_wins[best_info['strategy']] += 1

        comparison['summary'] = {
            'strategy_wins': strategy_wins,
            'overall_best': max(strategy_wins.keys(), key=lambda k: strategy_wins[k])
        }

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(comparison, f, indent=2)

        return comparison

def evaluate_few_shot_sample_efficiency(results_by_sample_size: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    """
    Analyze how performance scales with few-shot sample size

    Args:
        results_by_sample_size: Dict mapping sample sizes to metric dictionaries

    Returns:
        Sample efficiency analysis
    """
    sample_sizes = sorted(results_by_sample_size.keys())

    efficiency_analysis = {
        'sample_sizes': sample_sizes,
        'metrics_vs_samples': {},
        'improvement_rates': {},
        'saturation_analysis': {}
    }

    # Track how each metric changes with sample size
    all_metrics = set()
    for results in results_by_sample_size.values():
        all_metrics.update(results.keys())

    for metric in all_metrics:
        values = []
        for size in sample_sizes:
            if metric in results_by_sample_size[size]:
                values.append(results_by_sample_size[size][metric])
            else:
                values.append(None)

        efficiency_analysis['metrics_vs_samples'][metric] = {
            'sample_sizes': sample_sizes,
            'values': values
        }

        # Compute improvement rates between consecutive sample sizes
        improvements = []
        valid_values = [v for v in values if v is not None]
        if len(valid_values) > 1:
            for i in range(1, len(valid_values)):
                if valid_values[i-1] != 0:
                    improvement = (valid_values[i] - valid_values[i-1]) / valid_values[i-1]
                    improvements.append(improvement)

        efficiency_analysis['improvement_rates'][metric] = improvements

        # Check for saturation (diminishing returns)
        if len(improvements) >= 2:
            recent_improvements = improvements[-2:]
            if all(imp < 0.05 for imp in recent_improvements):  # Less than 5% improvement
                efficiency_analysis['saturation_analysis'][metric] = 'saturated'
            elif recent_improvements[-1] < recent_improvements[-2]:
                efficiency_analysis['saturation_analysis'][metric] = 'diminishing'
            else:
                efficiency_analysis['saturation_analysis'][metric] = 'improving'

    return efficiency_analysis

if __name__ == "__main__":
    # Test the evaluation metrics
    print("Testing evaluation metrics...")

    # Create dummy data
    batch_size = 4
    num_orders = 10

    # Dummy scores and targets
    scores = torch.randn(batch_size, num_orders)
    targets = torch.randn(batch_size, num_orders)

    # Test NDCG@k
    ndcg5 = ndcg_at_k(scores, targets, k=5)
    print(f"NDCG@5: {ndcg5:.4f}")

    # Test ranking correlation
    spearman_corr = ranking_correlation(scores, targets, 'spearman')
    print(f"Spearman correlation: {spearman_corr:.4f}")

    # Test MAP
    binary_targets = (targets > 0).float()
    map_score = mean_average_precision(scores, binary_targets)
    print(f"MAP: {map_score:.4f}")

    # Test runtime metrics
    pred_runtimes = np.random.uniform(0.5, 2.0, 20)
    opt_runtimes = np.random.uniform(0.1, 0.5, 20)
    baseline_runtimes = np.random.uniform(1.0, 3.0, 20)

    runtime_metrics = runtime_improvement_metrics(pred_runtimes, opt_runtimes, baseline_runtimes)
    print("\nRuntime metrics:")
    for metric, value in runtime_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nEvaluation metrics test completed!")