#!/usr/bin/env python3
"""
Complete experimental comparison:
- Head-Only Transfer Learning
- Adapters Transfer Learning
- LoRA Transfer Learning
- From-Scratch Baseline
"""
import sys
import os
sys.path.append('scripts')

import torch
import json
import subprocess
from pathlib import Path
import time

def run_transfer_learning_experiments():
    """Run all transfer learning experiments"""
    print("\n" + "="*80)
    print("RUNNING TRANSFER LEARNING EXPERIMENTS (HEAD-ONLY, ADAPTERS, LORA)")
    print("="*80)

    result = subprocess.run(
        [sys.executable, 'run_simplified_advanced_experiments.py'],
        cwd='.',
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("[SUCCESS] Transfer learning experiments completed")
        print(result.stdout)
    else:
        print("[ERROR] Transfer learning experiments failed")
        print(result.stderr)
        return False

    return True

def run_from_scratch_baseline():
    """Run from-scratch baseline experiments"""
    print("\n" + "="*80)
    print("RUNNING FROM-SCRATCH BASELINE EXPERIMENTS")
    print("="*80)

    result = subprocess.run(
        [sys.executable, 'run_from_scratch_baseline.py'],
        cwd='.',
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("[SUCCESS] Baseline experiments completed")
        print(result.stdout)
    else:
        print("[ERROR] Baseline experiments failed")
        print(result.stderr)
        return False

    return True

def create_comparison_report():
    """Create comprehensive comparison report"""
    print("\n" + "="*80)
    print("CREATING COMPARISON REPORT")
    print("="*80)

    # Load transfer learning results
    transfer_results_path = Path('results/enhanced_experiments/all_enhanced_results.json')
    if not transfer_results_path.exists():
        print("[ERROR] Transfer learning results not found")
        return

    with open(transfer_results_path) as f:
        transfer_results = json.load(f)

    # Load baseline results
    baseline_results_path = Path('results/from_scratch_baseline/all_results.json')
    if not baseline_results_path.exists():
        print("[ERROR] Baseline results not found")
        return

    with open(baseline_results_path) as f:
        baseline_results = json.load(f)

    # Create comparison
    comparison = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sample_sizes': [10, 25, 50],
        'strategies': ['head_only', 'adapters', 'lora', 'from_scratch'],
        'results_by_sample_size': {}
    }

    for sample_size in [10, 25, 50]:
        sample_size_str = str(sample_size)
        comparison['results_by_sample_size'][sample_size] = {
            'head_only': transfer_results.get('head_only', {}).get(sample_size_str, {}),
            'adapters': transfer_results.get('adapters', {}).get(sample_size_str, {}),
            'lora': transfer_results.get('lora', {}).get(sample_size_str, {}),
            'from_scratch': baseline_results.get(sample_size_str, {}).get('final_metrics', {})
        }

    # Save comparison
    os.makedirs('results/final_comparison', exist_ok=True)
    with open('results/final_comparison/complete_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print comparison table
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)

    print("\nNDCG@5 Scores:")
    print("-" * 80)
    print(f"{'Sample Size':<15} {'Head-Only':<15} {'Adapters':<15} {'LoRA':<15} {'From-Scratch':<15}")
    print("-" * 80)

    for sample_size in [10, 25, 50]:
        results = comparison['results_by_sample_size'][sample_size]

        head_only_ndcg = results['head_only'].get('ndcg@5', 'ERROR')
        adapters_ndcg = results['adapters'].get('ndcg@5', 'ERROR')
        lora_ndcg = results['lora'].get('ndcg@5', 'ERROR')
        baseline_ndcg = results['from_scratch'].get('ndcg@5', 'ERROR')

        def format_score(score):
            if isinstance(score, (int, float)):
                return f"{score:.4f}"
            else:
                return str(score)

        print(f"{sample_size:<15} {format_score(head_only_ndcg):<15} {format_score(adapters_ndcg):<15} "
              f"{format_score(lora_ndcg):<15} {format_score(baseline_ndcg):<15}")

    print("\nSpearman Correlation:")
    print("-" * 80)
    print(f"{'Sample Size':<15} {'Head-Only':<15} {'Adapters':<15} {'LoRA':<15} {'From-Scratch':<15}")
    print("-" * 80)

    for sample_size in [10, 25, 50]:
        results = comparison['results_by_sample_size'][sample_size]

        head_only_spear = results['head_only'].get('spearman', 'ERROR')
        adapters_spear = results['adapters'].get('spearman', 'ERROR')
        lora_spear = results['lora'].get('spearman', 'ERROR')
        baseline_spear = results['from_scratch'].get('spearman', 'ERROR')

        def format_score(score):
            if isinstance(score, (int, float)):
                return f"{score:+.4f}"
            else:
                return str(score)

        print(f"{sample_size:<15} {format_score(head_only_spear):<15} {format_score(adapters_spear):<15} "
              f"{format_score(lora_spear):<15} {format_score(baseline_spear):<15}")

    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)

    # Find best strategy for each sample size
    for sample_size in [10, 25, 50]:
        results = comparison['results_by_sample_size'][sample_size]

        scores = {}
        for strategy in ['head_only', 'adapters', 'lora', 'from_scratch']:
            ndcg = results[strategy].get('ndcg@5', 0)
            if isinstance(ndcg, (int, float)):
                scores[strategy] = ndcg

        if scores:
            best_strategy = max(scores, key=scores.get)
            best_score = scores[best_strategy]
            print(f"\n{sample_size} samples: {best_strategy.upper()} performs best with NDCG@5 = {best_score:.4f}")

            # Compare transfer vs from-scratch
            if 'from_scratch' in scores:
                for transfer_strategy in ['head_only', 'adapters', 'lora']:
                    if transfer_strategy in scores:
                        improvement = ((scores[transfer_strategy] - scores['from_scratch']) / scores['from_scratch']) * 100
                        if improvement > 0:
                            print(f"  - {transfer_strategy} is {improvement:+.2f}% better than from-scratch")
                        else:
                            print(f"  - {transfer_strategy} is {improvement:.2f}% worse than from-scratch")

    print(f"\nFull comparison saved to: results/final_comparison/complete_comparison.json")

def main():
    """Run complete experimental pipeline"""
    start_time = time.time()

    print("\n" + "="*80)
    print("COMPLETE EXPERIMENTAL COMPARISON")
    print("Few-Shot Transfer Learning for Join Order Selection")
    print("="*80)

    # Step 1: Run transfer learning experiments
    if not run_transfer_learning_experiments():
        print("\n[FAILED] Could not complete transfer learning experiments")
        return

    # Step 2: Run from-scratch baseline
    if not run_from_scratch_baseline():
        print("\n[FAILED] Could not complete baseline experiments")
        return

    # Step 3: Create comparison report
    create_comparison_report()

    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"COMPLETE EXPERIMENTAL PIPELINE FINISHED IN {total_time:.2f}s")
    print("="*80)

if __name__ == '__main__':
    main()
