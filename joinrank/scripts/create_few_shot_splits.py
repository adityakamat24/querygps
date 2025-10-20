#!/usr/bin/env python3
"""
Create few-shot training splits from IMDB data
Samples 10, 25, and 50 examples for transfer learning experiments
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

def create_few_shot_splits(
    train_data_path: str,
    output_dir: str,
    sample_sizes: list = [10, 25, 50],
    seed: int = 42
):
    """
    Create few-shot splits by sampling from training data
    
    Args:
        train_data_path: Path to full training data JSON
        output_dir: Directory to save few-shot splits
        sample_sizes: List of sample sizes to create
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load full training data
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    
    print(f"Loaded {len(train_data)} training examples")
    
    # Group by query to ensure diverse sampling
    query_groups = defaultdict(list)
    for example in train_data:
        # Handle different possible key names for query identifier
        query_id = example.get('query_id') or example.get('base_query') or example.get('query')
        if query_id is None:
            # If no query identifier, create one from the example index
            query_id = f"query_{len(query_groups)}"
        query_groups[query_id].append(example)
    
    print(f"Found {len(query_groups)} unique queries")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create splits for each sample size
    for n_samples in sample_sizes:
        if n_samples > len(train_data):
            print(f"Warning: Requested {n_samples} samples but only {len(train_data)} available")
            n_samples = len(train_data)
        
        # Strategy 1: Sample diverse queries first
        n_queries = min(n_samples, len(query_groups))
        sampled_queries = random.sample(list(query_groups.keys()), n_queries)
        
        # Get one example from each sampled query
        few_shot_data = []
        for query_id in sampled_queries:
            examples = query_groups[query_id]
            example = random.choice(examples)
            # Ensure query_id field exists for compatibility
            if 'query_id' not in example:
                example = example.copy()
                example['query_id'] = query_id
            few_shot_data.append(example)
        
        # If we need more samples, add more from existing queries
        while len(few_shot_data) < n_samples:
            remaining = [ex for ex in train_data if ex not in few_shot_data]
            if not remaining:
                break
            example = random.choice(remaining)
            # Ensure query_id field exists for compatibility
            if 'query_id' not in example:
                example = example.copy()
                query_id = example.get('base_query') or example.get('query') or f"query_{len(few_shot_data)}"
                example['query_id'] = query_id
            few_shot_data.append(example)
        
        # Save
        output_file = output_path / f"train_{n_samples}.json"
        with open(output_file, 'w') as f:
            json.dump(few_shot_data, f, indent=2)
        
        unique_queries = len(set(ex['query_id'] for ex in few_shot_data))
        print(f"Created {output_file}: {len(few_shot_data)} examples from {unique_queries} queries")

def main():
    parser = argparse.ArgumentParser(description='Create few-shot training splits')
    parser.add_argument('--train-data', required=True, help='Path to full training data JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory for few-shot splits')
    parser.add_argument('--sample-sizes', nargs='+', type=int, default=[10, 25, 50],
                        help='Sample sizes to create')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_few_shot_splits(
        args.train_data,
        args.output_dir,
        args.sample_sizes,
        args.seed
    )

if __name__ == "__main__":
    main()