#!/usr/bin/env python3
"""
Build ranking dataset from measured runtimes with multiple join orders per query
"""
import json
import argparse
import psycopg2
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import json
import psycopg2
import random
from pathlib import Path
from tqdm import tqdm
import argparse
from torch_geometric.data import Data
import hashlib

def get_table_stats(db_name: str, db_user: str, db_password: str = None) -> dict:
    """Fetch table statistics from PostgreSQL"""
    conn_params = {
        'dbname': db_name,
        'user': db_user,
        'host': 'localhost'
    }
    if db_password:
        conn_params['password'] = db_password
    
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    
    # Get table statistics from pg_class
    cur.execute("""
        SELECT 
            c.relname as table_name,
            c.reltuples::bigint as row_count,
            pg_total_relation_size(c.oid) as total_bytes
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'r'
          AND n.nspname = 'public'
          AND c.reltuples > 0
        ORDER BY c.relname;
    """)
    
    stats = {}
    for row in cur.fetchall():
        table_name, row_count, total_bytes = row
        stats[table_name] = {
            'row_count': int(row_count),
            'size_bytes': int(total_bytes),
            'size_mb': round(total_bytes / (1024 * 1024), 2)
        }
    
    cur.close()
    conn.close()
    
    print(f"Got stats for {len(stats)} tables")
    return stats

def build_ranking_dataset(args):
    """Build ranking dataset from runtimes"""
    
    # Load runtime measurements
    with open(args.runtimes) as f:
        runtimes = json.load(f)
    print(f"Loaded {len(runtimes)} runtime measurements")
    
    # Get table statistics
    table_stats = get_table_stats(args.db_name, args.db_user, args.db_password)
    
    # Group runtimes by base query
    query_groups = {}
    for item in runtimes:
        # Extract base query name from query_name
        # e.g., "q001_q_customer_orders_lineitem_v0_order_000" -> "q001_q_customer_orders_lineitem_v0"
        query_name = item.get('query_name', '')
        
        # Remove the _order_XXX suffix to get base query
        if '_order_' in query_name:
            base_query = query_name.rsplit('_order_', 1)[0]
        else:
            base_query = query_name
        
        if base_query not in query_groups:
            query_groups[base_query] = []
        query_groups[base_query].append(item)
    
    print(f"Organized into {len(query_groups)} unique queries")
    
    # Process each query group
    all_examples = []
    for base_query, variants in tqdm(query_groups.items(), desc="Processing queries"):
        # Sort by runtime to create ranking
        variants_sorted = sorted(variants, key=lambda x: x['runtime_ms'])
        
        # Create pairwise comparisons
        for i in range(len(variants_sorted)):
            for j in range(i + 1, len(variants_sorted)):
                better = variants_sorted[i]
                worse = variants_sorted[j]
                
                # Only include if runtime difference is significant (>5%)
                if worse['runtime_ms'] > better['runtime_ms'] * 1.05:
                    example = {
                        'base_query': base_query,
                        'better_order': better['query_name'],
                        'worse_order': worse['query_name'],
                        'better_runtime_ms': better['runtime_ms'],
                        'worse_runtime_ms': worse['runtime_ms'],
                        'speedup': worse['runtime_ms'] / better['runtime_ms'],
                        'better_sql_file': better['sql_file'],
                        'worse_sql_file': worse['sql_file']
                    }
                    all_examples.append(example)
    
    print(f"Created {len(all_examples)} pairwise ranking examples")
    
    # Split into train/val/test by base query
    base_queries = list(query_groups.keys())
    random.shuffle(base_queries)
    
    n_train = int(len(base_queries) * args.train_split)
    n_val = int(len(base_queries) * args.val_split)
    
    train_queries = set(base_queries[:n_train])
    val_queries = set(base_queries[n_train:n_train + n_val])
    test_queries = set(base_queries[n_train + n_val:])
    
    train_data = [ex for ex in all_examples if ex['base_query'] in train_queries]
    val_data = [ex for ex in all_examples if ex['base_query'] in val_queries]
    test_data = [ex for ex in all_examples if ex['base_query'] in test_queries]
    
    print(f"Split: Train={len(train_data)} examples, Val={len(val_data)} examples, Test={len(test_data)} examples")
    print(f"Query split: Train={len(train_queries)}, Val={len(val_queries)}, Test={len(test_queries)}")
    
    # Save datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'train_data.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / 'val_data.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(output_dir / 'test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    with open(output_dir / 'table_stats.json', 'w') as f:
        json.dump(table_stats, f, indent=2)
    
    print(f"Dataset saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Build ranking dataset from runtimes')
    parser.add_argument('--db-name', type=str, required=True, help='Database name')
    parser.add_argument('--db-user', type=str, required=True, help='Database user')
    parser.add_argument('--db-password', type=str, default=None, help='Database password')
    parser.add_argument('--runtimes', type=str, required=True, help='Runtime measurements JSON')
    parser.add_argument('--queries-dir', type=str, required=True, help='Directory with SQL queries')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--train-split', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation set ratio')
    
    args = parser.parse_args()
    build_ranking_dataset(args)

if __name__ == '__main__':
    main()