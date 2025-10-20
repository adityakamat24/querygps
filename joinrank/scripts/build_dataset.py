#!/usr/bin/env python3
"""
Build training datasets from measured runtimes and query features
"""
import json
import torch
import psycopg2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
sys.path.append('features')
sys.path.append('model')

from graph_builder import QueryGraphBuilder, JoinOrderEncoder

def get_table_stats(db_name: str, db_user: str) -> dict:
    """Query database for table cardinalities"""
    conn = psycopg2.connect(dbname=db_name, user=db_user)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("""
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public'
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    stats = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        
        cursor.execute(f"""
            SELECT pg_total_relation_size('{table}')::bigint / (1024*1024) as size_mb
        """)
        size_mb = cursor.fetchone()[0]
        
        stats[table] = {
            'cardinality': count,
            'size_mb': float(size_mb)
        }
    
    cursor.close()
    conn.close()
    
    return stats

def build_dataset(args):
    """Build dataset from queries, orders, and runtimes"""
    
    # Load runtime measurements
    with open(args.runtimes) as f:
        runtimes_data = json.load(f)
    
    print(f"Loaded {len(runtimes_data)} runtime measurements")
    
    # Get table statistics
    print("Fetching table statistics from database...")
    table_stats = get_table_stats(args.db_name, args.db_user)
    print(f"Got stats for {len(table_stats)} tables")
    
    # Initialize builders
    graph_builder = QueryGraphBuilder(table_stats)
    order_encoder = JoinOrderEncoder(max_tables=10)
    
    # Group runtimes by query
    queries_dict = {}
    for runtime_entry in runtimes_data:
        query_name = runtime_entry['query_name']
        
        # Extract base query name (remove _order_XXX suffix)
        parts = query_name.split('_order_')
        if len(parts) > 1:
            base_name = parts[0]
        else:
            base_name = query_name
        
        if base_name not in queries_dict:
            queries_dict[base_name] = {
                'orders': [],
                'runtimes': [],
                'sql_files': []
            }
        
        queries_dict[base_name]['orders'].append(runtime_entry)
        queries_dict[base_name]['runtimes'].append(runtime_entry['runtime_ms'])
        queries_dict[base_name]['sql_file'] = runtime_entry.get('sql_file', '')
    
    print(f"Organized into {len(queries_dict)} unique queries")
    
    # Process each query
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graphs_dir = output_dir / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    
    dataset = []
    query_ids = sorted(queries_dict.keys())
    
    for query_id in tqdm(query_ids, desc="Processing queries"):
        query_info = queries_dict[query_id]
        
        # Load original query SQL
        query_file = Path(args.queries_dir) / f"{query_id}.sql"
        if not query_file.exists():
            print(f"Warning: Query file not found: {query_file}")
            continue
        
        with open(query_file) as f:
            query_sql = f.read()
        
        # Build graph
        graph = graph_builder.build_graph(query_sql)
        if graph is None:
            print(f"Warning: Could not build graph for {query_id}")
            continue
        
        # Convert to PyG format
        graph_data = graph_builder.to_pyg_data(graph)
        
        # Save graph
        graph_file = graphs_dir / f"{query_id}_graph.pt"
        torch.save(graph_data, graph_file)
        
        # Encode join orders (simplified - use random for now)
        tables = list(graph.nodes())
        order_encodings = []
        for _ in range(len(query_info['orders'])):
            # Simple random encoding (in practice, extract actual order from SQL)
            order_enc = torch.randn(32)  # Simple random encoding
            order_encodings.append(order_enc.tolist())
        
        # Add to dataset
        dataset.append({
            'query_id': query_id,
            'graph_file': str(graph_file.relative_to(output_dir)),
            'orders': order_encodings,
            'runtimes': query_info['runtimes'],
            'num_tables': len(tables),
            'num_orders': len(query_info['orders'])
        })
    
    print(f"\nProcessed {len(dataset)} queries")
    
    # Split into train/val/test
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    
    n_train = int(len(dataset) * args.train_split)
    n_val = int(len(dataset) * args.val_split)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    test_data = [dataset[i] for i in test_indices]
    
    print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Save splits
    with open(output_dir / 'train_data.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / 'val_data.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(output_dir / 'test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Save table stats
    with open(output_dir / 'table_stats.json', 'w') as f:
        json.dump(table_stats, f, indent=2)
    
    print(f"\nDataset saved to {output_dir}")
    print(f"  - Train: {len(train_data)} queries")
    print(f"  - Val: {len(val_data)} queries")
    print(f"  - Test: {len(test_data)} queries")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-name', required=True, help='Database name')
    parser.add_argument('--db-user', default='postgres', help='Database user')
    parser.add_argument('--runtimes', required=True, help='Runtime measurements JSON')
    parser.add_argument('--queries-dir', required=True, help='Directory with query SQL files')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--train-split', type=float, default=0.7, help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    args = parser.parse_args()
    
    build_dataset(args)

if __name__ == "__main__":
    main()