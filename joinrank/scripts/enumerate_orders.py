#!/usr/bin/env python3
"""
Simple join order enumeration using table reordering
"""
import random
from pathlib import Path
import json
import re
import argparse

def extract_query_info(sql):
    """Extract tables and conditions from query"""
    # Extract FROM clause
    from_match = re.search(r'FROM\s+(.*?)\s+WHERE', sql, re.IGNORECASE | re.DOTALL)
    if not from_match:
        return None, None
    
    tables = [t.strip() for t in from_match.group(1).split(',')]
    
    # Extract WHERE clause
    where_match = re.search(r'WHERE\s+(.*?)(?:;|$)', sql, re.IGNORECASE | re.DOTALL)
    where_clause = where_match.group(1).strip() if where_match else ""
    
    return tables, where_clause

def generate_join_order_sql(tables, where_clause, order):
    """Generate SQL with specific table order"""
    # Reorder tables according to 'order' (list of indices)
    ordered_tables = [tables[i] for i in order]
    table_list = ', '.join(ordered_tables)
    
    sql = f"""SET join_collapse_limit = 1;
SET from_collapse_limit = 1;

SELECT COUNT(*)
FROM {table_list}
WHERE {where_clause};
"""
    return sql

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-file', required=True)
    parser.add_argument('--n-orders', type=int, default=64)
    parser.add_argument('--output-dir', default='data/tpch/join_orders')
    args = parser.parse_args()
    
    # Read query
    with open(args.query_file) as f:
        original_sql = f.read()
    
    tables, where_clause = extract_query_info(original_sql)
    
    if not tables or not where_clause:
        print("Could not parse query")
        return
    
    print(f"Query has {len(tables)} tables: {tables}")
    
    # Generate different orderings
    # Generate different orderings
    n_tables = len(tables)
    indices = list(range(n_tables))

    # Calculate maximum possible permutations
    import math
    max_permutations = math.factorial(n_tables)

    # Adjust n_orders if it exceeds possible permutations
    actual_n_orders = min(args.n_orders, max_permutations)
    if actual_n_orders < args.n_orders:
        print(f"Note: Only {max_permutations} unique permutations possible for {n_tables} tables")
        print(f"Generating {actual_n_orders} orders instead of {args.n_orders}")

    # Get n_orders random permutations
    random.seed(42)
    orders = []

    # Add some systematic orders first
    orders.append(list(indices))  # Original order
    if n_tables > 1:
        orders.append(list(reversed(indices)))  # Reversed

    # Add random permutations
    while len(orders) < actual_n_orders:
        perm = indices.copy()
        random.shuffle(perm)
        if perm not in orders:
            orders.append(perm)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    query_name = Path(args.query_file).stem
    
    # Generate SQL files
    for i, order in enumerate(orders):
        sql = generate_join_order_sql(tables, where_clause, order)
        
        output_file = output_dir / f"{query_name}_order_{i:03d}.sql"
        with open(output_file, 'w') as f:
            f.write(sql)
    
    # Save metadata
    metadata = {
        'query_name': query_name,
        'num_tables': n_tables,
        'num_orders': len(orders),
        'tables': tables
    }
    
    metadata_file = output_dir / f"{query_name}_orders_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {len(orders)} join orders")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    main()