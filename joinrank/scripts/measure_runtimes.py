#!/usr/bin/env python3
"""
Measure runtime for join orders using EXPLAIN ANALYZE
"""
import psycopg2
import time
import json
import statistics
from pathlib import Path
from tqdm import tqdm
import argparse

class RuntimeMeasurer:
    def __init__(self, db_name: str, db_user: str, db_host: str = 'localhost'):
        self.conn_params = {
            'dbname': db_name,
            'user': db_user,
            'host': db_host
        }
        self.conn = None
        
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.conn_params)
        
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def set_params(self):
        """Set PostgreSQL parameters for consistent measurements"""
        cursor = self.conn.cursor()
        params = [
            "SET max_parallel_workers_per_gather = 0;",  # Disable parallelism
            "SET random_page_cost = 1.1;",
            "SET work_mem = '256MB';",
            # Removed shared_buffers - requires restart
        ]
        for param in params:
            cursor.execute(param)
        self.conn.commit()
        cursor.close()
    
    def measure_query(self, sql: str, n_runs: int = 3) -> dict:
        """
        Execute query with EXPLAIN ANALYZE multiple times and collect stats
        """
        cursor = self.conn.cursor()
        runtimes = []
        plan_json = None
        
        for run in range(n_runs):
            try:
                # Run EXPLAIN ANALYZE
                explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
                cursor.execute(explain_sql)
                result = cursor.fetchone()[0]
                
                # Extract runtime
                if isinstance(result, list) and len(result) > 0:
                    plan = result[0]
                    execution_time = plan.get('Execution Time', 0)
                    planning_time = plan.get('Planning Time', 0)
                    total_time = execution_time + planning_time
                    runtimes.append(total_time)
                    
                    if run == 0:  # Save plan from first run
                        plan_json = plan
                
                # Commit to clear transaction
                self.conn.commit()
                
                # Small delay between runs
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error executing query: {e}")
                self.conn.rollback()
                return None
        
        cursor.close()
        
        if not runtimes:
            return None
        
        return {
            'runtime_ms': statistics.median(runtimes),
            'runtime_mean_ms': statistics.mean(runtimes),
            'runtime_std_ms': statistics.stdev(runtimes) if len(runtimes) > 1 else 0,
            'all_runtimes_ms': runtimes,
            'n_runs': n_runs,
            'plan': plan_json
        }
    
    def measure_order_file(self, sql_file: Path, n_runs: int = 3) -> dict:
        """Measure a single join order SQL file"""
        with open(sql_file) as f:
            sql = f.read()
        
        # Extract just the query (skip SET commands)
        lines = [l for l in sql.split('\n') if not l.strip().startswith('SET') and l.strip()]
        query_sql = '\n'.join(lines)
        
        result = self.measure_query(query_sql, n_runs)
        if result:
            result['sql_file'] = str(sql_file)
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-name', required=True, help='Database name')
    parser.add_argument('--db-user', default='postgres', help='Database user')
    parser.add_argument('--orders-dir', required=True, help='Directory with join order SQL files')
    parser.add_argument('--query-pattern', default='*.sql', help='Pattern to match SQL files')
    parser.add_argument('--n-runs', type=int, default=3, help='Number of runs per query')
    parser.add_argument('--output', required=True, help='Output JSON file')
    args = parser.parse_args()
    
    # Find all order files
    orders_dir = Path(args.orders_dir)
    sql_files = sorted(orders_dir.glob(args.query_pattern))
    print(f"Found {len(sql_files)} SQL files in {orders_dir}")
    
    # Connect to database
    measurer = RuntimeMeasurer(args.db_name, args.db_user)
    measurer.connect()
    measurer.set_params()
    print("Connected to database and set parameters")
    
    # Measure each order
    results = []
    for sql_file in tqdm(sql_files, desc="Measuring runtimes"):
        result = measurer.measure_order_file(sql_file, args.n_runs)
        if result:
            # Add file metadata
            result['query_name'] = sql_file.stem
            results.append(result)
    
    measurer.disconnect()
    
    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nMeasured {len(results)} join orders")
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    if results:
        runtimes = [r['runtime_ms'] for r in results]
        print(f"\nRuntime summary:")
        print(f"  Min: {min(runtimes):.2f} ms")
        print(f"  Median: {statistics.median(runtimes):.2f} ms")
        print(f"  Max: {max(runtimes):.2f} ms")
        print(f"  Std: {statistics.stdev(runtimes):.2f} ms")

if __name__ == "__main__":
    main()