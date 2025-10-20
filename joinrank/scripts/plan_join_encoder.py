#!/usr/bin/env python3
"""
Extract join order from PostgreSQL execution plans
"""
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PlanJoinOrderEncoder:
    """
    Extract and encode the actual join order from PostgreSQL execution plans
    This captures the REAL join tree structure that was executed
    """
    
    def __init__(self, table_stats: Dict, runtimes_file: Optional[str] = None):
        self.table_stats = table_stats

        # Load all execution plans if file is available
        if runtimes_file is not None:
            try:
                with open(runtimes_file) as f:
                    self.plans_data = json.load(f)

                # Create mapping from SQL file to plan
                self.plan_map = {}
                for item in self.plans_data:
                    sql_file = item['sql_file']
                    self.plan_map[sql_file] = item['plan']

                print(f"Loaded {len(self.plan_map)} execution plans")
            except Exception as e:
                print(f"Warning: Could not load runtime file {runtimes_file}: {e}")
                self.plans_data = []
                self.plan_map = {}
        else:
            print("No runtime file provided, using SQL-based join order encoding")
            self.plans_data = []
            self.plan_map = {}

        # Initialize SQL-based encoder as fallback
        from sql_join_encoder import SQLJoinOrderEncoder
        self.sql_encoder = SQLJoinOrderEncoder(table_stats)
    
    def encode_from_sql_file(self, sql_file: str) -> torch.Tensor:
        """
        Extract join order from execution plan and encode it
        
        Returns 128-dim encoding capturing:
        - Join tree structure (depth, breadth)
        - Join types (nested loop, hash, merge)
        - Table access order
        - Cardinality estimates
        """
        # Normalize path
        sql_file = str(Path(sql_file))
        
        if sql_file not in self.plan_map:
            # Try with forward slashes
            sql_file_alt = sql_file.replace('\\', '/')
            if sql_file_alt in self.plan_map:
                sql_file = sql_file_alt
            else:
                # Use SQL-based encoding as fallback
                return self.sql_encoder.encode_from_sql_file(sql_file)
        
        plan = self.plan_map[sql_file]
        
        # Extract join tree from plan
        join_tree = self._extract_join_tree(plan['Plan'])
        
        # Encode the join tree
        encoding = self._encode_join_tree(join_tree)
        
        return encoding
    
    def _extract_join_tree(self, node: Dict) -> Dict:
        """
        Recursively extract join tree structure from plan node
        
        Returns tree with:
        - node_type: 'join', 'scan', or 'other'
        - join_type: 'Nested Loop', 'Hash Join', 'Merge Join', etc.
        - table: table name if this is a scan
        - left: left subtree
        - right: right subtree
        - rows: estimated rows
        - cost: estimated cost
        """
        node_type = node.get('Node Type', '')
        
        tree = {
            'node_type': node_type,
            'rows': node.get('Plan Rows', 0),
            'cost': node.get('Total Cost', 0),
            'left': None,
            'right': None
        }
        
        # Check if this is a join node
        if 'Join' in node_type:
            tree['node_type'] = 'join'
            tree['join_type'] = node_type
            
            # Get children
            plans = node.get('Plans', [])
            if len(plans) >= 2:
                tree['left'] = self._extract_join_tree(plans[0])
                tree['right'] = self._extract_join_tree(plans[1])
            elif len(plans) == 1:
                tree['left'] = self._extract_join_tree(plans[0])
        
        # Check if this is a scan node
        elif 'Scan' in node_type:
            tree['node_type'] = 'scan'
            tree['table'] = node.get('Relation Name', '').lower()
            tree['alias'] = node.get('Alias', '').lower()
        
        # Other node types (Aggregate, Sort, etc.)
        else:
            tree['node_type'] = 'other'
            plans = node.get('Plans', [])
            if plans:
                tree['left'] = self._extract_join_tree(plans[0])
                if len(plans) > 1:
                    tree['right'] = self._extract_join_tree(plans[1])
        
        return tree
    
    def _encode_join_tree(self, tree: Dict) -> torch.Tensor:
        """
        Encode join tree as 128-dim feature vector
        
        Features:
        [0-7]: Tree structure (depth, num_joins, num_scans, branching)
        [8-15]: Join type distribution (nested loop, hash, merge)
        [16-31]: Table access order (which tables joined first)
        [32-47]: Cardinality features (row estimates at each level)
        [48-63]: Cost features (cost estimates at each level)
        [64-127]: Reserved for additional features
        """
        encoding = torch.zeros(128)
        
        # Extract tree statistics
        stats = self._compute_tree_stats(tree)
        
        # Feature 1: Tree structure [0-7]
        encoding[0] = min(stats['depth'] / 10.0, 1.0)
        encoding[1] = min(stats['num_joins'] / 10.0, 1.0)
        encoding[2] = min(stats['num_scans'] / 8.0, 1.0)
        encoding[3] = stats['avg_branching']
        encoding[4] = 1.0 if stats['is_left_deep'] else 0.0
        encoding[5] = 1.0 if stats['is_right_deep'] else 0.0
        encoding[6] = 1.0 if stats['is_bushy'] else 0.0
        encoding[7] = stats['balance_factor']
        
        # Feature 2: Join type distribution [8-15]
        total_joins = max(stats['num_joins'], 1)
        encoding[8] = stats['nested_loop_joins'] / total_joins
        encoding[9] = stats['hash_joins'] / total_joins
        encoding[10] = stats['merge_joins'] / total_joins
        
        # Feature 3: Table access order [16-31]
        # Encode which tables appear in what order
        for i, table in enumerate(stats['table_order'][:8]):
            if table in self.table_stats:
                import math
                size = self.table_stats[table].get('row_count', 1)
                encoding[16 + i] = math.log10(max(size, 1)) / 10.0
                encoding[24 + i] = i / max(len(stats['table_order']) - 1, 1)
        
        # Feature 4: Cardinality at different tree levels [32-47]
        for i, rows in enumerate(stats['rows_per_level'][:8]):
            import math
            encoding[32 + i] = math.log10(max(rows, 1)) / 10.0
            encoding[40 + i] = min(rows / 1e6, 1.0)
        
        # Feature 5: Cost at different tree levels [48-63]
        for i, cost in enumerate(stats['cost_per_level'][:8]):
            import math
            encoding[48 + i] = math.log10(max(cost, 1)) / 10.0
            encoding[56 + i] = min(cost / 1e6, 1.0)
        
        # Feature 6: Join selectivity estimates [64-79]
        for i, sel in enumerate(stats['selectivities'][:8]):
            encoding[64 + i] = sel
        
        return encoding
    
    def _compute_tree_stats(self, tree: Dict) -> Dict:
        """Compute statistics about the join tree"""
        stats = {
            'depth': 0,
            'num_joins': 0,
            'num_scans': 0,
            'nested_loop_joins': 0,
            'hash_joins': 0,
            'merge_joins': 0,
            'table_order': [],
            'rows_per_level': [],
            'cost_per_level': [],
            'selectivities': [],
            'is_left_deep': True,
            'is_right_deep': True,
            'is_bushy': False,
            'avg_branching': 0.0,
            'balance_factor': 0.5
        }
        
        # Traverse tree to collect statistics
        self._traverse_tree(tree, stats, level=0)
        
        # Compute derived statistics
        if stats['num_joins'] > 0:
            stats['avg_branching'] = stats['num_scans'] / stats['num_joins']
        
        return stats
    
    def _traverse_tree(self, node: Dict, stats: Dict, level: int):
        """Recursively traverse tree and collect statistics"""
        if node is None:
            return
        
        stats['depth'] = max(stats['depth'], level)
        
        # Count node types
        if node['node_type'] == 'join':
            stats['num_joins'] += 1
            
            # Count join types
            join_type = node.get('join_type', '')
            if 'Nested Loop' in join_type:
                stats['nested_loop_joins'] += 1
            elif 'Hash' in join_type:
                stats['hash_joins'] += 1
            elif 'Merge' in join_type:
                stats['merge_joins'] += 1
            
            # Check tree shape
            if node['right'] and node['right']['node_type'] == 'join':
                stats['is_left_deep'] = False
            if node['left'] and node['left']['node_type'] == 'join':
                stats['is_right_deep'] = False
            if (node['left'] and node['left']['node_type'] == 'join' and
                node['right'] and node['right']['node_type'] == 'join'):
                stats['is_bushy'] = True
        
        elif node['node_type'] == 'scan':
            stats['num_scans'] += 1
            table = node.get('table', '')
            if table:
                stats['table_order'].append(table)
        
        # Collect cardinality and cost
        if level < 16:  # Limit depth
            while len(stats['rows_per_level']) <= level:
                stats['rows_per_level'].append(0)
            while len(stats['cost_per_level']) <= level:
                stats['cost_per_level'].append(0)
            
            stats['rows_per_level'][level] = max(stats['rows_per_level'][level], node.get('rows', 0))
            stats['cost_per_level'][level] = max(stats['cost_per_level'][level], node.get('cost', 0))
        
        # Recurse
        if node['left']:
            self._traverse_tree(node['left'], stats, level + 1)
        if node['right']:
            self._traverse_tree(node['right'], stats, level + 1)