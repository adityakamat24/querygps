#!/usr/bin/env python3
"""
Encode join orders as tree structures
"""
import re
import torch
from typing import List, Dict, Tuple
from pathlib import Path

class JoinTreeEncoder:
    """
    Encodes join orders by analyzing the execution plan structure
    """
    
    def __init__(self, table_stats: Dict):
        self.table_stats = table_stats
    
    def encode_from_sql(self, sql_file: str) -> torch.Tensor:
        """
        Extract join order from SQL file and encode as feature vector
        
        The encoding captures:
        - Which tables are joined
        - Order of joins (left-deep, right-deep, bushy)
        - Cardinality estimates at each join
        """
        with open(sql_file, 'r') as f:
            sql = f.read()
        
        # Extract tables from FROM clause
        tables = self._extract_tables(sql)
        
        # Extract join predicates from WHERE clause
        joins = self._extract_joins(sql)
        
        # Build join graph
        join_graph = self._build_join_graph(tables, joins)
        
        # Encode the join order
        encoding = self._encode_join_order(tables, join_graph)
        
        return encoding
    
    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL"""
        from_match = re.search(r'FROM\s+(.*?)\s+WHERE', sql, re.IGNORECASE | re.DOTALL)
        if not from_match:
            from_match = re.search(r'FROM\s+(.*?)(?:;|\s*$)', sql, re.IGNORECASE | re.DOTALL)
        
        if not from_match:
            return []
        
        from_clause = from_match.group(1).strip()
        table_list = from_clause.split(',')
        tables = []
        
        for table in table_list:
            table = table.strip()
            parts = table.split()
            if parts:
                table_name = parts[0].lower()
                if table_name and table_name.upper() not in ['SELECT', 'WHERE', 'JOIN', 'ON', 'AND']:
                    tables.append(table_name)
        
        return tables
    
    def _extract_joins(self, sql: str) -> List[Tuple[str, str]]:
        """Extract join predicates"""
        joins = []
        where_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        matches = re.finditer(where_pattern, sql, re.IGNORECASE)
        
        for match in matches:
            table1 = match.group(1).lower()
            table2 = match.group(3).lower()
            if table1 != table2:
                joins.append((table1, table2))
        
        return list(set(joins))
    
    def _build_join_graph(self, tables: List[str], joins: List[Tuple[str, str]]) -> Dict:
        """Build adjacency list for join graph"""
        graph = {table: [] for table in tables}
        
        for t1, t2 in joins:
            if t1 in graph and t2 in graph:
                graph[t1].append(t2)
                graph[t2].append(t1)
        
        return graph
    
    def _encode_join_order(self, tables: List[str], join_graph: Dict) -> torch.Tensor:
        """
        Encode the join order as a feature vector
        
        Features (128-dim):
        - [0:8] One-hot for number of tables (max 8)
        - [8:16] Table cardinality features (one per table, normalized)
        - [16:24] Join selectivity estimates
        - [24:32] Position of each table in join order
        - [32:64] Pairwise join indicators (which tables join together)
        - [64:128] Reserved for additional features
        """
        encoding = torch.zeros(128)
        
        num_tables = len(tables)
        
        # Feature 1: One-hot encode number of tables (capped at 8)
        if num_tables <= 8:
            encoding[num_tables - 1] = 1.0
        else:
            encoding[7] = 1.0
        
        # Feature 2: Table sizes (log-normalized)
        import math
        for i, table in enumerate(tables[:8]):
            if table in self.table_stats:
                size = self.table_stats[table].get('row_count', 1)
                encoding[8 + i] = math.log10(max(size, 1)) / 10.0  # Normalize to ~[0, 1]
        
        # Feature 3: Join connectivity (number of joins per table)
        for i, table in enumerate(tables[:8]):
            if table in join_graph:
                num_joins = len(join_graph[table])
                encoding[16 + i] = min(num_joins / 5.0, 1.0)  # Normalize
        
        # Feature 4: Table position encoding (normalized position in FROM clause)
        for i, table in enumerate(tables[:8]):
            encoding[24 + i] = i / max(num_tables - 1, 1)
        
        # Feature 5: Pairwise join existence (compressed)
        # Use a hash to represent which pairs join
        join_hash = 0
        for t1, t2 in [(t1, t2) for t1 in tables for t2 in join_graph.get(t1, [])]:
            if tables.index(t1) < tables.index(t2):  # Avoid duplicates
                idx1 = tables.index(t1)
                idx2 = tables.index(t2)
                join_hash ^= (1 << (idx1 * 8 + idx2))
        
        # Convert hash to binary features
        for i in range(32):
            encoding[32 + i] = float((join_hash >> i) & 1)
        
        # Feature 6: Derived statistics
        # Average table size
        if tables and self.table_stats:
            sizes = [self.table_stats.get(t, {}).get('row_count', 0) for t in tables]
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            encoding[64] = math.log10(max(avg_size, 1)) / 10.0
        
        # Join density (how connected is the join graph)
        max_possible_joins = num_tables * (num_tables - 1) // 2
        actual_joins = sum(len(neighbors) for neighbors in join_graph.values()) // 2
        encoding[65] = actual_joins / max(max_possible_joins, 1)
        
        # Size variance
        if tables and self.table_stats:
            sizes = [self.table_stats.get(t, {}).get('row_count', 1) for t in tables]
            if len(sizes) > 1:
                mean_size = sum(sizes) / len(sizes)
                variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
                encoding[66] = min(math.log10(max(variance, 1)) / 20.0, 1.0)
        
        return encoding
    
    def encode_from_filename(self, filename: str, table_stats: Dict) -> torch.Tensor:
        """
        Quick encoding from filename when SQL file isn't available
        Extracts table names from filename pattern like:
        'q001_q_customer_orders_lineitem_v0_order_042.sql'
        """
        # Extract base query name
        base_query = Path(filename).stem.rsplit('_order_', 1)[0]
        
        # Try to extract table names from query name
        # Pattern: q###_q_table1_table2_table3_v#
        parts = base_query.split('_')
        tables = []
        
        # Skip 'q###' and 'q', then collect table names until 'v#'
        in_tables = False
        for part in parts:
            if part.startswith('q') and part[1:].isdigit():
                continue
            if part.startswith('v') and len(part) > 1 and part[1:].isdigit():
                break
            if part == 'q':
                in_tables = True
                continue
            if in_tables:
                tables.append(part.lower())
        
        # If we found tables, create a simple encoding
        if tables:
            # Create minimal join graph (assume sequential joins)
            join_graph = {tables[i]: [tables[i+1]] if i < len(tables)-1 else [] 
                         for i in range(len(tables))}
            if len(tables) > 1:
                join_graph[tables[-1]].append(tables[-2])
            
            return self._encode_join_order(tables, join_graph)
        else:
            # Fallback: return zero encoding
            return torch.zeros(128)