#!/usr/bin/env python3
"""
Parse SQL queries and build graph representations
"""
import re
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Dict, List, Tuple

class QueryGraphBuilder:
    """Build graph representation from SQL query"""
    
    def __init__(self, table_stats: Dict):
        self.table_stats = table_stats
        
    def parse_sql_file(self, sql_file: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Parse SQL file to extract tables and joins
        Returns: (tables, joins)
        """
        with open(sql_file, 'r') as f:
            sql = f.read()
        
        # Extract table names using regex
        tables = self._extract_tables_regex(sql)
        
        # Extract join conditions
        joins = self._extract_joins(sql, tables)
        
        return tables, joins
    
    def _extract_tables_regex(self, sql: str) -> List[str]:
        """Extract table names using regex - handles comma-separated FROM clause"""
        # Find FROM clause
        from_match = re.search(r'FROM\s+(.*?)\s+WHERE', sql, re.IGNORECASE | re.DOTALL)
        if not from_match:
            # Try without WHERE
            from_match = re.search(r'FROM\s+(.*?)(?:;|\s*$)', sql, re.IGNORECASE | re.DOTALL)
        
        if not from_match:
            return []
        
        from_clause = from_match.group(1).strip()
        
        # Split by comma and clean up
        table_list = from_clause.split(',')
        tables = []
        for table in table_list:
            table = table.strip()
            # Remove alias if present (e.g., "customer c" -> "customer")
            parts = table.split()
            if parts:
                table_name = parts[0].lower()
                if table_name and table_name.upper() not in ['SELECT', 'WHERE', 'JOIN', 'ON', 'AND']:
                    tables.append(table_name)
        
        return tables
    
    def _extract_joins(self, sql: str, tables: List[str]) -> List[Tuple[str, str]]:
        """Extract join conditions from WHERE clause"""
        joins = []
        
        # Look for equality conditions in WHERE clause like: table1.col = table2.col
        where_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        matches = re.finditer(where_pattern, sql, re.IGNORECASE)
        
        for match in matches:
            table1 = match.group(1).lower()
            table2 = match.group(3).lower()
            
            # Only include if both tables are in our table list
            if table1 in tables and table2 in tables and table1 != table2:
                joins.append((table1, table2))
        
        # Remove duplicates
        joins = list(set(joins))
        
        return joins
    
    def build_graph(self, sql_file: str) -> Data:
        """
        Build PyTorch Geometric graph from SQL file
        
        Returns:
            Data object with:
                x: Node features [num_nodes, node_feature_dim]
                edge_index: Edge connectivity [2, num_edges]
                edge_attr: Edge features [num_edges, edge_feature_dim]
        """
        try:
            tables, joins = self.parse_sql_file(sql_file)
        except Exception as e:
            print(f"Error parsing {sql_file}: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal valid graph
            x = torch.zeros((1, 64), dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, 64), dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        if not tables:
            print(f"Warning: No tables found in {sql_file}, creating dummy graph")
            x = torch.zeros((1, 64), dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, 64), dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Create node features
        node_features = []
        table_to_idx = {}
        
        for idx, table in enumerate(tables):
            table_to_idx[table] = idx
            features = self._get_table_features(table)
            node_features.append(features)
        
        # Convert to tensor - ensure 2D shape
        x = torch.tensor(node_features, dtype=torch.float)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Create edge index and edge features
        edge_index = []
        edge_attr = []
        
        for table1, table2 in joins:
            if table1 in table_to_idx and table2 in table_to_idx:
                idx1 = table_to_idx[table1]
                idx2 = table_to_idx[table2]
                
                # Add bidirectional edges
                edge_index.append([idx1, idx2])
                edge_index.append([idx2, idx1])
                
                # Edge features
                edge_feat = self._get_edge_features(table1, table2)
                edge_attr.append(edge_feat)
                edge_attr.append(edge_feat)
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            # No joins - create self-loops
            num_nodes = len(tables)
            edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
            edge_attr = torch.zeros((num_nodes, 64), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _get_table_features(self, table_name: str) -> List[float]:
        """Extract features for a table node"""
        if table_name not in self.table_stats:
            return [0.0] * 64
        
        stats = self.table_stats[table_name]
        row_count = stats.get('row_count', 0)
        size_mb = stats.get('size_mb', 0)
        
        features = [0.0] * 64
        
        import math
        features[0] = math.log10(max(row_count, 1))
        features[1] = math.log10(max(size_mb, 0.1))
        features[2] = row_count / 1e6
        features[3] = size_mb / 1000.0
        
        # One-hot encoding for table types
        table_types = ['customer', 'orders', 'lineitem', 'part', 'supplier', 'partsupp', 'nation', 'region']
        for i, ttype in enumerate(table_types):
            if ttype in table_name.lower():
                features[4 + i] = 1.0
        
        return features
    
    def _get_edge_features(self, table1: str, table2: str) -> List[float]:
        """Extract features for a join edge"""
        features = [0.0] * 64
        
        if table1 in self.table_stats and table2 in self.table_stats:
            size1 = self.table_stats[table1].get('row_count', 1)
            size2 = self.table_stats[table2].get('row_count', 1)
            
            import math
            features[0] = math.log10(max(size1 * size2, 1))
            features[1] = min(size1, size2) / max(size1, size2)
            features[2] = math.log10(max(size1, 1))
            features[3] = math.log10(max(size2, 1))
        
        return features