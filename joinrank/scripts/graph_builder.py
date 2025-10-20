#!/usr/bin/env python3
"""
Build query graphs and extract schema-invariant features
"""
import re
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

class QueryGraphBuilder:
    """Convert SQL query to a graph representation"""
    
    def __init__(self, table_stats: Dict = None):
        """
        Args:
            table_stats: Dictionary with table cardinalities and stats
                        {table_name: {'cardinality': int, 'size_mb': float}}
        """
        self.table_stats = table_stats or {}
    
    def parse_query(self, sql: str) -> Dict:
        """Parse SQL query to extract tables, joins, and predicates"""
        # Extract tables
        from_match = re.search(r'FROM\s+(.*?)\s+WHERE', sql, re.IGNORECASE | re.DOTALL)
        if not from_match:
            return None
        
        tables_str = from_match.group(1)
        tables = [t.strip() for t in tables_str.split(',')]
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*)', sql, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return None
        
        where_clause = where_match.group(1).replace(';', '')
        predicates = [p.strip() for p in where_clause.split('AND')]
        
        # Classify predicates
        join_preds = []
        local_preds = {t: [] for t in tables}
        
        for pred in predicates:
            # Check if it's a join (table1.col = table2.col)
            join_match = re.match(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', pred)
            if join_match:
                t1, c1, t2, c2 = join_match.groups()
                if t1 != t2 and t1 in tables and t2 in tables:
                    join_preds.append({
                        'left_table': t1,
                        'right_table': t2,
                        'left_col': c1,
                        'right_col': c2,
                        'predicate': pred
                    })
                    continue
            
            # Otherwise, it's a local predicate
            for table in tables:
                if table in pred:
                    local_preds[table].append(pred)
                    break
        
        return {
            'tables': tables,
            'join_predicates': join_preds,
            'local_predicates': local_preds
        }
    
    def build_graph(self, sql: str) -> nx.Graph:
        """Build NetworkX graph from query"""
        query_info = self.parse_query(sql)
        if not query_info:
            return None
        
        G = nx.Graph()
        
        # Add table nodes with features
        for table in query_info['tables']:
            G.add_node(table, **self._get_node_features(table, query_info))
        
        # Add join edges with features
        for join in query_info['join_predicates']:
            t1, t2 = join['left_table'], join['right_table']
            G.add_edge(t1, t2, **self._get_edge_features(join))
        
        return G
    
    def _get_node_features(self, table: str, query_info: Dict) -> Dict:
        """Extract schema-invariant node features"""
        features = {}
        
        # Table cardinality (log scale)
        if table in self.table_stats:
            cardinality = self.table_stats[table].get('cardinality', 1)
            features['log_cardinality'] = np.log10(max(cardinality, 1))
        else:
            features['log_cardinality'] = 0.0
        
        # Number of local predicates
        n_local_preds = len(query_info['local_predicates'].get(table, []))
        features['n_local_predicates'] = float(n_local_preds)
        
        # Estimated selectivity (rough approximation)
        # More predicates = lower selectivity
        features['estimated_selectivity'] = 1.0 / (1.0 + n_local_preds)
        
        # Degree (number of joins involving this table)
        degree = sum(1 for j in query_info['join_predicates'] 
                    if table in [j['left_table'], j['right_table']])
        features['degree'] = float(degree)
        
        return features
    
    def _get_edge_features(self, join: Dict) -> Dict:
        """Extract schema-invariant edge features"""
        features = {}
        
        # Join type (equi-join vs others)
        features['is_equi_join'] = 1.0
        
        # Check if it's likely a PK-FK join (heuristic)
        # Look for patterns like id, key in column names
        left_col = join['left_col'].lower()
        right_col = join['right_col'].lower()
        is_key_join = any(kw in left_col or kw in right_col 
                         for kw in ['key', 'id'])
        features['is_pk_fk'] = float(is_key_join)
        
        # Estimated join selectivity (placeholder - could use histograms)
        features['join_selectivity'] = 0.1
        
        return features
    
    def to_pyg_data(self, G: nx.Graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        if G is None or len(G.nodes()) == 0:
            return None
        
        # Convert to PyG format
        data = from_networkx(G)
        
        # Aggregate node features into tensor
        node_features = []
        for node in G.nodes():
            feats = G.nodes[node]
            feat_vector = [
                feats.get('log_cardinality', 0.0),
                feats.get('n_local_predicates', 0.0),
                feats.get('estimated_selectivity', 1.0),
                feats.get('degree', 0.0)
            ]
            node_features.append(feat_vector)
        
        data.x = torch.tensor(node_features, dtype=torch.float)
        
        # Aggregate edge features into tensor
        edge_features = []
        for u, v in G.edges():
            feats = G.edges[u, v]
            feat_vector = [
                feats.get('is_equi_join', 1.0),
                feats.get('is_pk_fk', 0.0),
                feats.get('join_selectivity', 0.1)
            ]
            edge_features.append(feat_vector)
            edge_features.append(feat_vector)  # Undirected, so duplicate
        
        if edge_features:
            data.edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return data

class JoinOrderEncoder:
    """Encode join orders as sequences or trees"""
    
    def __init__(self, max_tables: int = 10):
        self.max_tables = max_tables
    
    def encode_sequence(self, join_order: List[str]) -> torch.Tensor:
        """
        Encode join order as a sequence.
        Returns: tensor of shape (max_tables,) with table indices
        """
        # Simple sequential encoding: assign each table an index
        table_to_idx = {t: i for i, t in enumerate(join_order)}
        
        # Pad to max length
        sequence = list(range(len(join_order)))
        sequence += [-1] * (self.max_tables - len(sequence))
        
        return torch.tensor(sequence[:self.max_tables], dtype=torch.long)
    
    def encode_tree(self, tree_structure: str) -> torch.Tensor:
        """
        Encode join tree structure.
        For now, use a simple linearization.
        """
        # Placeholder: could use tree-LSTM or graph encoding
        # For simplicity, just return a fixed-size encoding
        return torch.randn(32)  # Simple embedding

# Example usage
if __name__ == "__main__":
    # Sample query
    sql = """
    SELECT COUNT(*)
    FROM customer, orders, lineitem
    WHERE customer.c_custkey = orders.o_custkey
      AND orders.o_orderkey = lineitem.l_orderkey
      AND customer.c_mktsegment = 'BUILDING'
      AND lineitem.l_shipdate < DATE '1995-01-01'
    """
    
    # Sample table stats
    table_stats = {
        'customer': {'cardinality': 150000},
        'orders': {'cardinality': 1500000},
        'lineitem': {'cardinality': 6000000}
    }
    
    # Build graph
    builder = QueryGraphBuilder(table_stats)
    G = builder.build_graph(sql)
    
    print(f"Graph nodes: {list(G.nodes())}")
    print(f"Graph edges: {list(G.edges())}")
    print(f"\nNode features for 'customer':")
    print(G.nodes['customer'])
    
    # Convert to PyG
    data = builder.to_pyg_data(G)
    print(f"\nPyG Data:")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge index shape: {data.edge_index.shape}")
    if hasattr(data, 'edge_attr'):
        print(f"  Edge features shape: {data.edge_attr.shape}")