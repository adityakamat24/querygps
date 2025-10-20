#!/usr/bin/env python3
"""
SQL-based join order encoder that extracts join order features directly from SQL files
"""
import torch
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SQLJoinOrderEncoder:
    """
    Extract join order features directly from SQL files without requiring execution plans
    """

    def __init__(self, table_stats: Dict):
        self.table_stats = table_stats

        # Create table name to index mapping
        self.table_to_idx = {}
        self.idx_to_table = {}
        for i, table_name in enumerate(sorted(table_stats.keys())):
            # Handle aliased table names
            base_name = table_name.split(' AS ')[0] if ' AS ' in table_name else table_name
            self.table_to_idx[base_name] = i
            self.idx_to_table[i] = base_name

        self.num_tables = len(self.table_to_idx)
        print(f"Initialized SQL encoder with {self.num_tables} tables")

    def encode_from_sql_file(self, sql_file: str) -> torch.Tensor:
        """
        Extract join order encoding from SQL file

        Returns 128-dim encoding capturing:
        - Table ordering features (32 dims)
        - Join pattern features (32 dims)
        - Selectivity features (32 dims)
        - Structural features (32 dims)
        """
        try:
            # Read SQL file
            with open(sql_file, 'r') as f:
                sql_content = f.read()

            # Extract features
            table_order_features = self._extract_table_order_features(sql_content)
            join_pattern_features = self._extract_join_pattern_features(sql_content)
            selectivity_features = self._extract_selectivity_features(sql_content)
            structural_features = self._extract_structural_features(sql_content)

            # Combine into 128-dim vector
            encoding = torch.cat([
                table_order_features,      # [0:32]
                join_pattern_features,     # [32:64]
                selectivity_features,      # [64:96]
                structural_features        # [96:128]
            ])

            return encoding

        except Exception as e:
            print(f"Warning: Failed to encode SQL file {sql_file}: {e}")
            return torch.zeros(128)

    def _extract_table_order_features(self, sql_content: str) -> torch.Tensor:
        """Extract features based on table ordering in FROM clause"""
        features = torch.zeros(32)

        # Extract FROM clause
        from_match = re.search(r'FROM\s+(.*?)\s+WHERE', sql_content, re.IGNORECASE | re.DOTALL)
        if not from_match:
            return features

        from_clause = from_match.group(1)

        # Parse table order
        tables = []
        # Split by comma and extract table names
        table_parts = [t.strip() for t in from_clause.split(',')]

        for part in table_parts:
            # Handle "table AS alias" format
            if ' AS ' in part.upper():
                table_name = part.split(' AS ')[0].strip()
                alias = part.split(' AS ')[1].strip()
            else:
                table_name = part.strip()
                alias = table_name

            tables.append((table_name, alias))

        # Encode table positions (first 16 features)
        for i, (table_name, alias) in enumerate(tables[:8]):  # Max 8 tables
            if table_name in self.table_to_idx:
                table_idx = self.table_to_idx[table_name]
                # Position encoding
                features[i] = (table_idx + 1) / self.num_tables  # Normalized table index
                features[i + 8] = (i + 1) / len(tables)  # Normalized position

        # Table size ordering features (features 16-31)
        if len(tables) > 1:
            table_sizes = []
            for table_name, alias in tables:
                if table_name in self.table_stats:
                    size = self.table_stats[table_name].get('row_count', 1)
                    table_sizes.append(size)
                else:
                    table_sizes.append(1)

            # Check if tables are ordered by size
            sorted_sizes = sorted(table_sizes)
            reverse_sorted_sizes = sorted(table_sizes, reverse=True)

            if table_sizes == sorted_sizes:
                features[16] = 1.0  # Ascending size order
            elif table_sizes == reverse_sorted_sizes:
                features[17] = 1.0  # Descending size order

            # Size variance features
            if table_sizes:
                import math
                max_size = max(table_sizes)
                min_size = min(table_sizes)
                avg_size = sum(table_sizes) / len(table_sizes)

                features[18] = math.log10(max_size / max(min_size, 1)) / 10.0  # Size ratio
                features[19] = len([s for s in table_sizes if s < avg_size]) / len(table_sizes)  # Small table ratio

        return features

    def _extract_join_pattern_features(self, sql_content: str) -> torch.Tensor:
        """Extract features based on join patterns in WHERE clause"""
        features = torch.zeros(32)

        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|;|$)',
                               sql_content, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return features

        where_clause = where_match.group(1)

        # Find join conditions (equality conditions between tables)
        join_conditions = re.findall(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', where_clause)

        # Count different types of joins
        star_joins = 0  # One table joins to many
        chain_joins = 0  # Linear chain pattern
        cycle_joins = 0  # Circular patterns

        # Build join graph
        join_graph = {}
        for t1_alias, col1, t2_alias, col2 in join_conditions:
            if t1_alias not in join_graph:
                join_graph[t1_alias] = set()
            if t2_alias not in join_graph:
                join_graph[t2_alias] = set()
            join_graph[t1_alias].add(t2_alias)
            join_graph[t2_alias].add(t1_alias)

        # Analyze join patterns
        if join_graph:
            # Count join degrees
            degrees = [len(neighbors) for neighbors in join_graph.values()]
            max_degree = max(degrees) if degrees else 0
            avg_degree = sum(degrees) / len(degrees) if degrees else 0

            features[0] = min(max_degree / 10.0, 1.0)  # Max join degree
            features[1] = min(avg_degree / 5.0, 1.0)   # Average join degree
            features[2] = len(join_conditions) / 20.0  # Number of join conditions

            # Star join detection (one table with high degree)
            if max_degree >= 3:
                features[3] = 1.0
                star_joins = 1

            # Chain detection (most tables have degree 2)
            degree_2_count = len([d for d in degrees if d == 2])
            if degree_2_count >= len(degrees) * 0.6:
                features[4] = 1.0
                chain_joins = 1

        # Count selection predicates
        selection_patterns = [
            r"LIKE\s+['\"].*?['\"]",
            r"NOT LIKE\s+['\"].*?['\"]",
            r"=\s+['\"].*?['\"]",
            r"<\s+\d+",
            r">\s+\d+",
            r"BETWEEN\s+\d+\s+AND\s+\d+"
        ]

        for i, pattern in enumerate(selection_patterns):
            matches = len(re.findall(pattern, where_clause, re.IGNORECASE))
            features[5 + i] = min(matches / 5.0, 1.0)

        return features

    def _extract_selectivity_features(self, sql_content: str) -> torch.Tensor:
        """Extract selectivity-related features"""
        features = torch.zeros(32)

        # Count different types of predicates that affect selectivity
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|;|$)',
                               sql_content, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return features

        where_clause = where_match.group(1)

        # String comparisons (typically selective)
        string_equals = len(re.findall(r"=\s*['\"].*?['\"]", where_clause))
        string_likes = len(re.findall(r"LIKE\s*['\"].*?['\"]", where_clause))
        string_not_likes = len(re.findall(r"NOT LIKE\s*['\"].*?['\"]", where_clause))

        features[0] = min(string_equals / 10.0, 1.0)
        features[1] = min(string_likes / 5.0, 1.0)
        features[2] = min(string_not_likes / 5.0, 1.0)

        # Range predicates
        range_predicates = len(re.findall(r"(BETWEEN|<|>|<=|>=)\s*\d+", where_clause))
        features[3] = min(range_predicates / 10.0, 1.0)

        # OR conditions (typically less selective)
        or_conditions = len(re.findall(r"\bOR\b", where_clause, re.IGNORECASE))
        features[4] = min(or_conditions / 5.0, 1.0)

        # AND conditions
        and_conditions = len(re.findall(r"\bAND\b", where_clause, re.IGNORECASE))
        features[5] = min(and_conditions / 20.0, 1.0)

        return features

    def _extract_structural_features(self, sql_content: str) -> torch.Tensor:
        """Extract structural features of the query"""
        features = torch.zeros(32)

        # Query complexity features
        features[0] = len(sql_content) / 2000.0  # Query length
        features[1] = sql_content.count('(') / 20.0  # Parentheses count
        features[2] = sql_content.count('SELECT') / 5.0  # Subquery count

        # Keywords count
        keywords = ['JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'UNION', 'DISTINCT', 'GROUP BY', 'ORDER BY']
        for i, keyword in enumerate(keywords):
            count = len(re.findall(f'\\b{keyword}\\b', sql_content, re.IGNORECASE))
            features[3 + i] = min(count / 3.0, 1.0)

        # Table count
        from_match = re.search(r'FROM\s+(.*?)\s+WHERE', sql_content, re.IGNORECASE | re.DOTALL)
        if from_match:
            table_count = len([t.strip() for t in from_match.group(1).split(',')])
            features[13] = min(table_count / 10.0, 1.0)

        # Generate a hash-based feature for unique identification
        query_hash = int(hashlib.md5(sql_content.encode()).hexdigest()[:8], 16)
        features[14] = (query_hash % 1000) / 1000.0  # Normalized hash

        return features

def test_sql_encoder():
    """Test the SQL encoder"""
    # Load table stats
    with open('data/imdb/processed/table_stats.json') as f:
        table_stats = json.load(f)

    encoder = SQLJoinOrderEncoder(table_stats)

    # Test with sample SQL files
    test_files = [
        'data/imdb/join_orders/1a_order_000.sql',
        'data/imdb/join_orders/1a_order_001.sql'
    ]

    print("Testing SQL encoder:")
    for sql_file in test_files:
        if Path(sql_file).exists():
            encoding = encoder.encode_from_sql_file(sql_file)
            print(f"{sql_file}: {encoding[:8]}...")  # Show first 8 features
        else:
            print(f"{sql_file}: File not found")

if __name__ == "__main__":
    test_sql_encoder()