#!/usr/bin/env python3
"""
Dataset for few-shot transfer learning experiments
Converts pairwise ranking data to multi-order ranking format
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from query_graph import QueryGraphBuilder
from plan_join_encoder import PlanJoinOrderEncoder

class FewShotRankingDataset(Dataset):
    """
    Dataset for few-shot transfer learning on join order ranking
    Converts pairwise comparisons to multi-order ranking format
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 max_orders_per_query: int = 10,
                 cache_dir: str = None):
        """
        Args:
            data_dir: Directory containing processed data
            split: 'train', 'val', or 'test' (or custom split name)
            max_orders_per_query: Maximum number of orders to include per query
            cache_dir: Directory for caching graph data
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_orders_per_query = max_orders_per_query

        # Load data
        if split.startswith('custom_'):
            # Handle custom splits like from few-shot
            sample_size = split.split('_')[1]
            data_file = self.data_dir.parent / 'few_shot_splits' / f'train_{sample_size}.json'
        else:
            data_file = self.data_dir / f'{split}_data.json'

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file) as f:
            self.raw_examples = json.load(f)

        # Load table stats
        table_stats_file = self.data_dir / 'table_stats.json'
        with open(table_stats_file) as f:
            self.table_stats = json.load(f)

        # Initialize builders
        self.graph_builder = QueryGraphBuilder(self.table_stats)

        # Try to load runtime data for proper encoding
        runtime_file = self.data_dir.parent.parent / 'results' / 'imdb_runtimes.json'
        runtime_data = None
        runtime_file_path = None

        if runtime_file.exists():
            try:
                with open(runtime_file) as f:
                    runtime_data = json.load(f)
                runtime_file_path = str(runtime_file)
            except Exception as e:
                print(f"Warning: Could not load runtime file {runtime_file}: {e}")

        self.plan_encoder = PlanJoinOrderEncoder(self.table_stats, runtime_file_path)

        # Setup caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.graph_cache = {}

        # Group examples by query and convert to ranking format
        self.ranking_examples = self._convert_to_ranking_format()

        print(f"Loaded {len(self.ranking_examples)} ranking examples from {len(self.raw_examples)} pairwise comparisons")

    def _convert_to_ranking_format(self) -> List[Dict[str, Any]]:
        """Convert pairwise comparisons to multi-order ranking examples"""

        # Group by base query
        query_groups = defaultdict(list)
        for example in self.raw_examples:
            base_query = example.get('query_id', example.get('base_query', 'unknown'))
            query_groups[base_query].append(example)

        ranking_examples = []

        for query_id, examples in query_groups.items():
            # Collect all unique SQL files and their runtimes for this query
            order_data = {}

            for example in examples:
                # Add better order
                better_file = example['better_sql_file']
                better_runtime = example['better_runtime_ms']
                if better_file not in order_data:
                    order_data[better_file] = {
                        'runtime': better_runtime,
                        'sql_file': better_file
                    }

                # Add worse order
                worse_file = example['worse_sql_file']
                worse_runtime = example['worse_runtime_ms']
                if worse_file not in order_data:
                    order_data[worse_file] = {
                        'runtime': worse_runtime,
                        'sql_file': worse_file
                    }

            # Sort by runtime (ascending = better)
            sorted_orders = sorted(order_data.values(), key=lambda x: x['runtime'])

            # Limit number of orders
            if len(sorted_orders) > self.max_orders_per_query:
                sorted_orders = sorted_orders[:self.max_orders_per_query]

            # Create ranking targets (normalized inverse runtime)
            runtimes = [order['runtime'] for order in sorted_orders]
            if len(runtimes) > 1:
                # Convert to ranking scores (lower runtime = higher score)
                max_runtime = max(runtimes)
                min_runtime = min(runtimes)
                if max_runtime > min_runtime:
                    targets = [(max_runtime - runtime) / (max_runtime - min_runtime) for runtime in runtimes]
                else:
                    targets = [1.0] * len(runtimes)
            else:
                targets = [1.0] * len(runtimes)

            # Create ranking example
            ranking_example = {
                'query_id': query_id,
                'sql_files': [order['sql_file'] for order in sorted_orders],
                'runtimes': runtimes,
                'targets': targets,
                'num_orders': len(sorted_orders)
            }

            ranking_examples.append(ranking_example)

        return ranking_examples

    def __len__(self):
        return len(self.ranking_examples)

    def __getitem__(self, idx):
        example = self.ranking_examples[idx]

        # Get query graph (same for all orders of this query)
        # Use the first SQL file to build the base query graph
        base_sql_file = example['sql_files'][0]
        query_graph = self._get_graph(base_sql_file)

        if query_graph is None or query_graph.x.size(0) == 0:
            return None

        # Encode all join orders for this query
        order_encodings = []
        for sql_file in example['sql_files']:
            try:
                # Normalize path for cross-platform compatibility
                normalized_path = str(Path(sql_file))
                order_enc = self.plan_encoder.encode_from_sql_file(normalized_path)
                order_encodings.append(order_enc)
            except Exception as e:
                print(f"Warning: Failed to encode {sql_file}: {e}")
                # Use zero encoding as fallback
                order_encodings.append(torch.zeros(32))  # Assuming 32-dim encoding

        if not order_encodings:
            return None

        # Stack order encodings
        orders = torch.stack(order_encodings)  # [num_orders, encoding_dim]

        # Convert targets to tensor
        targets = torch.tensor(example['targets'], dtype=torch.float)

        return {
            'x': query_graph.x,
            'edge_index': query_graph.edge_index,
            'edge_attr': query_graph.edge_attr,
            'orders': orders,
            'targets': targets,
            'query_id': example['query_id'],
            'num_orders': example['num_orders']
        }

    def _get_graph(self, sql_file: str) -> Optional[Data]:
        """Get or build query graph for SQL file"""
        if sql_file in self.graph_cache:
            return self.graph_cache[sql_file]

        # Try cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{Path(sql_file).stem}.pt"
            if cache_file.exists():
                try:
                    graph = torch.load(cache_file, weights_only=False)
                    self.graph_cache[sql_file] = graph
                    return graph
                except Exception as e:
                    print(f"Warning: Cache file corrupted, removing: {cache_file}")
                    cache_file.unlink()

        # Build graph
        try:
            # Normalize path for cross-platform compatibility
            normalized_path = str(Path(sql_file))
            graph = self.graph_builder.build_graph(normalized_path)
            self.graph_cache[sql_file] = graph

            # Save to cache
            if self.cache_dir:
                torch.save(graph, cache_file)

            return graph
        except Exception as e:
            print(f"Warning: Failed to build graph for {sql_file}: {e}")
            return None

def collate_fewshot_batch(batch):
    """Collate function for few-shot ranking dataset"""
    # Filter out None items
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # Find max number of orders in batch
    max_orders = max(item['num_orders'] for item in batch)

    # Pad orders and targets to max_orders
    padded_orders = []
    padded_targets = []

    for item in batch:
        orders = item['orders']
        targets = item['targets']
        num_orders = item['num_orders']

        if num_orders < max_orders:
            # Pad with zeros
            padding_orders = torch.zeros(max_orders - num_orders, orders.size(1))
            padding_targets = torch.zeros(max_orders - num_orders)

            orders = torch.cat([orders, padding_orders], dim=0)
            targets = torch.cat([targets, padding_targets], dim=0)

        padded_orders.append(orders)
        padded_targets.append(targets)

    # Stack
    orders_batch = torch.stack(padded_orders)  # [batch_size, max_orders, encoding_dim]
    targets_batch = torch.stack(padded_targets)  # [batch_size, max_orders]

    # Create graph batch
    graphs = []
    for item in batch:
        graph = Data(
            x=item['x'],
            edge_index=item['edge_index'],
            edge_attr=item['edge_attr']
        )
        graphs.append(graph)

    graph_batch = Batch.from_data_list(graphs)

    # Create result batch
    result = Batch(
        x=graph_batch.x,
        edge_index=graph_batch.edge_index,
        edge_attr=graph_batch.edge_attr,
        batch=graph_batch.batch,
        orders=orders_batch,
        targets=targets_batch,
        query_ids=[item['query_id'] for item in batch]
    )

    return result

# Alias for backward compatibility
RankingDataset = FewShotRankingDataset

if __name__ == "__main__":
    # Test the dataset
    print("Testing FewShotRankingDataset...")

    # Test with IMDB data
    try:
        dataset = FewShotRankingDataset(
            'data/imdb/processed',
            split='train',
            max_orders_per_query=5
        )

        print(f"Dataset length: {len(dataset)}")

        # Test first item
        item = dataset[0]
        if item is not None:
            print(f"First item:")
            print(f"  Query ID: {item['query_id']}")
            print(f"  Graph nodes: {item['x'].shape}")
            print(f"  Graph edges: {item['edge_index'].shape}")
            print(f"  Orders: {item['orders'].shape}")
            print(f"  Targets: {item['targets'].shape}")

        # Test DataLoader
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_fewshot_batch,
            shuffle=False
        )

        batch = next(iter(loader))
        if batch is not None:
            print(f"\nBatch test:")
            print(f"  Batch nodes: {batch.x.shape}")
            print(f"  Batch edges: {batch.edge_index.shape}")
            print(f"  Batch orders: {batch.orders.shape}")
            print(f"  Batch targets: {batch.targets.shape}")

        print("Dataset test completed successfully!")

    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()