#!/usr/bin/env python3
"""
Dataset for pairwise ranking of join orders
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import json
from pathlib import Path
from typing import Dict, List
from query_graph import QueryGraphBuilder
from plan_join_encoder import PlanJoinOrderEncoder

class JoinOrderRankingDataset(Dataset):
    """Dataset for learning to rank join orders"""
    
    def __init__(self, data_file: str, table_stats_file: str, cache_dir: str = None):
        with open(data_file) as f:
            self.examples = json.load(f)
        
        with open(table_stats_file) as f:
            self.table_stats = json.load(f)
        
        self.graph_builder = QueryGraphBuilder(self.table_stats)
        self.plan_encoder = PlanJoinOrderEncoder(self.table_stats, 'results/tpch_runtimes.json')
        
        self.graph_cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loaded {len(self.examples)} pairwise ranking examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        better_graph = self._get_graph(example['better_sql_file'])
        worse_graph = self._get_graph(example['worse_sql_file'])
        
        # USE PLAN-BASED ENCODING
        better_order_enc = self.plan_encoder.encode_from_sql_file(example['better_sql_file'])
        worse_order_enc = self.plan_encoder.encode_from_sql_file(example['worse_sql_file'])
        
        return {
            'better_graph': better_graph,
            'worse_graph': worse_graph,
            'better_order_enc': better_order_enc,
            'worse_order_enc': worse_order_enc,
            'speedup': torch.tensor(example['speedup'], dtype=torch.float),
            'better_runtime': torch.tensor(example['better_runtime_ms'], dtype=torch.float),
            'worse_runtime': torch.tensor(example['worse_runtime_ms'], dtype=torch.float)
        }
    
    def _get_graph(self, sql_file: str) -> Data:
        if sql_file in self.graph_cache:
            return self.graph_cache[sql_file]
        
        if self.cache_dir:
            cache_file = self.cache_dir / f"{Path(sql_file).stem}.pt"
            if cache_file.exists():
                try:
                    graph = torch.load(cache_file, weights_only=False)
                    self.graph_cache[sql_file] = graph
                    return graph
                except Exception as e:
                    cache_file.unlink()
        
        graph = self.graph_builder.build_graph(sql_file)
        self.graph_cache[sql_file] = graph
        if self.cache_dir:
            torch.save(graph, cache_file)
        
        return graph

def collate_ranking_batch(batch):
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    better_graphs = [item['better_graph'] for item in batch]
    worse_graphs = [item['worse_graph'] for item in batch]
    
    better_graphs = [g for g in better_graphs if g is not None and g.x.size(0) > 0]
    worse_graphs = [g for g in worse_graphs if g is not None and g.x.size(0) > 0]
    
    min_len = min(len(better_graphs), len(worse_graphs))
    better_graphs = better_graphs[:min_len]
    worse_graphs = worse_graphs[:min_len]
    batch = batch[:min_len]
    
    if min_len == 0:
        return None
    
    better_batch = Batch.from_data_list(better_graphs)
    worse_batch = Batch.from_data_list(worse_graphs)
    
    better_order_encs = torch.stack([item['better_order_enc'] for item in batch[:min_len]])
    worse_order_encs = torch.stack([item['worse_order_enc'] for item in batch[:min_len]])
    speedups = torch.stack([item['speedup'] for item in batch[:min_len]])
    better_runtimes = torch.stack([item['better_runtime'] for item in batch[:min_len]])
    worse_runtimes = torch.stack([item['worse_runtime'] for item in batch[:min_len]])
    
    return {
        'better_batch': better_batch,
        'worse_batch': worse_batch,
        'better_order_encs': better_order_encs,
        'worse_order_encs': worse_order_encs,
        'speedups': speedups,
        'better_runtimes': better_runtimes,
        'worse_runtimes': worse_runtimes
    }