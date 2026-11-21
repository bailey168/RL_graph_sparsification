"""Times one training epoch (train + val pass) with the PyG DataLoader vs the tensor loader. Not collected by pytest.

python tests/bench_training.py [--device cuda] [--subjects 20000] [--batch-size 512] [--keep 1.0]
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*is deprecated", category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import parse_region_map
from common.graphs import build_graphs
from common.models import build_model
from common.train import eval_loss, get_optimizer_params, train_epoch
from topk.run_sweep import make_loaders, stage_graphs
from topk.sparsify import topk_edge_indices

REGION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'region_names')
FC = {'FC21': ('FC21_regions.txt', 21), 'FC55': ('FC55_regions.txt', 55)}
MODEL = {'embedding_dim': 16, 'hidden_channels': 64, 'gnn_num_layers': 3, 'classifier_hidden_dims': [32, 16],
         'node_dropout': 0.3, 'edge_dropout': 0.1, 'classifier_dropout': 0.3, 'pooling': 'attention',
         'include_demo': True, 'demo_embed_dim': 8, 'heads': 4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--subjects', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--keep', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--fc', nargs='+', default=list(FC))
    p.add_argument('--models', nargs='+', default=['gatv2', 'gcn'])
    args = p.parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(0)
    n_train, n_val = int(args.subjects * 0.7), int(args.subjects * 0.15)

    for fc in args.fc:
        regions, num_nodes = FC[fc]
        _, edge_list, _ = parse_region_map(os.path.join(REGION_DIR, regions))
        X = rng.normal(0, 0.5, size=(n_train + n_val, len(edge_list))).astype(np.float32)
        demo = rng.normal(size=(len(X), 5)).astype(np.float32)
        y = rng.normal(size=len(X)).astype(np.float32)
        top_idx = topk_edge_indices(np.abs(X), args.keep)
        for kind in args.models:
            fmt = 'gcn_norm' if kind == 'gcn' else 'edge_attr'
            full = build_graphs(X, top_idx, demo, y, np.arange(len(X)), edge_list, num_nodes, 1.0, graph_format=fmt)
            graphs = {'train': full[:n_train], 'val': full[n_train:], 'test': full[n_train:n_train + 1]}
            times = {}
            for loader in ('pyg', 'tensor'):
                t0 = time.perf_counter()
                staged = stage_graphs(graphs, {'loader': loader}, device)
                stage_s = time.perf_counter() - t0
                loaders = make_loaders(staged, args.batch_size, 0, torch.Generator().manual_seed(0))
                model = build_model({**MODEL, 'type': kind}, num_nodes, 5).to(device)
                opt = torch.optim.AdamW(get_optimizer_params(model, 0.01), lr=1e-3)
                crit = nn.MSELoss()
                train_epoch(model, loaders['train'], opt, crit, device)   # warmup
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.epochs):
                    train_epoch(model, loaders['train'], opt, crit, device)
                    eval_loss(model, loaders['val'], crit, device)
                times[loader] = (time.perf_counter() - t0) / args.epochs
                print(f"{fc} {kind:5s} keep={args.keep} {loader:6s}: {times[loader]:.3f} s/epoch "
                      f"(staging {stage_s:.2f} s, once per study)", flush=True)
            print(f"{fc} {kind:5s} speedup x{times['pyg'] / times['tensor']:.1f}\n", flush=True)


if __name__ == '__main__':
    main()
