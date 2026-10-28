"""Times sparse PyG inference vs the dense path. Not collected by pytest.

python tests/bench_inference.py [--device cuda] [--subjects 4096]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import parse_region_map
from common.graphs import build_graphs
from common.infer import predict_dense
from common.models import GCNNet
from topk.sparsify import topk_edge_indices

REGION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'region_names')


def timed(fn, device, reps=5):
    fn()  # warmup
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--subjects', type=int, default=4096)
    ap.add_argument('--keep', type=float, default=0.3)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--compile', action='store_true')
    args = ap.parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(0)

    for fc, n in [('FC21', 21), ('FC55', 55)]:
        _, edge_list, _ = parse_region_map(os.path.join(REGION_DIR, f'{fc}_regions.txt'))
        X = rng.normal(0, 0.3, size=(args.subjects, len(edge_list))).astype(np.float32)
        demo = rng.normal(size=(args.subjects, 8)).astype(np.float32)
        top_idx = topk_edge_indices(np.abs(X), args.keep)
        mask = np.zeros_like(X, dtype=bool)
        np.put_along_axis(mask, top_idx, True, axis=1)

        model = GCNNet(num_nodes=n, embedding_dim=32, hidden_channels=64, gnn_num_layers=2,
                       classifier_hidden_dims=[32], node_dropout=0.3, classifier_dropout=0.3,
                       demo_dim=8).to(device).eval()

        def sparse_end_to_end():   # what the RL loop would pay today: build graphs + collate + forward
            graphs = build_graphs(X, top_idx, demo, np.zeros(len(X), np.float32), np.arange(len(X)), edge_list, n)
            with torch.inference_mode():
                for b in DataLoader(graphs, batch_size=args.batch_size):
                    model(b.to(device))

        graphs = build_graphs(X, top_idx, demo, np.zeros(len(X), np.float32), np.arange(len(X)), edge_list, n)

        def sparse_forward_only():
            with torch.inference_mode():
                for b in DataLoader(graphs, batch_size=args.batch_size):
                    model(b.to(device))

        X_t, demo_t = torch.from_numpy(X).to(device), torch.from_numpy(demo).to(device)
        mask_t, el_t = torch.from_numpy(mask).to(device), torch.from_numpy(edge_list).long().to(device)
        if args.compile:
            model.forward_dense = torch.compile(model.forward_dense, mode='reduce-overhead')

        def dense():
            predict_dense(model, X_t, el_t, n, demo_t, mask=mask_t)

        t_e2e = timed(sparse_end_to_end, device, reps=2)
        t_fwd = timed(sparse_forward_only, device)
        t_dense = timed(dense, device, reps=20)
        print(f'{fc} B={args.subjects} keep={args.keep} [{device}]  '
              f'sparse build+fwd {t_e2e*1e3:8.1f} ms | sparse fwd {t_fwd*1e3:8.1f} ms | '
              f'dense (on-device, 1 pass) {t_dense*1e3:7.2f} ms  '
              f'→ {t_e2e/t_dense:6.0f}× / {t_fwd/t_dense:5.0f}×')


if __name__ == '__main__':
    main()
