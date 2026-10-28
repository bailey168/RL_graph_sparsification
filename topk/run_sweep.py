"""Train/evaluate a fixed GCN on top-K sparsified FC graphs across FC types, tests, keep levels and seeds.

Example:
    python topk/run_sweep.py --fc FC21 --tests GF --keep 0.1 1.0 --max-subjects 2000 --max-epochs 3
"""
import argparse
import gc
import json
import os
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load_config, snapshot
from common.data import SPLITS, extract_arrays, load_subject_table, make_split, prepare_splits, save_split
from common.graphs import build_graphs
from common.models import build_model
from topk.sparsify import edge_retention, num_kept_edges, topk_edge_indices
from common.train import fit
from common.utils import set_deterministic_training


def parse_args():
    parser = argparse.ArgumentParser(description="Top-K FC sparsification sweep with a fixed GCN")
    parser.add_argument('--config', default='configs/topk.yaml')
    parser.add_argument('--fc', nargs='+', help='FC types (default: all in config)')
    parser.add_argument('--tests', nargs='+', help='Cognitive tests (default: all in config)')
    parser.add_argument('--keep', nargs='+', type=float, help='Keep fractions (default: sweep.keep_fracs)')
    parser.add_argument('--seeds', nargs='+', type=int, help='Training seeds (default: sweep.seeds)')
    parser.add_argument('--max-subjects', type=int, help='Subsample subjects (smoke tests)')
    parser.add_argument('--max-epochs', type=int, help='Override training.max_epochs')
    parser.add_argument('--overwrite', action='store_true', help='Rerun runs that already have metrics.json')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='Config overrides')
    return parser.parse_args()


def run_dir_for(output_dir, fc_type, test, keep_frac, seed):
    return os.path.join(output_dir, fc_type, test, f'keep{int(round(keep_frac * 100)):03d}', f'seed{seed}')


def make_loaders(graphs, batch_size, num_workers, generator):
    return {
        name: DataLoader(graphs[name], batch_size=batch_size, shuffle=(name == 'train'),
                         num_workers=num_workers, generator=generator if name == 'train' else None)
        for name in SPLITS
    }


def run_one(cfg, fc_type, test, keep_frac, seed, prepared, preprocessors, arrays, run_dir, device):
    graph_cfg, train_cfg = cfg['graph'], cfg['training']
    num_nodes = cfg['fc_types'][fc_type]['num_nodes']
    num_edges = len(arrays['edge_list'])

    graphs, top_idx = {}, {}
    for name in SPLITS:
        p = prepared[name]
        top_idx[name] = topk_edge_indices(p['rank_scores'], keep_frac)
        graphs[name] = build_graphs(p['edge_vals'], top_idx[name], p['demo'], p['y'], p['eids'],
                                    arrays['edge_list'], num_nodes, graph_cfg['self_loop_weight'])

    generator = set_deterministic_training(seed)
    loaders = make_loaders(graphs, train_cfg['batch_size'], train_cfg['num_workers'], generator)
    model = build_model(cfg['model'], num_nodes, demo_dim=prepared['train']['demo'].shape[1])

    result = fit(model, loaders, train_cfg, device, preprocessors['y'])

    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))
    pd.DataFrame(result['history']).to_csv(os.path.join(run_dir, 'losses.csv'), index_label='epoch')
    pd.DataFrame(result['test_predictions']).to_csv(os.path.join(run_dir, 'test_predictions.csv'), index=False)
    np.save(os.path.join(run_dir, 'edge_retention.npy'), edge_retention(top_idx['train'], num_edges))
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(snapshot(cfg, fc=fc_type, test=test, keep_frac=keep_frac, seed=seed), f, sort_keys=False)

    row = {
        'fc': fc_type, 'test': test, 'keep_frac': keep_frac,
        'k_edges': num_kept_edges(num_edges, keep_frac), 'num_edges': num_edges, 'seed': seed,
        'n_train': len(graphs['train']), 'n_val': len(graphs['val']), 'n_test': len(graphs['test']),
        'best_epoch': result['best_epoch'], 'epochs_run': result['epochs_run'],
        **result['metrics'], 'minutes': result['minutes'],
    }
    # metrics.json is written last: its presence marks the run as complete
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(row, f, indent=2)
    return row


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if args.max_epochs is not None:
        cfg['training']['max_epochs'] = args.max_epochs

    fc_types = args.fc or list(cfg['fc_types'])
    tests = args.tests or list(cfg['tests'])
    keep_fracs = args.keep or cfg['sweep']['keep_fracs']
    seeds = args.seeds or cfg['sweep']['seeds']

    output_dir = cfg['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.csv')
    failures_path = os.path.join(output_dir, 'failures.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for test in tests:
        pending = [(fc, k, s) for fc in fc_types for k in keep_fracs for s in seeds
                   if args.overwrite or not os.path.exists(
                       os.path.join(run_dir_for(output_dir, fc, test, k, s), 'metrics.json'))]
        if not pending:
            print(f"[{test}] All runs complete, skipping")
            continue

        df = load_subject_table(test, cfg, args.max_subjects)
        split_idx = make_split(df.index.to_numpy(), cfg)
        save_split(df.index.to_numpy(), split_idx, os.path.join(output_dir, 'splits'), test)
        print(f"[{test}] Split - Train: {len(split_idx['train'])}, Val: {len(split_idx['val'])}, "
              f"Test: {len(split_idx['test'])}")

        for fc_type in fc_types:
            fc_pending = [(k, s) for fc, k, s in pending if fc == fc_type]
            if not fc_pending:
                continue

            arrays = extract_arrays(df, test, fc_type, cfg)
            prepared, preprocessors = prepare_splits(arrays, split_idx, cfg)
            pre_dir = os.path.join(output_dir, fc_type, test)
            os.makedirs(pre_dir, exist_ok=True)
            joblib.dump(preprocessors, os.path.join(pre_dir, 'preprocessors.pkl'))

            for keep_frac, seed in fc_pending:
                run_dir = run_dir_for(output_dir, fc_type, test, keep_frac, seed)
                print(f"\n=== {fc_type} | {test} | keep={keep_frac:.2f} | seed={seed} ===")
                try:
                    row = run_one(cfg, fc_type, test, keep_frac, seed, prepared, preprocessors,
                                  arrays, run_dir, device)
                    append_row(results_path, row)
                    print(f"  best_epoch={row['best_epoch']} test R²={row['test_r2']:.4f} "
                          f"R²corr={row['test_r2_corr']:.4f} ({row['minutes']:.1f}m)")
                except Exception as e:
                    traceback.print_exc()
                    append_row(failures_path, {'fc': fc_type, 'test': test, 'keep_frac': keep_frac,
                                               'seed': seed, 'error': repr(e)})
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            del arrays, prepared, preprocessors
            gc.collect()
        del df


if __name__ == '__main__':
    main()
