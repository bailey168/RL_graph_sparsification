"""Train/evaluate a fixed GNN (model.type: gcn | gatv2) on top-K sparsified FC graphs across FC types, tests,
keep levels and seeds.

Each keep level is run with a personal mask (every subject keeps their own top-K) and/or a shared
mask (one top-K edge set derived from the training group, applied to everyone). The shared mask is
skipped at keep 0.0 and 1.0, where it gives the same graphs as the personal mask.

With split.per_seed (default) every training seed gets its own train/val/test split, shared by all
keep levels, masks and FC types for that seed. Splits go to splits/seed{s}/ and preprocessors to
{FC}/{TEST}/preprocessors_seed{s}.pkl.

Example:
    python topk/run_sweep.py --fc FC21 --tests GF --keep 0.1 1.0 --max-subjects 2000 --max-epochs 3
"""
import argparse
import fcntl
import gc
import json
import os
import sys
import traceback
import warnings

# torch_geometric calls torch.jit.script internally at import time; silence the resulting deprecation warning
warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*is deprecated", category=FutureWarning)

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
from common.loader import GraphTensors, TensorLoader
from common.models import build_model, graph_format
from common.train import fit
from common.utils import set_deterministic_training
from topk.sparsify import (edge_retention, is_redundant, num_kept_edges, shared_edge_indices, tile_mask,
                           topk_edge_indices)
from topk.summarize import auto_summarize


def parse_args():
    parser = argparse.ArgumentParser(description="Top-K FC sparsification sweep with a fixed GNN")
    parser.add_argument('--config', default='configs/topk.yaml')
    parser.add_argument('--fc', nargs='+', help='FC types (default: all in config)')
    parser.add_argument('--tests', nargs='+', help='Cognitive tests (default: all in config)')
    parser.add_argument('--keep', nargs='+', type=float, help='Keep fractions (default: sweep.keep_fracs)')
    parser.add_argument('--seeds', nargs='+', type=int, help='Training seeds (default: sweep.seeds)')
    parser.add_argument('--masks', nargs='+', choices=['personal', 'shared'],
                        help='Mask types (default: sweep.masks)')
    parser.add_argument('--max-subjects', type=int, help='Subsample subjects (smoke tests)')
    parser.add_argument('--max-epochs', type=int, help='Override training.max_epochs')
    parser.add_argument('--overwrite', action='store_true', help='Rerun runs that already have metrics.json')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='Config overrides')
    parser.add_argument('--no-summary', action='store_true',
                        help="Don't run topk/summarize.py on {output_dir}/results.csv after training")
    return parser.parse_args()


def keep_dir(output_dir, fc_type, test, mask, keep_frac):
    return os.path.join(output_dir, fc_type, test, mask, f'keep{int(round(keep_frac * 100)):03d}')


def run_dir_for(output_dir, fc_type, test, mask, keep_frac, seed):
    return os.path.join(keep_dir(output_dir, fc_type, test, mask, keep_frac), f'seed{seed}')


def select_edges(prepared, keep_frac, mask):
    """Kept edge indices (N, k) per split, for a personal or shared mask."""
    if mask == 'personal':
        return {name: topk_edge_indices(prepared[name]['rank_scores'], keep_frac) for name in SPLITS}, None
    shared = shared_edge_indices(prepared['train']['rank_scores'], keep_frac)
    return {name: tile_mask(shared, len(prepared[name]['eids'])) for name in SPLITS}, shared


def build_split_graphs(prepared, top_idx, edge_list, num_nodes, self_loop_weight, graph_format='gcn_norm'):
    return {name: build_graphs(prepared[name]['edge_vals'], top_idx[name], prepared[name]['demo'],
                               prepared[name]['y'], prepared[name]['eids'], edge_list, num_nodes, self_loop_weight,
                               graph_format=graph_format)
            for name in SPLITS}


def loader_kind(train_cfg):
    kind = train_cfg.get('loader', 'tensor')
    if kind not in ('tensor', 'pyg'):
        raise ValueError(f"Unknown training.loader '{kind}' (expected tensor | pyg)")
    return kind


def stage_graphs(graphs, train_cfg, device):
    """Per-split data for make_loaders: GraphTensors on `device` (training.loader=tensor, the default) or the
    Data lists themselves (pyg). Build once and reuse across runs/trials on the same graphs."""
    if loader_kind(train_cfg) == 'pyg':
        return graphs
    return {name: GraphTensors(graphs[name], device) for name in SPLITS}


def make_loaders(staged, batch_size, num_workers, generator):
    """Train (shuffled with `generator`), val and test loaders. Both loader kinds give identical batches in
    the same order; the tensor loader assembles them on the device without PyG collation (num_workers unused)."""
    loaders = {}
    for name in SPLITS:
        shuffle, gen = name == 'train', generator if name == 'train' else None
        if isinstance(staged[name], GraphTensors):
            loaders[name] = TensorLoader(staged[name], batch_size, shuffle=shuffle, generator=gen)
        else:
            loaders[name] = DataLoader(staged[name], batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, generator=gen)
    return loaders


def prepare_fc(df, test, fc_type, split_idx, split_seed, cfg, output_dir):
    """Arrays + train-fitted preprocessing for one FC type; saves the preprocessors next to the runs."""
    arrays = extract_arrays(df, test, fc_type, cfg)
    prepared, preprocessors = prepare_splits(arrays, split_idx, cfg)
    pre_dir = os.path.join(output_dir, fc_type, test)
    os.makedirs(pre_dir, exist_ok=True)
    joblib.dump(preprocessors, os.path.join(pre_dir, f'preprocessors_seed{split_seed}.pkl'))
    return arrays, prepared, preprocessors


def save_history(out_dir, result):
    pd.DataFrame(result['history']).to_csv(os.path.join(out_dir, 'losses.csv'), index_label='epoch')
    pd.DataFrame(result['test_predictions']).to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)


def save_mask(out_dir, top_idx, shared, num_edges):
    np.save(os.path.join(out_dir, 'edge_retention.npy'), edge_retention(top_idx['train'], num_edges))
    if shared is not None:
        np.save(os.path.join(out_dir, 'shared_edges.npy'), shared)


def result_row(fc_type, test, mask, keep_frac, num_edges, seed, split_seed, graphs, best_epoch, epochs_run,
               metrics, minutes):
    """The results.csv columns shared by run_sweep.py and run_optuna.py, in order."""
    return {
        'fc': fc_type, 'test': test, 'mask': mask, 'keep_frac': keep_frac,
        'k_edges': num_kept_edges(num_edges, keep_frac), 'num_edges': num_edges, 'seed': seed,
        'split_seed': split_seed,
        'n_train': len(graphs['train']), 'n_val': len(graphs['val']), 'n_test': len(graphs['test']),
        'best_epoch': best_epoch, 'epochs_run': epochs_run, **metrics, 'minutes': minutes,
    }


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def append_row(path, row):
    """Append one row; the file lock lets several processes (e.g. parallel studies) share one CSV."""
    with open(path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            pd.DataFrame([row]).to_csv(f, header=f.tell() == 0, index=False)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(cfg, fc_type, test, mask, keep_frac, seed, prepared, preprocessors, arrays, run_dir, device,
            split_seed=None):
    train_cfg = cfg['training']
    num_nodes = cfg['fc_types'][fc_type]['num_nodes']
    num_edges = len(arrays['edge_list'])

    top_idx, shared = select_edges(prepared, keep_frac, mask)
    graphs = build_split_graphs(prepared, top_idx, arrays['edge_list'], num_nodes, cfg['graph']['self_loop_weight'],
                                graph_format(cfg['model']))

    staged = stage_graphs(graphs, train_cfg, device)
    generator = set_deterministic_training(seed)
    loaders = make_loaders(staged, train_cfg['batch_size'], train_cfg['num_workers'], generator)
    model = build_model(cfg['model'], num_nodes, demo_dim=prepared['train']['demo'].shape[1])

    result = fit(model, loaders, train_cfg, device, preprocessors['y'])

    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))
    save_history(run_dir, result)
    save_mask(run_dir, top_idx, shared, num_edges)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(snapshot(cfg, fc=fc_type, test=test, mask=mask, keep_frac=keep_frac, seed=seed), f,
                       sort_keys=False)

    row = result_row(fc_type, test, mask, keep_frac, num_edges, seed, split_seed, graphs, result['best_epoch'],
                     result['epochs_run'], result['metrics'], result['minutes'])
    write_json(os.path.join(run_dir, 'metrics.json'), row)   # written last: marks the run as complete
    return row


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if args.max_epochs is not None:
        cfg['training']['max_epochs'] = args.max_epochs

    fc_types = args.fc or list(cfg['fc_types'])
    tests = args.tests or list(cfg['tests'])
    keep_fracs = args.keep or cfg['sweep']['keep_fracs']
    seeds = args.seeds or cfg['sweep']['seeds']
    masks = args.masks or cfg['sweep']['masks']

    output_dir = cfg['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.csv')
    failures_path = os.path.join(output_dir, 'failures.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    per_seed = cfg['split'].get('per_seed', False)

    def split_seed_for(seed):
        return seed if per_seed else cfg['split']['seed']

    for test in tests:
        pending = [(fc, k, m, s) for fc in fc_types for k in keep_fracs for m in masks for s in seeds
                   if not is_redundant(m, k)
                   and (args.overwrite or not os.path.exists(
                       os.path.join(run_dir_for(output_dir, fc, test, m, k, s), 'metrics.json')))]
        if not pending:
            print(f"[{test}] All runs complete, skipping")
            continue

        df = load_subject_table(test, cfg, args.max_subjects)
        eids = df.index.to_numpy()
        for split_seed in sorted({split_seed_for(s) for _, _, _, s in pending}):
            split_pending = [p for p in pending if split_seed_for(p[3]) == split_seed]
            split_idx = make_split(eids, cfg, seed=split_seed)
            save_split(eids, split_idx, os.path.join(output_dir, 'splits', f'seed{split_seed}'), test)
            print(f"\n[{test}] Split seed {split_seed} - Train: {len(split_idx['train'])}, "
                  f"Val: {len(split_idx['val'])}, Test: {len(split_idx['test'])}")

            for fc_type in fc_types:
                fc_pending = [(k, m, s) for fc, k, m, s in split_pending if fc == fc_type]
                if not fc_pending:
                    continue

                arrays, prepared, preprocessors = prepare_fc(df, test, fc_type, split_idx, split_seed, cfg,
                                                             output_dir)

                for keep_frac, mask, seed in fc_pending:
                    run_dir = run_dir_for(output_dir, fc_type, test, mask, keep_frac, seed)
                    print(f"\n=== {fc_type} | {test} | {mask} | keep={keep_frac:.2f} | seed={seed} ===")
                    try:
                        row = run_one(cfg, fc_type, test, mask, keep_frac, seed, prepared, preprocessors,
                                      arrays, run_dir, device, split_seed=split_seed)
                        append_row(results_path, row)
                        print(f"  best_epoch={row['best_epoch']} test R²={row['test_r2']:.4f} "
                              f"R²corr={row['test_r2_corr']:.4f} ({row['minutes']:.1f}m)")
                    except Exception as e:
                        traceback.print_exc()
                        append_row(failures_path, {'fc': fc_type, 'test': test, 'mask': mask,
                                                   'keep_frac': keep_frac, 'seed': seed, 'error': repr(e)})
                    finally:
                        free_memory()

                del arrays, prepared, preprocessors
                gc.collect()
        del df

    if not args.no_summary:
        auto_summarize(output_dir)


if __name__ == '__main__':
    main()
