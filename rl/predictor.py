"""Stage 1: mask-augmented GATv2 reward predictor (frozen during PPO), fixed architecture or Optuna-tuned.

Trained on half A of the train split. Every training batch draws, per subject, a keep level uniformly from the
n_steps+1 schedule levels (0%, 5%, ..., 100%) and keeps that many edges chosen uniformly at random, so the predictor
gives sensible predictions on any pruned graph the agent can produce. Val/test use fixed level-stratified random
masks; early stopping (and, with Optuna, the objective) uses val MSE (scaled target).
rl.gnn_tuning: 'fixed' (default) trains one GATv2 with the model / training blocks of configs/rl.yaml; 'optuna' runs a
study with the inherited Optuna setup (topk_gat_optuna.yaml: search space, trials, seed, sampler).

Output in {output_dir}/{FC}/{TEST}/predictor/: config.yaml, model.pt, losses.csv, test_predictions.csv,
best_trial.json, r2_by_level.csv/.png (test R² per keep level on random masks), timing.json (minutes per component,
accumulated across resumes), metrics.json (written last; includes time_*_min).
Optuna mode also writes study.db, trials.csv, best_params.json and plots/.

Example:
    python rl/predictor.py --tests GF --max-subjects 2000 --max-epochs 5
    python rl/predictor.py --tests GF --max-subjects 2000 --n-trials 3 --max-epochs 5 --set rl.gnn_tuning=optuna
"""
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*is deprecated", category=FutureWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.masking import level_masks, removal_schedule, stratified_masks
from rl.setup import base_parser, get_device, load_cli_config, load_rl_data, stage_dir
from rl.study import (gnn_tuning_mode, load_trained_gnn, original_scale_metrics, predict_masked,
                      train_stage_gnn)
from rl.timing import TIMING_FILE, StageTimer
from topk.run_sweep import free_memory, write_json

STUDY_FILES = ('study.db', 'metrics.json', 'best_trial.json', TIMING_FILE)


def clear_stage(out_dir, files):
    for f in files:
        path = os.path.join(out_dir, f)
        if os.path.exists(path):
            os.remove(path)


def r2_by_level(model, subjects, data, n_steps, y_pre, seed):
    """Test metrics per schedule level on random masks (a sanity check that the predictor works at every sparsity)."""
    generator = torch.Generator().manual_seed(seed + 3)
    schedule = removal_schedule(data.num_edges, n_steps)
    rows = []
    for level, k in enumerate(schedule):
        mask = level_masks(len(subjects), data.num_edges, level, n_steps, generator)
        pred = predict_masked(model, subjects, mask, data)
        rows.append({'level': level, 'keep_frac': k / data.num_edges, 'k_edges': k,
                     'mse_scaled': float(((pred - subjects.y) ** 2).mean()),
                     **original_scale_metrics(subjects.y, pred, y_pre)})
    return pd.DataFrame(rows)


def plot_by_level(df, path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['keep_frac'] * 100, df['r2'], marker='o')
    ax.set_xlabel('Edges kept (%)')
    ax.set_ylabel('Test R² (random masks)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_predictor(cfg, fc_type, test, data, out_dir, device, n_trials, timer):
    seed, n_steps = cfg['rl']['seed'], cfg['rl']['n_steps']
    generator = torch.Generator().manual_seed(seed + 2)
    sets = {'train': data.sets['train_pred'], 'val': data.sets['val'], 'test': data.sets['test']}
    masks = {'train': None,   # random keep level + random edges, redrawn every batch
             'val': stratified_masks(len(sets['val']), data.num_edges, n_steps, generator),
             'test': stratified_masks(len(sets['test']), data.num_edges, n_steps, generator)}
    label = f"{fc_type}|{test}|predictor"
    with timer.section('gnn_training'):
        res = train_stage_gnn(cfg, out_dir, label, data, sets, masks, device, n_trials)

    with timer.section('eval_by_level'):
        model = load_trained_gnn(out_dir, data.num_nodes, data.demo_dim, device)
        by_level = r2_by_level(model, data.sets['test'], data, n_steps, data.preprocessors['y'], seed)
        by_level.to_csv(os.path.join(out_dir, 'r2_by_level.csv'), index=False)
        plot_by_level(by_level, os.path.join(out_dir, 'r2_by_level.png'), f"Reward predictor | {fc_type} {test}")
    timer.finish()

    saved = res['saved']
    row = {'fc': fc_type, 'test': test, 'stage': 'predictor', 'n_train': len(sets['train']),
           'n_val': len(sets['val']), 'n_test': len(sets['test']), 'best_trial': res['best_trial'],
           'best_val_loss': res['best_val_loss'], 'best_epoch': saved['best_epoch'], 'epochs_run': saved['epochs_run'],
           'n_trials_complete': res['states']['complete'], 'n_pruned': res['states']['pruned'],
           'n_failed': res['states']['failed'], 'study_minutes': res['study_minutes'],
           **{k: saved[k] for k in saved if k.split('_')[0] in ('train', 'val', 'test')},
           'test_r2_full_graph': float(by_level['r2'].iloc[0]), 'test_r2_empty_graph': float(by_level['r2'].iloc[-1]),
           **res['best_values'], **timer.flat()}
    write_json(os.path.join(out_dir, 'metrics.json'), row)   # written last: marks the stage as complete
    return row


def main():
    parser = base_parser("Stage 1: mask-augmented GATv2 reward predictor")
    parser.add_argument('--n-trials', type=int, help='Override optuna.n_trials (rl.gnn_tuning=optuna only)')
    parser.add_argument('--max-epochs', type=int, help='Override training.max_epochs')
    args = parser.parse_args()
    cfg, fc_types, tests = load_cli_config(args)
    if args.max_epochs is not None:
        cfg['training']['max_epochs'] = args.max_epochs
    n_trials = args.n_trials or cfg['optuna']['n_trials']
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = get_device()
    if gnn_tuning_mode(cfg) == 'fixed':
        if args.n_trials is not None:
            print("Note: --n-trials is ignored with rl.gnn_tuning=fixed")
        print(f"Using device: {device} | fixed GNN architecture (rl.gnn_tuning=fixed)")
    else:
        print(f"Using device: {device} | Optuna, {n_trials} trials")

    failed = False
    for test in tests:
        for fc_type in fc_types:
            out_dir = stage_dir(cfg, fc_type, test, 'predictor')
            if args.overwrite:
                clear_stage(out_dir, STUDY_FILES)
            if os.path.exists(os.path.join(out_dir, 'metrics.json')):
                print(f"[{fc_type}|{test}] Predictor complete, skipping")
                continue
            print(f"\n=== {fc_type} | {test} | reward predictor ===")
            try:
                timer = StageTimer(out_dir, device)
                with timer.section('load_data'):
                    data = load_rl_data(cfg, test, fc_type, device, args.max_subjects)
                row = run_predictor(cfg, fc_type, test, data, out_dir, device, n_trials, timer)
                print(f"  best trial #{row['best_trial']}: val MSE={row['best_val_loss']:.4f} "
                      f"test R² full={row['test_r2_full_graph']:.4f} empty={row['test_r2_empty_graph']:.4f} | "
                      f"{row['time_total_min']:.1f} min")
            except Exception:
                traceback.print_exc()
                failed = True
            finally:
                free_memory()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
