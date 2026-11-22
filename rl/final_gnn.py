"""Stage 3: prune every subject with the trained agent, then train the final GATv2 on the agent-stopped graphs.

The best stage-2 agent prunes each train (A+B), val and test subject greedily until it STOPs (or reaches 0 edges).
A fresh GATv2 is then tuned with the same Optuna setup as the top-K sweep (search space, trials, seed, TPE), trained on
the train subjects' stopped graphs, selected on val MSE and evaluated once on test. Each subject has its own sparsity.

Output in {output_dir}/{FC}/{TEST}/final/:
  stopped_masks_{split}.npy (n, E) bool   removal_step_{split}.npy (n, E) int8 (-1 = kept)   stop_step_{split}.npy
  eids_{split}.npy   Optuna outputs (study.db, trials.csv, model.pt, config.yaml, losses.csv, test_predictions.csv, ...)
  metrics.json (written last). Also appends a row to {output_dir}/results.csv, then runs rl/summarize.py.

Example:
    python rl/final_gnn.py --tests GF --max-subjects 2000 --n-trials 3 --max-epochs 5
"""
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*is deprecated", category=FutureWarning)

import numpy as np
import optuna
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.env import PruningEnv
from rl.ppo import greedy_metrics, run_greedy
from rl.predictor import STUDY_FILES, clear_stage
from rl.setup import SPLITS, base_parser, get_device, load_cli_config, load_rl_data, stage_dir
from rl.study import run_masked_study
from rl.summarize import auto_summarize
from rl.train_agent import load_agent, load_frozen_predictor
from topk.run_sweep import append_row, free_memory, write_json


def prune_all(cfg, data, predictor, policy, out_dir):
    """Greedy agent masks for train/val/test, saved as .npy. Returns ({split: mask tensor}, frozen-predictor metrics)."""
    env = PruningEnv(predictor, data, cfg['rl']['n_steps'])
    masks, metrics = {}, {}
    for name in SPLITS:
        subjects = data.sets[name]
        res = run_greedy(policy, env, subjects, cfg['rl']['ppo']['eval_batch_size'])
        np.save(os.path.join(out_dir, f'stopped_masks_{name}.npy'), res['mask'].numpy())
        np.save(os.path.join(out_dir, f'removal_step_{name}.npy'), res['removal_step'].numpy())
        np.save(os.path.join(out_dir, f'stop_step_{name}.npy'), res['stop_step'].numpy().astype(np.int8))
        np.save(os.path.join(out_dir, f'eids_{name}.npy'), subjects.eid.cpu().numpy())
        masks[name] = res['mask']
        metrics.update(greedy_metrics(res, subjects, data.preprocessors['y'], data.num_edges, f'predictor_{name}_'))
    return masks, metrics


def run_final(cfg, fc_type, test, data, out_dir, device, n_trials):
    os.makedirs(out_dir, exist_ok=True)
    predictor = load_frozen_predictor(cfg, fc_type, test, data, device)
    agent_dir = stage_dir(cfg, fc_type, test, 'agent')
    if not os.path.exists(os.path.join(agent_dir, 'metrics.json')):
        raise FileNotFoundError(f"Run rl/train_agent.py first (no {agent_dir}/metrics.json)")
    policy = load_agent(cfg, data, agent_dir, device)
    masks, predictor_metrics = prune_all(cfg, data, predictor, policy, out_dir)
    del predictor, policy
    free_memory()

    sets = {name: data.sets[name] for name in SPLITS}
    res = run_masked_study(cfg, out_dir, f"{fc_type}|{test}|final", data, sets, masks, device, n_trials)
    saved = res['saved']
    keep = {name: masks[name].sum(1).float() / data.num_edges for name in SPLITS}
    row = {'fc': fc_type, 'test': test, 'stage': 'final',
           'n_train': len(sets['train']), 'n_val': len(sets['val']), 'n_test': len(sets['test']),
           **{f'{name}_keep_frac_mean': keep[name].mean().item() for name in SPLITS},
           **{f'{name}_keep_frac_sd': keep[name].std().item() for name in SPLITS},
           'best_trial': res['best_trial'], 'best_val_loss': res['best_val_loss'],
           'best_epoch': saved['best_epoch'], 'epochs_run': saved['epochs_run'],
           'n_trials_complete': res['states']['complete'], 'n_pruned': res['states']['pruned'],
           'n_failed': res['states']['failed'], 'study_minutes': res['study_minutes'],
           **{k: saved[k] for k in saved if k.split('_')[0] in SPLITS},
           **predictor_metrics, **res['best_values']}
    write_json(os.path.join(out_dir, 'metrics.json'), row)   # written last: marks the stage as complete
    return row


def main():
    parser = base_parser("Stage 3: final GATv2 on agent-stopped graphs")
    parser.add_argument('--n-trials', type=int, help='Override optuna.n_trials')
    parser.add_argument('--max-epochs', type=int, help='Override training.max_epochs')
    parser.add_argument('--no-summary', action='store_true', help="Don't run rl/summarize.py at the end")
    args = parser.parse_args()
    cfg, fc_types, tests = load_cli_config(args)
    if args.max_epochs is not None:
        cfg['training']['max_epochs'] = args.max_epochs
    n_trials = args.n_trials or cfg['optuna']['n_trials']
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = get_device()
    print(f"Using device: {device} | {n_trials} trials")
    results_path = os.path.join(cfg['paths']['output_dir'], 'results.csv')

    failed = False
    for test in tests:
        for fc_type in fc_types:
            out_dir = stage_dir(cfg, fc_type, test, 'final')
            if args.overwrite:
                clear_stage(out_dir, STUDY_FILES)
            if os.path.exists(os.path.join(out_dir, 'metrics.json')):
                print(f"[{fc_type}|{test}] Final GNN complete, skipping")
                continue
            print(f"\n=== {fc_type} | {test} | final GNN on agent-stopped graphs ===")
            try:
                data = load_rl_data(cfg, test, fc_type, device, args.max_subjects, save=False)
                row = run_final(cfg, fc_type, test, data, out_dir, device, n_trials)
                append_row(results_path, row)
                print(f"  test R²={row['test_r2']:.4f} R²corr={row['test_r2_corr']:.4f} | "
                      f"edges kept {row['test_keep_frac_mean']:.2f}±{row['test_keep_frac_sd']:.2f} | "
                      f"frozen predictor test R²={row['predictor_test_r2']:.4f}")
            except Exception:
                traceback.print_exc()
                failed = True
            finally:
                free_memory()

    if not args.no_summary:
        auto_summarize(cfg)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
