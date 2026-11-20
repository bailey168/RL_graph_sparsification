"""Top-K sweep where each (FC, test, mask, keep) level gets its own Optuna hyperparameter search.

For every study the graphs are built once, then `optuna.n_trials` trials train a GCN with sampled
model/optimizer hyperparameters (search space in the config). The objective is the best val MSE from
early stopping; test metrics are recorded per trial but never used for selection. All trials use one
seed (optuna.seed) for the split, the training and the sampler. The best trial's weights are the final
model for that sparsity level.

Output per study, {output_dir}/{FC}/{TEST}/{mask}/keep{pp}/:
  study.db  trials.csv  best_params.json  param_importances.json  config.yaml (best params applied)
  model.pt  losses.csv  test_predictions.csv  best_trial.json  edge_retention.npy  [shared_edges.npy]
  plots/{optimization_history,parallel_coordinate,param_importances,slice}.{html,png}
  metrics.json (written last; marks the study as complete)
Plus {output_dir}/results.csv (same columns as run_sweep.py, seed = optuna.seed) and best_params.csv.
At the end, topk/summarize.py runs on results.csv into {output_dir}/summary/ (unless --no-summary).

A study with study.db but no metrics.json is resumed and topped up to n_trials finished trials.

Example:
    python topk/run_optuna.py --fc FC21 --tests GF --keep 0.1 1.0 --max-subjects 2000 --n-trials 5 --max-epochs 5
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*is deprecated", category=FutureWarning)

import optuna
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load_config, snapshot
from common.data import SPLITS, load_subject_table, make_split, save_split
from common.models import build_model
from common.train import fit
from common.tuning import (config_from_params, count_states, flatten_values, is_new_best, make_study,
                           progress_callback, pruner_enabled, pruning_callback, remaining_trials, save_study_outputs,
                           suggest_config)
from common.utils import set_deterministic_training
from topk.run_sweep import (append_row, build_split_graphs, free_memory, keep_dir, make_loaders, prepare_fc,
                            result_row, save_history, save_mask, select_edges, write_json)
from topk.sparsify import is_redundant
from topk.summarize import auto_summarize


def parse_args():
    parser = argparse.ArgumentParser(description="Top-K sparsification sweep with per-level Optuna tuning")
    parser.add_argument('--config', default='configs/topk_optuna.yaml')
    parser.add_argument('--fc', nargs='+', help='FC types (default: sweep.fc, else all in config)')
    parser.add_argument('--tests', nargs='+', help='Cognitive tests (default: sweep.tests, else all in config)')
    parser.add_argument('--keep', nargs='+', type=float, help='Keep fractions (default: sweep.keep_fracs)')
    parser.add_argument('--masks', nargs='+', choices=['personal', 'shared'],
                        help='Mask types (default: sweep.masks)')
    parser.add_argument('--n-trials', type=int, help='Override optuna.n_trials')
    parser.add_argument('--max-subjects', type=int, help='Subsample subjects (smoke tests)')
    parser.add_argument('--max-epochs', type=int, help='Override training.max_epochs')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete existing study.db/metrics.json and start the study from scratch')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='Config overrides')
    parser.add_argument('--no-summary', action='store_true',
                        help="Don't run topk/summarize.py on {output_dir}/results.csv after training")
    return parser.parse_args()


def train_trial(trial_cfg, graphs, demo_dim, num_nodes, seed, device, y_pre, epoch_callback=None):
    generator = set_deterministic_training(seed)
    train_cfg = trial_cfg['training']
    loaders = make_loaders(graphs, train_cfg['batch_size'], train_cfg['num_workers'], generator)
    model = build_model(trial_cfg['model'], num_nodes, demo_dim)
    result = fit(model, loaders, train_cfg, device, y_pre, epoch_callback=epoch_callback)
    return model, result


def write_best_artifacts(study_dir, trial_number, model, result):
    """Checkpoint of the study's best trial so far. best_trial.json is written last and names the trial."""
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(study_dir, 'model.pt'))
    save_history(study_dir, result)
    write_json(os.path.join(study_dir, 'best_trial.json'),
               {'trial_number': trial_number, 'best_val_loss': result['best_val_loss'],
                'best_epoch': result['best_epoch'], 'epochs_run': result['epochs_run'],
                'minutes': result['minutes'], **result['metrics']})


def read_best_trial(study_dir):
    path = os.path.join(study_dir, 'best_trial.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_study(cfg, fc_type, test, mask, keep_frac, prepared, preprocessors, arrays, study_dir, device, n_trials):
    """Run (or resume) one study, then write its outputs. Returns (results row, best-params row)."""
    opt_cfg = cfg['optuna']
    space = opt_cfg['search_space']
    seed = opt_cfg['seed']
    num_nodes = cfg['fc_types'][fc_type]['num_nodes']
    num_edges = len(arrays['edge_list'])
    demo_dim = prepared['train']['demo'].shape[1]
    y_pre = preprocessors['y']
    os.makedirs(study_dir, exist_ok=True)

    top_idx, shared = select_edges(prepared, keep_frac, mask)
    graphs = build_split_graphs(prepared, top_idx, arrays['edge_list'], num_nodes, cfg['graph']['self_loop_weight'])

    def objective(trial):
        trial_cfg, values = suggest_config(trial, cfg, space)
        try:
            model, result = train_trial(trial_cfg, graphs, demo_dim, num_nodes, seed, device, y_pre,
                                        epoch_callback=pruning_callback(trial))
        except FloatingPointError as e:   # non-finite loss: treat as a bad region of the search space
            trial.set_user_attr('error', repr(e))
            raise optuna.TrialPruned() from e
        for key, value in result['metrics'].items():
            trial.set_user_attr(key, value)
        trial.set_user_attr('best_epoch', result['best_epoch'])
        trial.set_user_attr('epochs_run', result['epochs_run'])
        trial.set_user_attr('minutes', result['minutes'])
        value = result['best_val_loss']
        if is_new_best(trial.study, value):
            write_best_artifacts(study_dir, trial.number, model, result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return value

    label = f"{fc_type}|{test}|{mask}|keep={keep_frac:.2f}"
    study = make_study(os.path.join(study_dir, 'study.db'), label, opt_cfg, cfg['training']['max_epochs'])
    todo = remaining_trials(study, n_trials)
    start = time.time()
    if todo:
        print(f"  {todo} trials to run ({n_trials - todo} already finished)")
        study.optimize(objective, n_trials=todo, callbacks=[progress_callback(label)], gc_after_trial=True)
    states = count_states(study)
    if states['complete'] == 0:
        raise RuntimeError(f"No completed trials ({states})")

    best = study.best_trial
    best_cfg, best_values = config_from_params(cfg, space, best.params)
    saved = read_best_trial(study_dir)
    if saved is None or saved['trial_number'] != best.number:
        # e.g. the process died between writing a checkpoint and Optuna recording the trial
        print(f"  Checkpoint does not match best trial #{best.number}; retraining it (same seed)")
        model, result = train_trial(best_cfg, graphs, demo_dim, num_nodes, seed, device, y_pre)
        write_best_artifacts(study_dir, best.number, model, result)
        saved = read_best_trial(study_dir)

    save_study_outputs(study, study_dir, importance_seed=seed)
    save_mask(study_dir, top_idx, shared, num_edges)
    with open(os.path.join(study_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(snapshot(best_cfg, fc=fc_type, test=test, mask=mask, keep_frac=keep_frac, seed=seed,
                                best_trial=best.number), f, sort_keys=False)

    metrics = {k: saved[k] for k in saved if k.split('_')[0] in SPLITS}
    row = {
        **result_row(fc_type, test, mask, keep_frac, num_edges, seed, seed, graphs, saved['best_epoch'],
                     saved['epochs_run'], metrics, saved['minutes']),
        'best_trial': best.number, 'best_val_loss': best.value,
        'n_trials_complete': states['complete'], 'n_pruned': states['pruned'], 'n_failed': states['failed'],
        'study_minutes': (time.time() - start) / 60.0,
    }
    params_row = {'fc': fc_type, 'test': test, 'mask': mask, 'keep_frac': keep_frac, 'best_trial': best.number,
                  'best_val_loss': best.value, 'test_r2': saved['test_r2'], **flatten_values(best_values)}
    write_json(os.path.join(study_dir, 'metrics.json'), row)   # written last: marks the study as complete
    return row, params_row


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if args.max_epochs is not None:
        cfg['training']['max_epochs'] = args.max_epochs
    n_trials = args.n_trials or cfg['optuna']['n_trials']
    seed = cfg['optuna']['seed']
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fc_types = args.fc or cfg['sweep'].get('fc') or list(cfg['fc_types'])
    tests = args.tests or cfg['sweep'].get('tests') or list(cfg['tests'])
    keep_fracs = args.keep or cfg['sweep']['keep_fracs']
    masks = args.masks or cfg['sweep']['masks']

    output_dir = cfg['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.csv')
    params_path = os.path.join(output_dir, 'best_params.csv')
    failures_path = os.path.join(output_dir, 'failures.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pruner_cfg = cfg['optuna'].get('pruner')
    pruner = pruner_cfg['type'] if pruner_enabled(pruner_cfg) else 'off'
    print(f"Using device: {device} | {n_trials} trials per study | seed {seed} | pruner: {pruner}")

    for test in tests:
        studies = [(fc, k, m) for fc in fc_types for k in keep_fracs for m in masks if not is_redundant(m, k)]
        if args.overwrite:
            for fc, k, m in studies:
                for f in ('study.db', 'metrics.json', 'best_trial.json'):
                    path = os.path.join(keep_dir(output_dir, fc, test, m, k), f)
                    if os.path.exists(path):
                        os.remove(path)
        pending = [(fc, k, m) for fc, k, m in studies
                   if not os.path.exists(os.path.join(keep_dir(output_dir, fc, test, m, k), 'metrics.json'))]
        if not pending:
            print(f"[{test}] All studies complete, skipping")
            continue

        df = load_subject_table(test, cfg, args.max_subjects)
        eids = df.index.to_numpy()
        split_idx = make_split(eids, cfg, seed=seed)
        save_split(eids, split_idx, os.path.join(output_dir, 'splits', f'seed{seed}'), test)
        print(f"\n[{test}] Split seed {seed} - Train: {len(split_idx['train'])}, "
              f"Val: {len(split_idx['val'])}, Test: {len(split_idx['test'])}")

        for fc_type in fc_types:
            fc_pending = [(k, m) for fc, k, m in pending if fc == fc_type]
            if not fc_pending:
                continue
            arrays, prepared, preprocessors = prepare_fc(df, test, fc_type, split_idx, seed, cfg, output_dir)

            for keep_frac, mask in fc_pending:
                study_dir = keep_dir(output_dir, fc_type, test, mask, keep_frac)
                print(f"\n=== {fc_type} | {test} | {mask} | keep={keep_frac:.2f} ===")
                try:
                    row, params_row = run_study(cfg, fc_type, test, mask, keep_frac, prepared, preprocessors,
                                                arrays, study_dir, device, n_trials)
                    append_row(results_path, row)
                    append_row(params_path, params_row)
                    print(f"  best trial #{row['best_trial']}: val MSE={row['best_val_loss']:.4f} "
                          f"test R²={row['test_r2']:.4f} R²corr={row['test_r2_corr']:.4f} "
                          f"({row['n_trials_complete']} complete, {row['n_pruned']} pruned, "
                          f"{row['study_minutes']:.1f}m)")
                except Exception as e:
                    traceback.print_exc()
                    append_row(failures_path, {'fc': fc_type, 'test': test, 'mask': mask,
                                               'keep_frac': keep_frac, 'error': repr(e)})
                finally:
                    free_memory()

            del arrays, prepared, preprocessors
            gc.collect()
        del df

    if not args.no_summary:
        auto_summarize(output_dir)


if __name__ == '__main__':
    main()
