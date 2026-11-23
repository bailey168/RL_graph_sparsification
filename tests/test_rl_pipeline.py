"""End-to-end smoke test of the RL stages (predictor -> agent -> final GNN -> summary) on a synthetic CSV.

The full pipeline runs with the default fixed GNN architecture (rl.gnn_tuning=fixed); the Optuna mode is checked on
the predictor stage.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rl.summarize import topk_results_path
from test_topk_pipeline import make_fake_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_stage(tmp_path, script, *extra):
    cmd = [sys.executable, os.path.join(ROOT, 'rl', script), '--config', os.path.join(ROOT, 'configs', 'rl.yaml'),
           '--fc', 'FC21', '--tests', 'GF',
           '--set', f'paths.data_dir={tmp_path / "data"}', f'paths.output_dir={tmp_path / "out"}',
           f'paths.region_dir={os.path.join(ROOT, "region_names")}', f'paths.topk_results={tmp_path / "topk.csv"}',
           'split.use_eid_file=false', 'training.batch_size=64', 'rl.ppo.subjects_per_iter=32',
           'rl.ppo.minibatch_size=128', 'rl.ppo.eval_every=1', 'rl.ppo.eval_batch_size=50', 'rl.policy.hidden_channels=8',
           'rl.policy.mlp_hidden=8', *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_rl_pipeline_end_to_end(tmp_path):
    make_fake_csv(tmp_path / 'data')
    pd.DataFrame({'fc': 'FC21', 'test': 'GF', 'mask': 'personal', 'keep_frac': [0.0, 0.5, 1.0],
                  'test_r2': [0.1, 0.12, 0.15]}).to_csv(tmp_path / 'topk.csv', index=False)
    base = tmp_path / 'out' / 'FC21' / 'GF'

    proc = run_stage(tmp_path, 'predictor.py', '--max-epochs', '2')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'rl.gnn_tuning=fixed' in proc.stdout
    for f in ('model.pt', 'config.yaml', 'best_trial.json', 'losses.csv', 'r2_by_level.csv', 'timing.json',
              'metrics.json'):
        assert (base / 'predictor' / f).exists(), f
    assert not (base / 'predictor' / 'study.db').exists()   # fixed architecture: no Optuna study
    pred_metrics = json.loads((base / 'predictor' / 'metrics.json').read_text())
    assert pred_metrics['best_trial'] is None
    assert {'time_load_data_min', 'time_gnn_training_min', 'time_eval_by_level_min'} <= set(pred_metrics)
    by_level = pd.read_csv(base / 'predictor' / 'r2_by_level.csv')
    assert len(by_level) == 21 and by_level['k_edges'].iloc[0] == 210 and by_level['k_edges'].iloc[-1] == 0
    splits = {name: np.loadtxt(base / 'splits' / f'GF_{name}_eids.txt')
              for name in ('train', 'val', 'test', 'train_pred', 'train_agent')}
    assert len(splits['train']) + len(splits['val']) + len(splits['test']) == 300
    assert not set(splits['train_pred']) & set(splits['train_agent'])
    assert set(splits['train_pred']) | set(splits['train_agent']) == set(splits['train'])

    # Agent: stops early after 2 evaluations without improvement at most, or after 3 iterations
    proc = run_stage(tmp_path, 'train_agent.py', '--iterations', '3')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    agent = base / 'agent'
    for f in ('best.pt', 'last.pt', 'log.csv', 'evals.csv', 'metrics.json', 'plots/learning_curves.png'):
        assert (agent / f).exists(), f
    metrics = json.loads((agent / 'metrics.json').read_text())
    assert 0.0 <= metrics['test_keep_frac_mean'] <= 1.0 and np.isfinite(metrics['test_mse'])
    assert len(pd.read_csv(agent / 'log.csv')) == metrics['iterations_run']

    # Resume: an unfinished run continues from last.pt instead of restarting
    (agent / 'metrics.json').unlink()
    proc = run_stage(tmp_path, 'train_agent.py', '--iterations', '4')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'Resuming from iteration 3' in proc.stdout
    assert pd.read_csv(agent / 'log.csv')['iteration'].tolist() == [1, 2, 3, 4]
    timing = json.loads((agent / 'timing.json').read_text())
    assert timing['sessions'] == 2 and timing['finished_at']
    assert {'load_data', 'load_predictor', 'setup', 'ppo_rollout', 'ppo_update', 'ppo_eval',
            'final_eval'} <= set(timing['sections'])
    log = pd.read_csv(agent / 'log.csv')
    assert timing['sections']['ppo_rollout'] == pytest.approx(log['rollout_seconds'].sum() / 60, rel=1e-6)

    proc = run_stage(tmp_path, 'final_gnn.py', '--max-epochs', '2')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    final = base / 'final'
    assert (final / 'model.pt').exists() and not (final / 'study.db').exists()
    masks = np.load(final / 'stopped_masks_test.npy')
    stop = np.load(final / 'stop_step_test.npy')
    removal = np.load(final / 'removal_step_test.npy')
    assert masks.shape == (len(splits['test']), 210) and masks.dtype == bool
    assert ((removal >= 0) == ~masks).all()
    sched = [max(1, (20 - t) * 210 // 20) if t < 20 else 0 for t in range(21)]
    assert (masks.sum(1) == np.array(sched)[stop]).all()
    final_metrics = json.loads((final / 'metrics.json').read_text())
    assert {'test_r2', 'predictor_test_r2', 'test_keep_frac_mean', 'time_prune_all_min'} <= set(final_metrics)
    stage_totals = [json.loads((base / s / 'timing.json').read_text())['total_minutes']
                    for s in ('predictor', 'agent', 'final')]
    assert final_metrics['pipeline_minutes_total'] == pytest.approx(sum(stage_totals))
    assert len(pd.read_csv(tmp_path / 'out' / 'results.csv')) == 1

    summary = tmp_path / 'out' / 'summary'
    for f in ('rl_results.csv', 'rl_vs_topk_FC21_GF.png', 'keep_hist_FC21_GF.png', 'r2_by_keep_FC21_GF.csv',
              'edge_retention_FC21_GF.png', 'network_retention_FC21_GF.csv', 'edge_ranking_FC21_GF.csv'):
        assert (summary / f).exists(), f
    assert len(pd.read_csv(summary / 'edge_ranking_FC21_GF.csv')) == 210
    timing_csv = pd.read_csv(summary / 'timing.csv')
    pipe = timing_csv[(timing_csv['stage'] == 'pipeline') & (timing_csv['component'] == 'total')]
    assert len(pipe) == 1 and pipe['minutes'].iloc[0] == pytest.approx(final_metrics['pipeline_minutes_total'])

    # Reruns skip finished stages
    for script in ('predictor.py', 'train_agent.py', 'final_gnn.py'):
        proc = run_stage(tmp_path, script)
        assert proc.returncode == 0 and 'complete, skipping' in proc.stdout, proc.stdout + proc.stderr


def test_rl_predictor_optuna_mode(tmp_path):
    make_fake_csv(tmp_path / 'data')
    proc = run_stage(tmp_path, 'predictor.py', 'rl.gnn_tuning=optuna',   # first: joins run_stage's --set list
                     '--n-trials', '2', '--max-epochs', '2')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    pred = tmp_path / 'out' / 'FC21' / 'GF' / 'predictor'
    for f in ('study.db', 'trials.csv', 'best_params.json', 'model.pt', 'config.yaml', 'best_trial.json',
              'metrics.json'):
        assert (pred / f).exists(), f
    assert len(pd.read_csv(pred / 'trials.csv')) == 2


def test_topk_results_path_follows_gnn_tuning():
    paths = {'topk_results': {'fixed': 'gat.csv', 'optuna': 'gat_optuna.csv'}}
    assert topk_results_path({'paths': paths, 'rl': {}}) == 'gat.csv'   # default: fixed
    assert topk_results_path({'paths': paths, 'rl': {'gnn_tuning': 'optuna'}}) == 'gat_optuna.csv'
    assert topk_results_path({'paths': {'topk_results': 'any.csv'}, 'rl': {'gnn_tuning': 'optuna'}}) == 'any.csv'
    assert topk_results_path({'paths': {}, 'rl': {}}) is None
    with pytest.raises(ValueError):
        topk_results_path({'paths': paths, 'rl': {'gnn_tuning': 'grid'}})
