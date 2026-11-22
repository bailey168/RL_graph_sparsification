"""End-to-end smoke test of topk/run_optuna.py on a synthetic combined_data CSV."""
import json
import os
import subprocess
import sys

import optuna
import pandas as pd
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from test_topk_pipeline import make_fake_csv

N_TRIALS = 3


def run_optuna(tmp_path, *extra, config='topk_optuna.yaml', keep=('0.0', '0.5', '1.0'), n_trials=N_TRIALS):
    cmd = [sys.executable, os.path.join(ROOT, 'topk', 'run_optuna.py'),
           '--config', os.path.join(ROOT, 'configs', config),
           '--fc', 'FC21', '--tests', 'GF', '--keep', *keep,
           '--n-trials', str(n_trials), '--max-epochs', '2',
           '--set', f'paths.data_dir={tmp_path / "data"}', f'paths.output_dir={tmp_path / "out"}',
           f'paths.region_dir={os.path.join(ROOT, "region_names")}', 'split.use_eid_file=false',
           'training.batch_size=64', 'optuna.pruner.enabled=true', 'optuna.pruner.min_resource=1', *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def load_study(study_dir):
    summaries = optuna.get_all_study_summaries(f'sqlite:///{study_dir / "study.db"}')
    return optuna.load_study(study_name=summaries[0].study_name, storage=f'sqlite:///{study_dir / "study.db"}')


def test_optuna_sweep_end_to_end(tmp_path):
    make_fake_csv(tmp_path / 'data')
    proc = run_optuna(tmp_path, '--masks', 'personal', 'shared')   # explicit: config default is personal only
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / 'out'
    assert not (out / 'failures.csv').exists(), (out / 'failures.csv').read_text()

    # personal 0.0, 0.5, 1.0 + shared 0.5 (shared skipped at 0 and 1)
    results = pd.read_csv(out / 'results.csv')
    assert len(results) == 4
    assert set(zip(results['mask'], results['keep_frac'])) == {
        ('personal', 0.0), ('personal', 0.5), ('personal', 1.0), ('shared', 0.5)}
    assert (results['seed'] == 0).all() and (results['split_seed'] == 0).all()
    assert (results['n_trials_complete'] + results['n_pruned'] == N_TRIALS).all()
    assert set(results['fc']) == {'FC21'}   # --fc overrides sweep.fc (FC55)
    params = pd.read_csv(out / 'best_params.csv')
    assert 'pooling' not in params.columns
    assert len(params) == 4 and {'hidden_channels', 'lr', 'edge_dropout', 'classifier_hidden_dims'} <= set(params.columns)

    study_dir = out / 'FC21' / 'GF' / 'shared' / 'keep050'
    for f in ('study.db', 'trials.csv', 'best_params.json', 'config.yaml', 'model.pt', 'losses.csv',
              'test_predictions.csv', 'best_trial.json', 'edge_retention.npy', 'shared_edges.npy', 'metrics.json'):
        assert (study_dir / f).exists(), f
    for name in ('optimization_history', 'parallel_coordinate', 'slice'):
        for ext in ('html', 'png'):
            assert (study_dir / 'plots' / f'{name}.{ext}').exists(), f'{name}.{ext}'
    assert (out / 'splits' / 'seed0' / 'GF_test_eids.txt').exists()
    assert (out / 'FC21' / 'GF' / 'preprocessors_seed0.pkl').exists()

    # The saved checkpoint and config belong to the study's best trial
    study = load_study(study_dir)
    best_trial = json.loads((study_dir / 'best_trial.json').read_text())
    metrics = json.loads((study_dir / 'metrics.json').read_text())
    assert best_trial['trial_number'] == study.best_trial.number == metrics['best_trial']
    assert abs(best_trial['test_r2'] - study.best_trial.user_attrs['test_r2']) < 1e-9

    # Rerun skips completed studies
    proc = run_optuna(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'All studies complete' in proc.stdout
    assert len(pd.read_csv(out / 'results.csv')) == 4

    # An unfinished study resumes and tops up to n_trials, without rerunning finished trials
    (study_dir / 'metrics.json').unlink()
    proc = run_optuna(tmp_path, '--keep', '0.5', '--masks', 'shared', '--n-trials', str(N_TRIALS + 2))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '2 trials to run (3 already finished)' in proc.stdout
    assert len(load_study(study_dir).trials) == N_TRIALS + 2

    # summarize.py ran automatically after training (single seed + Optuna figures)
    assert 'Summarizing' in proc.stdout
    for f in ('test_r2_vs_keep.png', 'test_r2_delta_vs_full.png', 'best_params_vs_keep_GF.png'):
        assert (out / 'summary' / f).exists(), f


def test_gat_optuna_sweep(tmp_path):
    """configs/topk_gat_optuna.yaml: GATv2 studies with the inherited search space plus heads."""
    make_fake_csv(tmp_path / 'data')
    proc = run_optuna(tmp_path, '--masks', 'personal', '--no-summary', config='topk_gat_optuna.yaml',
                      keep=('0.0', '1.0'), n_trials=2)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / 'out'
    assert not (out / 'failures.csv').exists(), (out / 'failures.csv').read_text()
    results = pd.read_csv(out / 'results.csv')
    assert len(results) == 2 and (results['n_trials_complete'] + results['n_pruned'] == 2).all()
    params = pd.read_csv(out / 'best_params.csv')
    assert {'heads', 'hidden_channels', 'edge_dropout', 'lr'} <= set(params.columns)
    study_dir = out / 'FC21' / 'GF' / 'personal' / 'keep100'
    cfg = yaml.safe_load((study_dir / 'config.yaml').read_text())
    assert cfg['model']['type'] == 'gatv2'
    assert cfg['model']['heads'] in (2, 4, 8, 16)
    state = torch.load(study_dir / 'model.pt')
    assert any('att' in k for k in state)   # GATv2Conv attention parameters were saved
