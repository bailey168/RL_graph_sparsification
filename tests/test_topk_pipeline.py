"""End-to-end smoke test of topk/run_sweep.py on a synthetic combined_data CSV."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.data import parse_region_map


def make_fake_csv(data_dir, test='GF', target_col='p20016_i2', n=300):
    rng = np.random.default_rng(0)
    cols = {}
    for fc in ('FC21', 'FC55'):
        _, _, brain_cols = parse_region_map(os.path.join(ROOT, 'region_names', f'{fc}_regions.txt'))
        for c in brain_cols:
            cols[c] = rng.normal(0, 0.3, n)
    df = pd.DataFrame(cols)
    df.insert(0, 'eid', np.arange(1000000, 1000000 + n))
    df['p31'] = rng.integers(0, 2, n)
    df['p21003_i2'] = rng.normal(64, 7, n)
    df['p54_i2'] = rng.choice([11025, 11026, 11027, 11028], n)
    df['p25741_i2'] = rng.gamma(2, 0.05, n)
    df[target_col] = 6 + 0.5 * (df['p21003_i2'] - 64) / 7 + rng.normal(0, 1, n)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, f'combined_data_{test}_no_outliers.csv'), index=False)


def run_sweep(tmp_path, *extra):
    cmd = [sys.executable, os.path.join(ROOT, 'topk', 'run_sweep.py'),
           '--config', os.path.join(ROOT, 'configs', 'topk.yaml'),
           '--tests', 'GF', '--keep', '0.0', '0.1', '1.0', '--seeds', '0', '1', '--max-epochs', '3',
           '--set', f'paths.data_dir={tmp_path / "data"}', f'paths.output_dir={tmp_path / "out"}',
           f'paths.region_dir={os.path.join(ROOT, "region_names")}', 'split.use_eid_file=false',
           'training.batch_size=64', *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_sweep_end_to_end(tmp_path):
    make_fake_csv(tmp_path / 'data')
    proc = run_sweep(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    out = tmp_path / 'out'
    results = pd.read_csv(out / 'results.csv')
    # Per FC and seed: keep 0.0 personal, keep 0.1 personal + shared, keep 1.0 personal (shared skipped at 0 and 1)
    assert len(results) == 2 * 2 * 4
    assert set(results['fc']) == {'FC21', 'FC55'}
    assert set(results['mask']) == {'personal', 'shared'}
    assert set(results['seed']) == {0, 1}
    assert (results['split_seed'] == results['seed']).all()
    assert set(results.loc[results['mask'] == 'shared', 'keep_frac']) == {0.1}
    assert not (out / 'failures.csv').exists()
    assert (results[['n_train', 'n_val', 'n_test']].sum(axis=1) == 300).all()
    assert (results.loc[results['keep_frac'] == 0.0, 'k_edges'] == 0).all()

    run_dir = out / 'FC55' / 'GF' / 'personal' / 'keep010' / 'seed0'
    for f in ('metrics.json', 'config.yaml', 'losses.csv', 'test_predictions.csv', 'model.pt', 'edge_retention.npy'):
        assert (run_dir / f).exists(), f
    retention = np.load(run_dir / 'edge_retention.npy')
    assert retention.shape == (1485,)
    assert np.isclose(retention.sum(), 148)  # k = floor(0.1 * 1485) edges kept per subject
    assert np.load(out / 'FC55' / 'GF' / 'personal' / 'keep000' / 'seed0' / 'edge_retention.npy').sum() == 0
    assert not (out / 'FC55' / 'GF' / 'shared' / 'keep000').exists()
    assert not (out / 'FC55' / 'GF' / 'shared' / 'keep100').exists()

    shared_dir = out / 'FC55' / 'GF' / 'shared' / 'keep010' / 'seed0'
    shared = np.load(shared_dir / 'shared_edges.npy')
    assert shared.shape == (148,)
    retention = np.load(shared_dir / 'edge_retention.npy')
    assert set(np.unique(retention)) == {0.0, 1.0} and retention.sum() == 148
    assert np.array_equal(np.flatnonzero(retention), np.sort(shared))
    assert not (run_dir / 'shared_edges.npy').exists()

    # Each seed gets its own split, saved under splits/seed{s}/, and preprocessors per seed
    for seed in (0, 1):
        for split in ('train', 'val', 'test'):
            assert (out / 'splits' / f'seed{seed}' / f'GF_{split}_eids.txt').exists()
        assert (out / 'FC21' / 'GF' / f'preprocessors_seed{seed}.pkl').exists()
    test0 = np.loadtxt(out / 'splits' / 'seed0' / 'GF_test_eids.txt')
    test1 = np.loadtxt(out / 'splits' / 'seed1' / 'GF_test_eids.txt')
    assert len(test0) == len(test1) and not np.array_equal(np.sort(test0), np.sort(test1))

    # Rerun skips completed runs
    proc = run_sweep(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'All runs complete' in proc.stdout
    assert len(pd.read_csv(out / 'results.csv')) == 16

    # summarize.py runs on the real output and writes both figures
    proc = subprocess.run([sys.executable, os.path.join(ROOT, 'topk', 'summarize.py'),
                           '--results', str(out / 'results.csv'), '--out-dir', str(out / 'summary')],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for f in ('test_r2_by_keep.csv', 'test_r2_vs_keep.png', 'test_r2_delta_vs_full.csv', 'test_r2_delta_vs_full.png'):
        assert (out / 'summary' / f).exists(), f
