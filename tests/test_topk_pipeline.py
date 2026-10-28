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
           '--tests', 'GF', '--keep', '0.1', '1.0', '--max-epochs', '3',
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
    assert len(results) == 4  # 2 FC types x 2 keep levels
    assert set(results['fc']) == {'FC21', 'FC55'}
    assert not (out / 'failures.csv').exists()
    assert (results[['n_train', 'n_val', 'n_test']].sum(axis=1) == 300).all()

    run_dir = out / 'FC55' / 'GF' / 'keep010' / 'seed42'
    for f in ('metrics.json', 'config.yaml', 'losses.csv', 'test_predictions.csv', 'model.pt', 'edge_retention.npy'):
        assert (run_dir / f).exists(), f
    retention = np.load(run_dir / 'edge_retention.npy')
    assert retention.shape == (1485,)
    assert np.isclose(retention.sum(), 148)  # k = floor(0.1 * 1485) edges kept per subject
    for split in ('train', 'val', 'test'):
        assert (out / 'splits' / f'GF_{split}_eids.txt').exists()

    # Rerun skips completed runs
    proc = run_sweep(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'All runs complete' in proc.stdout
    assert len(pd.read_csv(out / 'results.csv')) == 4
