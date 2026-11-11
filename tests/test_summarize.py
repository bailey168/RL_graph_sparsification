"""summarize.py helpers on a synthetic results table."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topk.summarize import aggregate, load_results, paired_delta


def make_results(path, seeds=(0, 1, 2)):
    rng = np.random.default_rng(0)
    rows = []
    for seed in seeds:
        for fc in ('FC21', 'FC55'):
            for keep in (0.0, 0.5, 1.0):
                masks = ['personal', 'shared'] if keep == 0.5 else ['personal']   # shared skipped at 0 and 1
                for mask in masks:
                    rows.append({'fc': fc, 'test': 'GF', 'mask': mask, 'keep_frac': keep, 'seed': seed,
                                 'test_r2': 0.05 + 0.01 * keep + rng.normal(0, 0.002)})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_shared_rows_filled_and_delta(tmp_path):
    path = tmp_path / 'results.csv'
    make_results(path)
    df = load_results(path)

    # shared rows at 0 and 1 are copies of the personal ones
    for keep in (0.0, 1.0):
        p = df[(df['mask'] == 'personal') & (df['keep_frac'] == keep)].set_index(['fc', 'seed'])['test_r2']
        s = df[(df['mask'] == 'shared') & (df['keep_frac'] == keep)].set_index(['fc', 'seed'])['test_r2']
        assert len(s) == 6 and np.allclose(p.sort_index(), s.sort_index())

    agg = aggregate(df, 'test_r2')
    assert (agg['count'] == 3).all() and len(agg) == 2 * 2 * 3
    assert {'mean', 'std', 'count'} <= set(agg.columns)

    delta = paired_delta(df, 'test_r2')
    assert np.allclose(delta.loc[delta['keep_frac'] == 1.0, 'delta'], 0.0)
    d_agg = aggregate(delta, 'delta')
    assert np.allclose(d_agg.loc[d_agg['keep_frac'] == 1.0, ['mean', 'std']], 0.0)
    assert (d_agg.loc[d_agg['keep_frac'] == 0.0, 'mean'] < 0).all()
