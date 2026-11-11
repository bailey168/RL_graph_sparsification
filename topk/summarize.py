"""Aggregate {paths.output_dir}/results.csv into tables and R²-vs-keep% plots.

Outputs in {paths.output_dir}/summary/:
  {metric}_by_keep.csv / {metric}_vs_keep.png               mean ± SD across seeds
  {metric}_delta_vs_full.csv / {metric}_delta_vs_full.png   per-seed difference from personal keep=1.0
                                                             (same split), mean ± SD across seeds
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load_config
from topk.sparsify import is_redundant


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize sparsity sweep results")
    parser.add_argument('--config', default='configs/topk.yaml')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='Config overrides')
    parser.add_argument('--results', help='Default: {paths.output_dir}/results.csv')
    parser.add_argument('--out-dir', help='Default: {paths.output_dir}/summary')
    parser.add_argument('--metric', default='test_r2', help='Column to plot, e.g. test_r2 or test_r2_corr')
    return parser.parse_args()


RUN_KEYS = ['fc', 'test', 'mask', 'keep_frac', 'seed']


def load_results(path):
    """Read results.csv, keep the latest row per run, and fill in the shared runs that are not trained.

    At keep 0.0 and 1.0 the shared mask gives the same graphs as the personal mask, so run_sweep.py
    only trains the personal one; its rows are copied here as mask='shared'.
    """
    df = pd.read_csv(path)
    if 'mask' not in df:
        df['mask'] = 'personal'   # results written before shared masks existed
    df['mask'] = df['mask'].fillna('personal')
    # Keep the latest row per run in case of reruns with --overwrite
    df = df.drop_duplicates(subset=RUN_KEYS, keep='last')
    if (df['mask'] == 'shared').any():
        ends = df[(df['mask'] == 'personal') & df['keep_frac'].map(lambda k: is_redundant('shared', k))]
        df = pd.concat([df, ends.assign(mask='shared')], ignore_index=True)
        df = df.drop_duplicates(subset=RUN_KEYS, keep='first')   # a real shared row wins over a copied one
    return df


def paired_delta(df, metric):
    """Per seed: metric minus the personal keep-1.0 run with the same (test, fc, seed), i.e. on the same split."""
    ref = (df[(df['mask'] == 'personal') & (df['keep_frac'] >= 1.0)]
           .set_index(['test', 'fc', 'seed'])[metric].rename('ref'))
    out = df.join(ref, on=['test', 'fc', 'seed'])
    out['delta'] = out[metric] - out['ref']
    return out.dropna(subset=['delta'])


def aggregate(df, column):
    return (df.groupby(['test', 'fc', 'mask', 'keep_frac'])[column]
              .agg(['mean', 'std', 'count']).reset_index())


def warn_missing_seeds(agg):
    expected = agg['count'].max()
    short = agg[agg['count'] < expected]
    if len(short):
        print(f"WARNING: {len(short)} configs have fewer than {expected} seeds:")
        print(short[['test', 'fc', 'mask', 'keep_frac', 'count']].to_string(index=False))


def plot_panels(agg, ylabel, fig_path, zero_line=False):
    """One panel per test; mean across seeds with ±1 SD error bars. Colour = FC, line style = mask."""
    tests = list(dict.fromkeys(agg['test']))
    ncols = 2
    nrows = (len(tests) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
    colors = {fc: f'C{i}' for i, fc in enumerate(sorted(agg['fc'].unique()))}
    linestyles = {'personal': '-', 'shared': '--'}
    offsets = {'personal': -0.6, 'shared': 0.6}   # small x-shift so overlapping error bars stay readable
    for ax, test in zip(axes.flat, tests):
        for (fc, mask), sub in agg[agg['test'] == test].groupby(['fc', 'mask']):
            sub = sub.sort_values('keep_frac')
            x = sub['keep_frac'] * 100 + offsets.get(mask, 0.0)
            yerr = sub['std'].fillna(0.0)
            ax.errorbar(x, sub['mean'], yerr=yerr, marker='o', markersize=4, capsize=3, color=colors[fc],
                        linestyle=linestyles.get(mask, ':'), label=f'{fc} {mask}')
        if zero_line:
            ax.axhline(0.0, color='grey', linestyle=':', linewidth=1)
        ax.set_title(test)
        ax.set_xlabel('Edges kept (%)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(tests):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path}")


def main():
    args = parse_args()
    if args.results is None or args.out_dir is None:
        output_dir = load_config(args.config, args.set)['paths']['output_dir']
        args.results = args.results or os.path.join(output_dir, 'results.csv')
        args.out_dir = args.out_dir or os.path.join(output_dir, 'summary')
    df = load_results(args.results)
    os.makedirs(args.out_dir, exist_ok=True)

    agg = aggregate(df, args.metric)
    agg.to_csv(os.path.join(args.out_dir, f'{args.metric}_by_keep.csv'), index=False)
    warn_missing_seeds(agg)
    print(agg.pivot_table(index=['test', 'fc', 'mask'], columns='keep_frac', values='mean').round(4).to_string())
    plot_panels(agg, f'{args.metric} (mean ± SD over seeds)',
                os.path.join(args.out_dir, f'{args.metric}_vs_keep.png'))

    delta = paired_delta(df, args.metric)
    if delta.empty:
        print("No personal keep=1.0 runs found; skipping the paired delta analysis")
        return
    delta_agg = aggregate(delta, 'delta')
    delta_agg.to_csv(os.path.join(args.out_dir, f'{args.metric}_delta_vs_full.csv'), index=False)
    print("\nPaired delta vs 100% kept (mean over seeds):")
    print(delta_agg.pivot_table(index=['test', 'fc', 'mask'], columns='keep_frac', values='mean')
          .round(4).to_string())
    plot_panels(delta_agg, f'Δ {args.metric} vs 100% kept (mean ± SD)',
                os.path.join(args.out_dir, f'{args.metric}_delta_vs_full.png'), zero_line=True)


if __name__ == '__main__':
    main()
