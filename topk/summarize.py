"""Aggregate {paths.output_dir}/results.csv into a table and R²-vs-keep% plots."""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize sparsity sweep results")
    parser.add_argument('--config', default='configs/topk.yaml')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='Config overrides')
    parser.add_argument('--results', help='Default: {paths.output_dir}/results.csv')
    parser.add_argument('--out-dir', help='Default: {paths.output_dir}/summary')
    parser.add_argument('--metric', default='test_r2', help='Column to plot, e.g. test_r2 or test_r2_corr')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.results is None or args.out_dir is None:
        output_dir = load_config(args.config, args.set)['paths']['output_dir']
        args.results = args.results or os.path.join(output_dir, 'results.csv')
        args.out_dir = args.out_dir or os.path.join(output_dir, 'summary')
    df = pd.read_csv(args.results)
    if 'mask' not in df:
        df['mask'] = 'personal'   # results written before shared masks existed
    df['mask'] = df['mask'].fillna('personal')
    # Keep the latest row per run in case of reruns with --overwrite
    df = df.drop_duplicates(subset=['fc', 'test', 'mask', 'keep_frac', 'seed'], keep='last')
    os.makedirs(args.out_dir, exist_ok=True)

    agg = (df.groupby(['test', 'fc', 'mask', 'keep_frac'])[args.metric]
             .agg(['mean', 'std', 'count']).reset_index())
    agg.to_csv(os.path.join(args.out_dir, f'{args.metric}_by_keep.csv'), index=False)

    table = agg.pivot_table(index=['test', 'fc', 'mask'], columns='keep_frac', values='mean')
    print(table.round(4).to_string())

    tests = list(dict.fromkeys(agg['test']))
    ncols = 2
    nrows = (len(tests) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
    colors = {fc: f'C{i}' for i, fc in enumerate(sorted(agg['fc'].unique()))}
    linestyles = {'personal': '-', 'shared': '--'}
    for ax, test in zip(axes.flat, tests):
        for (fc, mask), sub in agg[agg['test'] == test].groupby(['fc', 'mask']):
            sub = sub.sort_values('keep_frac')
            x = sub['keep_frac'] * 100
            ax.plot(x, sub['mean'], marker='o', color=colors[fc], linestyle=linestyles.get(mask, ':'),
                    label=f'{fc} {mask}')
            if (sub['count'] > 1).any():
                ax.fill_between(x, sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                                color=colors[fc], alpha=0.2)
        ax.set_title(test)
        ax.set_xlabel('Edges kept (%)')
        ax.set_ylabel(args.metric)
        ax.legend()
    for ax in list(axes.flat)[len(tests):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, f'{args.metric}_vs_keep.png')
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")


if __name__ == '__main__':
    main()
