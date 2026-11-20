"""Aggregate {paths.output_dir}/results.csv into tables and R²-vs-keep% plots.

Outputs in {paths.output_dir}/summary/:
  {metric}_by_keep.csv / {metric}_vs_keep.png               mean ± SD across seeds
  {metric}_delta_vs_full.csv / {metric}_delta_vs_full.png   per-seed difference from personal keep=1.0
                                                             (same split), mean ± SD across seeds
For Optuna runs (topk/run_optuna.py; one seed, so no error bars), also:
  best_params_vs_keep_{TEST}.png                            best hyperparameters per keep level
  param_importance_by_keep.csv / param_importance_by_keep_{FC}_{mask}.png
                                                             fANOVA importance, mean over tests
"""
import argparse
import glob
import json
import os
import sys
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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


LINESTYLES = {'personal': '-', 'shared': '--'}   # other masks: ':'


def fc_colors(df):
    return {fc: f'C{i}' for i, fc in enumerate(sorted(df['fc'].unique()))}


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_panels(agg, ylabel, fig_path, zero_line=False):
    """One panel per test; mean across seeds with ±1 SD error bars. Colour = FC, line style = mask."""
    tests = list(dict.fromkeys(agg['test']))
    ncols = 2
    nrows = (len(tests) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
    colors = fc_colors(agg)
    offsets = {'personal': -0.6, 'shared': 0.6}   # small x-shift so overlapping error bars stay readable
    for ax, test in zip(axes.flat, tests):
        for (fc, mask), sub in agg[agg['test'] == test].groupby(['fc', 'mask']):
            sub = sub.sort_values('keep_frac')
            x = sub['keep_frac'] * 100 + offsets.get(mask, 0.0)
            yerr = sub['std'].fillna(0.0)
            ax.errorbar(x, sub['mean'], yerr=yerr, marker='o', markersize=4, capsize=3, color=colors[fc],
                        linestyle=LINESTYLES.get(mask, ':'), label=f'{fc} {mask}')
        if zero_line:
            ax.axhline(0.0, color='grey', linestyle=':', linewidth=1)
        ax.set_title(test)
        ax.set_xlabel('Edges kept (%)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(tests):]:
        ax.set_visible(False)
    save_fig(fig, fig_path)


STUDY_KEYS = ['fc', 'test', 'mask', 'keep_frac']


def plot_best_params(params_path, out_dir):
    """One figure per test: a panel per tuned hyperparameter, best value vs keep%. Colour = FC, line style = mask."""
    df = pd.read_csv(params_path).drop_duplicates(subset=STUDY_KEYS, keep='last')
    params = [c for c in df.columns if c not in STUDY_KEYS + ['best_trial', 'best_val_loss', 'test_r2']]
    colors = fc_colors(df)
    for test, tdf in df.groupby('test'):
        ncols = 4
        nrows = int(np.ceil(len(params) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for ax, param in zip(axes.flat, params):
            numeric = pd.api.types.is_numeric_dtype(df[param])
            if not numeric:   # categorical (e.g. pooling, classifier dims '64-32'): shared category axis
                cats = sorted(df[param].astype(str).unique())
                ax.set_yticks(range(len(cats)), cats, fontsize=7)
            for (fc, mask), sub in tdf.groupby(['fc', 'mask']):
                sub = sub.sort_values('keep_frac')
                y = sub[param] if numeric else sub[param].astype(str).map(cats.index)
                ax.plot(sub['keep_frac'] * 100, y, marker='o', markersize=3, color=colors[fc],
                        linestyle=LINESTYLES.get(mask, ':'), label=f'{fc} {mask}')
            if numeric and df[param].min() > 0 and df[param].max() / df[param].min() > 20:
                ax.set_yscale('log')
            ax.set_title(param, fontsize=9)
            ax.set_xlabel('Edges kept (%)', fontsize=8)
        for ax in list(axes.flat)[len(params):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{test}: best hyperparameters per sparsity level')
        save_fig(fig, os.path.join(out_dir, f'best_params_vs_keep_{test}.png'))


def collect_importances(output_dir):
    """Long table of per-study fANOVA importances (param_importances.json + metrics.json in each study dir)."""
    rows = []
    for path in glob.glob(os.path.join(output_dir, '*', '*', '*', 'keep*', 'param_importances.json')):
        meta_path = os.path.join(os.path.dirname(path), 'metrics.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        with open(path) as f:
            for param, imp in json.load(f).items():
                rows.append({**{k: meta[k] for k in STUDY_KEYS}, 'param': param, 'importance': imp})
    return pd.DataFrame(rows)


def plot_importances(imp, out_dir):
    """Heatmap per (FC, mask): hyperparameter x keep%, importance averaged over tests."""
    imp.to_csv(os.path.join(out_dir, 'param_importance_by_keep.csv'), index=False)
    for (fc, mask), sub in imp.groupby(['fc', 'mask']):
        table = sub.pivot_table(index='param', columns='keep_frac', values='importance', aggfunc='mean').fillna(0)
        table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(1 + 0.7 * table.shape[1], 1 + 0.4 * table.shape[0]))
        im = ax.imshow(table.values, aspect='auto', cmap='viridis', vmin=0)
        ax.set_xticks(range(table.shape[1]), [f'{k * 100:.0f}' for k in table.columns])
        ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
        ax.set_xlabel('Edges kept (%)')
        ax.set_title(f'{fc} {mask}: fANOVA importance (mean over tests)', fontsize=10)
        fig.colorbar(im, ax=ax)
        save_fig(fig, os.path.join(out_dir, f'param_importance_by_keep_{fc}_{mask}.png'))


def optuna_summaries(results_path, out_dir):
    """Extra figures for Optuna runs, when their files sit next to results.csv."""
    output_dir = os.path.dirname(os.path.abspath(results_path))
    params_path = os.path.join(output_dir, 'best_params.csv')
    if os.path.exists(params_path):
        plot_best_params(params_path, out_dir)
    imp = collect_importances(output_dir)
    if not imp.empty:
        plot_importances(imp, out_dir)


def summarize(results_path, out_dir, metric='test_r2'):
    """Write all summary tables and figures for one results.csv into out_dir."""
    df = load_results(results_path)
    os.makedirs(out_dir, exist_ok=True)

    agg = aggregate(df, metric)
    agg.to_csv(os.path.join(out_dir, f'{metric}_by_keep.csv'), index=False)
    warn_missing_seeds(agg)
    spread = '(mean ± SD over seeds)' if agg['count'].max() > 1 else '(single seed)'
    print(agg.pivot_table(index=['test', 'fc', 'mask'], columns='keep_frac', values='mean').round(4).to_string())
    plot_panels(agg, f'{metric} {spread}', os.path.join(out_dir, f'{metric}_vs_keep.png'))

    optuna_summaries(results_path, out_dir)

    delta = paired_delta(df, metric)
    if delta.empty:
        print("No personal keep=1.0 runs found; skipping the paired delta analysis")
        return
    delta_agg = aggregate(delta, 'delta')
    delta_agg.to_csv(os.path.join(out_dir, f'{metric}_delta_vs_full.csv'), index=False)
    print("\nPaired delta vs 100% kept (mean over seeds):")
    print(delta_agg.pivot_table(index=['test', 'fc', 'mask'], columns='keep_frac', values='mean')
          .round(4).to_string())
    plot_panels(delta_agg, f'Δ {metric} vs 100% kept {spread}',
                os.path.join(out_dir, f'{metric}_delta_vs_full.png'), zero_line=True)


def auto_summarize(output_dir):
    """Called at the end of run_sweep.py / run_optuna.py. Never raises: training results matter more than plots."""
    results_path = os.path.join(output_dir, 'results.csv')
    if not os.path.exists(results_path):
        print(f"No {results_path}; skipping summary")
        return
    print(f"\n=== Summarizing {results_path} ===")
    try:
        summarize(results_path, os.path.join(output_dir, 'summary'))
    except Exception:   # noqa: BLE001
        traceback.print_exc()
        print("Summary failed; rerun topk/summarize.py by hand")


def main():
    args = parse_args()
    if args.results is None or args.out_dir is None:
        output_dir = load_config(args.config, args.set)['paths']['output_dir']
        args.results = args.results or os.path.join(output_dir, 'results.csv')
        args.out_dir = args.out_dir or os.path.join(output_dir, 'summary')
    summarize(args.results, args.out_dir, args.metric)


if __name__ == '__main__':
    main()
