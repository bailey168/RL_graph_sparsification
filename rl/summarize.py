"""Stage 4: summarize the RL runs and compare them with the existing top-K (GATv2 Optuna) results.

For every {FC}/{TEST} with a finished final GNN, writes to {output_dir}/summary/:
  rl_results.csv                       one row per FC × test (final-GNN and frozen-predictor metrics, sparsity)
  rl_vs_topk_{FC}_{TEST}.png           top-K test R² vs keep % (read from paths.topk_results, not retrained) with the
                                       RL final GNN at its mean keep % (± SD across subjects)
  keep_hist_{FC}_{TEST}.png            distribution of the agent's chosen keep % on test subjects
  r2_by_keep_{FC}_{TEST}.csv           final-GNN test metrics grouped by the subject's chosen keep level
  edge_retention_{FC}_{TEST}.png       fraction of test subjects keeping each edge, nodes ordered by network
  network_retention_{FC}_{TEST}.csv    mean retention within/between networks
  edge_ranking_{FC}_{TEST}.csv         edges by retention, with node/network names and mean removal step

Example:
    python rl/summarize.py --config configs/rl.yaml
"""
import argparse
import json
import os
import sys
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load_config
from common.data import parse_region_map
from rl.masking import removal_schedule


def load_networks(cfg, fc_type, node_names):
    """Network label per node from region_names/{FC}_network_groupings.json ('Other' if unlisted)."""
    path = os.path.join(cfg['paths']['region_dir'], f'{fc_type}_network_groupings.json')
    labels = {n: 'Other' for n in node_names}
    order = []
    if os.path.exists(path):
        with open(path) as f:
            groups = json.load(f)
        for net, spec in groups.items():
            order.append(net)
            for node in spec['nodes']:
                if node in labels:
                    labels[node] = net
    if 'Other' in labels.values():
        order.append('Other')
    return [labels[n] for n in node_names], order


def plot_vs_topk(cfg, fc_type, test, row, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    topk_path = cfg['paths'].get('topk_results')
    if topk_path and os.path.exists(topk_path):
        topk = pd.read_csv(topk_path)
        topk = topk[(topk['fc'] == fc_type) & (topk['test'] == test)]
        if 'mask' not in topk:
            topk['mask'] = 'personal'
        for mask, style in (('personal', '-'), ('shared', '--')):
            sub = topk[topk['mask'] == mask].groupby('keep_frac')['test_r2'].mean().reset_index()
            if len(sub):
                ax.plot(sub['keep_frac'] * 100, sub['test_r2'], style, marker='o', color='grey',
                        label=f'top-K {mask}')
    x, xerr = row['test_keep_frac_mean'] * 100, row['test_keep_frac_sd'] * 100
    ax.errorbar([x], [row['test_r2']], xerr=[xerr], fmt='*', ms=14, color='crimson', capsize=4,
                label='RL agent + final GATv2')
    ax.errorbar([x], [row['predictor_test_r2']], xerr=[xerr], fmt='o', mfc='none', color='crimson', capsize=4,
                label='RL agent + frozen reward predictor')
    ax.set_xlabel('Edges kept (%)  [RL: mean ± SD across test subjects]')
    ax.set_ylabel('Test R²')
    ax.set_title(f'{fc_type} | {test}')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def keep_levels(stop_step, num_edges, n_steps):
    schedule = np.array(removal_schedule(num_edges, n_steps))
    return schedule[stop_step] / num_edges


def plot_keep_hist(keep, fc_type, test, n_steps, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 100, n_steps + 2) - 100 / (2 * n_steps)
    ax.hist(keep * 100, bins=bins, edgecolor='white')
    ax.set_xlabel("Edges kept at the agent's STOP (%)")
    ax.set_ylabel('Test subjects')
    ax.set_title(f'{fc_type} | {test}: personalised sparsity')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def r2_by_keep(final_dir, keep, eids):
    preds = pd.read_csv(os.path.join(final_dir, 'test_predictions.csv'))
    df = preds.merge(pd.DataFrame({'eid': eids, 'keep_frac': keep}), on='eid')
    rows = []
    for k, g in df.groupby('keep_frac'):
        rows.append({'keep_frac': k, 'n': len(g),
                     'r2': r2_score(g['actual'], g['predicted']) if len(g) > 1 else np.nan,
                     'rmse': float(np.sqrt(mean_squared_error(g['actual'], g['predicted'])))})
    return pd.DataFrame(rows)


def retention_outputs(cfg, fc_type, test, final_dir, out_dir):
    region_file = os.path.join(cfg['paths']['region_dir'], cfg['fc_types'][fc_type]['regions_file'])
    node_names, edge_list, _ = parse_region_map(region_file)
    masks = np.load(os.path.join(final_dir, 'stopped_masks_test.npy'))
    removal = np.load(os.path.join(final_dir, 'removal_step_test.npy')).astype(float)
    retention = masks.mean(0)
    removal[removal < 0] = np.nan
    with np.errstate(invalid='ignore'):
        mean_removal = np.nanmean(removal, axis=0) if np.isfinite(removal).any() else np.full(len(edge_list), np.nan)
    networks, net_order = load_networks(cfg, fc_type, node_names)

    edges = pd.DataFrame({'edge': np.arange(len(edge_list)),
                          'node_i': [node_names[i] for i in edge_list[:, 0]],
                          'node_j': [node_names[j] for j in edge_list[:, 1]],
                          'network_i': [networks[i] for i in edge_list[:, 0]],
                          'network_j': [networks[j] for j in edge_list[:, 1]],
                          'retention': retention, 'mean_removal_step': mean_removal})
    edges.sort_values('retention', ascending=False).to_csv(
        os.path.join(out_dir, f'edge_ranking_{fc_type}_{test}.csv'), index=False)

    pair = edges.assign(a=[min(a, b, key=net_order.index) for a, b in zip(edges['network_i'], edges['network_j'])],
                        b=[max(a, b, key=net_order.index) for a, b in zip(edges['network_i'], edges['network_j'])])
    table = pair.groupby(['a', 'b'])['retention'].mean().unstack()
    table = table.reindex(index=[n for n in net_order if n in table.index],
                          columns=[n for n in net_order if n in table.columns])
    table.to_csv(os.path.join(out_dir, f'network_retention_{fc_type}_{test}.csv'))

    N = len(node_names)
    mat = np.full((N, N), np.nan)
    mat[edge_list[:, 0], edge_list[:, 1]] = retention
    mat[edge_list[:, 1], edge_list[:, 0]] = retention
    order = sorted(range(N), key=lambda i: (net_order.index(networks[i]), i))
    mat = mat[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
    ax.grid(False)
    labels = [networks[i] for i in order]
    bounds = [i for i in range(1, N) if labels[i] != labels[i - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color='white', lw=0.8)
        ax.axvline(b - 0.5, color='white', lw=0.8)
    starts = [0] + bounds
    ends = bounds + [N]
    ax.set_xticks([(s + e - 1) / 2 for s, e in zip(starts, ends)])
    ax.set_xticklabels([labels[s] for s in starts], rotation=90)
    ax.set_yticks([(s + e - 1) / 2 for s, e in zip(starts, ends)])
    ax.set_yticklabels([labels[s] for s in starts])
    fig.colorbar(im, ax=ax, label='Fraction of test subjects keeping the edge')
    ax.set_title(f'{fc_type} | {test}: edge retention at STOP')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'edge_retention_{fc_type}_{test}.png'), dpi=150)
    plt.close(fig)


def summarize(cfg):
    output_dir = cfg['paths']['output_dir']
    out_dir = os.path.join(output_dir, 'summary')
    os.makedirs(out_dir, exist_ok=True)
    n_steps = cfg['rl']['n_steps']
    rows = []
    for fc_type in cfg['fc_types']:
        for test in cfg['tests']:
            final_dir = os.path.join(output_dir, fc_type, test, 'final')
            metrics_path = os.path.join(final_dir, 'metrics.json')
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as f:
                row = json.load(f)
            rows.append(row)
            stop_step = np.load(os.path.join(final_dir, 'stop_step_test.npy')).astype(int)
            eids = np.load(os.path.join(final_dir, 'eids_test.npy'))
            num_edges = np.load(os.path.join(final_dir, 'stopped_masks_test.npy'), mmap_mode='r').shape[1]
            keep = keep_levels(stop_step, num_edges, n_steps)
            plot_vs_topk(cfg, fc_type, test, row, os.path.join(out_dir, f'rl_vs_topk_{fc_type}_{test}.png'))
            plot_keep_hist(keep, fc_type, test, n_steps, os.path.join(out_dir, f'keep_hist_{fc_type}_{test}.png'))
            r2_by_keep(final_dir, keep, eids).to_csv(os.path.join(out_dir, f'r2_by_keep_{fc_type}_{test}.csv'),
                                                     index=False)
            retention_outputs(cfg, fc_type, test, final_dir, out_dir)
    if not rows:
        print(f"No finished final GNNs under {output_dir}")
        return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'rl_results.csv'), index=False)
    cols = ['fc', 'test', 'test_r2', 'predictor_test_r2', 'predictor_test_r2_full', 'test_keep_frac_mean',
            'test_keep_frac_sd']
    print(df[[c for c in cols if c in df]].to_string(index=False))
    print(f"Summary written to {out_dir}")
    return df


def auto_summarize(cfg):
    """Run summarize() at the end of a training script; never raises."""
    try:
        with plt.style.context('default'):   # Optuna's matplotlib plots switch the global style to ggplot
            summarize(cfg)
    except Exception:   # noqa: BLE001
        traceback.print_exc()
        print("Summary failed; rerun python rl/summarize.py by hand")


def main():
    parser = argparse.ArgumentParser(description="Summarize RL sparsification runs")
    parser.add_argument('--config', default='configs/rl.yaml')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='Config overrides')
    args = parser.parse_args()
    summarize(load_config(args.config, args.set))


if __name__ == '__main__':
    main()
