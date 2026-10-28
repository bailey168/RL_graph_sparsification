import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SPLITS = ('train', 'val', 'test')


def parse_region_map(region_file_path):
    """Parse a region .txt file (adapted from others/process_graph_pipeline/csv_to_graph.py).

    Returns:
     - node_names: unique regions in order of first appearance
     - edge_list: (E, 2) int array of (src, dst) node indices, in file order
     - columns: the exact CSV column names, in file order
    """
    with open(region_file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # e.g. "FC25 (IC1-IC2)" or "lh_bankssts-lh_caudalanteriorcingulate"
    raw_pairs = []
    for line in lines:
        if line.startswith("FC"):
            inner = line[line.find("(") + 1:line.find(")")]
            src, dst = inner.split("-")
        else:
            src, dst = line.split("-")
        raw_pairs.append((src, dst))

    node_names = []
    for src, dst in raw_pairs:
        if src not in node_names: node_names.append(src)
        if dst not in node_names: node_names.append(dst)

    node_to_idx = {name: idx for idx, name in enumerate(node_names)}
    edge_list = np.array([(node_to_idx[s], node_to_idx[d]) for s, d in raw_pairs], dtype=np.int64)
    return node_names, edge_list, lines


def load_subject_table(test, cfg, max_subjects=None):
    """Load the combined CSV for a cognitive test, indexed by eid, optionally filtered to the EID list."""
    paths = cfg['paths']
    csv_path = os.path.join(paths['data_dir'], f'combined_data_{test}_no_outliers.csv')
    df = pd.read_csv(csv_path)
    if 'eid' in df.columns:
        df = df.set_index('eid')
    print(f"[{test}] Loaded {len(df)} rows from {csv_path}")

    if cfg['split']['use_eid_file']:
        eid_file = os.path.join(paths['eid_dir'], test, f'{test}_all_eids.txt')
        if os.path.exists(eid_file):
            keep_eids = np.loadtxt(eid_file, dtype=int)
            df = df[df.index.isin(keep_eids)]
            print(f"[{test}] Filtered to {len(df)} rows using {eid_file}")
        else:
            print(f"[{test}] Warning: EID file not found ({eid_file}); using all rows")

    if max_subjects is not None and max_subjects < len(df):
        df = df.sample(n=max_subjects, random_state=cfg['split']['seed'])
        print(f"[{test}] Subsampled to {len(df)} rows (--max-subjects)")

    return df


def extract_arrays(df, test, fc_type, cfg):
    """Pull edge, demographic and target arrays for one FC type out of the subject table."""
    region_file = os.path.join(cfg['paths']['region_dir'], cfg['fc_types'][fc_type]['regions_file'])
    node_names, edge_list, brain_cols = parse_region_map(region_file)
    expected_nodes = cfg['fc_types'][fc_type]['num_nodes']
    if len(node_names) != expected_nodes:
        raise ValueError(f"{fc_type}: parsed {len(node_names)} nodes, expected {expected_nodes}")

    demo_cfg = cfg['demographics']
    demo_cols = demo_cfg['numeric'] + demo_cfg['categorical']
    target_col = cfg['tests'][test]

    missing = [c for c in brain_cols + demo_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} expected columns in CSV (e.g., {missing[:3]})")

    X_raw = df[brain_cols]
    if X_raw.isna().any().any():
        raise ValueError("Missing data (NaNs) detected in the brain edge features.")
    X_demo = df[demo_cols]
    if X_demo.isna().any().any():
        raise ValueError("Missing data (NaNs) detected in the demographic features.")
    y = df[target_col]
    if y.isna().any():
        raise ValueError(f"Missing data (NaNs) detected in the target variable '{target_col}'.")

    return {
        'X_raw': X_raw.to_numpy(dtype=np.float32),
        'X_demo': X_demo,
        'y': y.to_numpy(dtype=np.float64),
        'eids': df.index.to_numpy(),
        'node_names': node_names,
        'edge_list': edge_list,
    }


def make_split(eids, cfg):
    """70/15/15 split (as in the notebook). Depends only on the subject list, so FC21/FC55 share it."""
    split_cfg = cfg['split']
    holdout = split_cfg['val_frac'] + split_cfg['test_frac']
    train_idx, temp_idx = train_test_split(
        np.arange(len(eids)), test_size=holdout, random_state=split_cfg['seed']
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=split_cfg['test_frac'] / holdout, random_state=split_cfg['seed']
    )
    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


def save_split(eids, split_idx, out_dir, test):
    os.makedirs(out_dir, exist_ok=True)
    for name in SPLITS:
        np.savetxt(os.path.join(out_dir, f'{test}_{name}_eids.txt'), eids[split_idx[name]], fmt='%d')


def prepare_splits(arrays, split_idx, cfg):
    """Fit preprocessors on train only, and return per-split arrays ready for graph building.

    Per split: 'rank_scores' (N, E) used for top-K, 'edge_vals' (N, E) signed values put on edges,
    'demo' (N, D), 'y' (N,) scaled target, 'eids' (N,).
    """
    demo_cfg = cfg['demographics']
    graph_cfg = cfg['graph']
    tr = split_idx['train']

    demo_pre = ColumnTransformer(transformers=[
        ('num', StandardScaler(), demo_cfg['numeric']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), demo_cfg['categorical']),
    ])
    demo_pre.fit(arrays['X_demo'].iloc[tr])

    y_pre = StandardScaler()
    y_pre.fit(arrays['y'][tr].reshape(-1, 1))

    preprocessors = {'demo': demo_pre, 'y': y_pre}

    needs_std = 'standardized' in (graph_cfg['rank_by'], graph_cfg['edge_value'])
    if needs_std:
        edge_pre = StandardScaler()
        edge_pre.fit(arrays['X_raw'][tr])
        preprocessors['edge'] = edge_pre

    out = {}
    for name in SPLITS:
        idx = split_idx[name]
        X_raw = arrays['X_raw'][idx]
        X_std = preprocessors['edge'].transform(X_raw).astype(np.float32) if needs_std else None
        out[name] = {
            'rank_scores': np.abs(X_std if graph_cfg['rank_by'] == 'standardized' else X_raw),
            'edge_vals': X_std if graph_cfg['edge_value'] == 'standardized' else X_raw,
            'demo': demo_pre.transform(arrays['X_demo'].iloc[idx]).astype(np.float32),
            'y': y_pre.transform(arrays['y'][idx].reshape(-1, 1)).flatten().astype(np.float32),
            'eids': arrays['eids'][idx],
        }
    return out, preprocessors
