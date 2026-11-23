"""Shared data setup for every RL stage: seed-0 split, A/B halves of train, train-fitted preprocessing, GPU tensors.

The split is the same seed-0 70/15/15 split as the top-K Optuna sweep (common.data.make_split). Train is split
again into half A ('train_pred', trains the reward predictor) and half B ('train_agent', trains the PPO agent).
Preprocessors (demographics, target, edge z-scores) are fit on the whole train split and shared by all stages.
Every stage recomputes this deterministically from the config, so stages can run as separate jobs.
"""
import argparse
import os
from dataclasses import dataclass

import joblib
import numpy as np
import torch

from common.config import load_config
from common.data import SPLITS, extract_arrays, load_subject_table, make_split, prepare_splits
from rl.masking import SubjectTensors

RL_SPLITS = SPLITS + ('train_pred', 'train_agent')
STAGES = ('predictor', 'agent', 'final')


@dataclass
class RLData:
    sets: dict              # split name -> SubjectTensors (train, val, test, train_pred, train_agent)
    edge_list: torch.Tensor  # (E, 2) long, on the device
    node_names: list
    num_nodes: int
    preprocessors: dict      # 'demo', 'y', 'edge' (fit on train)
    halves: dict             # 'train_pred' / 'train_agent' -> positions within the train split

    @property
    def num_edges(self):
        return self.edge_list.shape[0]

    @property
    def demo_dim(self):
        return self.sets['train'].demo.shape[1]


def base_parser(description):
    """CLI arguments shared by the rl/ scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default='configs/rl.yaml')
    parser.add_argument('--fc', nargs='+', help='FC types (default: sweep.fc)')
    parser.add_argument('--tests', nargs='+', help='Cognitive tests (default: sweep.tests)')
    parser.add_argument('--max-subjects', type=int, help='Subsample subjects (smoke tests)')
    parser.add_argument('--overwrite', action='store_true', help='Redo this stage even if it finished')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='Config overrides')
    return parser


def load_cli_config(args):
    cfg = load_config(args.config, args.set)
    if cfg['graph']['edge_value'] != 'standardized':
        raise ValueError("The RL stage expects graph.edge_value: standardized")
    if cfg['model'].get('type') != 'gatv2':
        raise ValueError("The RL stage uses the GATv2 predictor (model.type: gatv2)")
    fc_types = args.fc or cfg['sweep'].get('fc') or list(cfg['fc_types'])
    tests = args.tests or cfg['sweep'].get('tests') or list(cfg['tests'])
    return cfg, fc_types, tests


def stage_dir(cfg, fc_type, test, stage=None):
    path = os.path.join(cfg['paths']['output_dir'], fc_type, test)
    return path if stage is None else os.path.join(path, stage)


def pipeline_stage_dirs(cfg, fc_type, test):
    """{stage: dir} for the three trained stages, in pipeline order (used for timing)."""
    return {stage: stage_dir(cfg, fc_type, test, stage) for stage in ('predictor', 'agent', 'final')}


def split_halves(n_train, agent_frac, seed):
    """Positions (within the train split) of half A (predictor) and half B (agent)."""
    perm = np.random.default_rng(seed).permutation(n_train)
    n_agent = int(round(agent_frac * n_train))
    return {'train_agent': np.sort(perm[:n_agent]), 'train_pred': np.sort(perm[n_agent:])}


def load_rl_data(cfg, test, fc_type, device, max_subjects=None, save=True):
    rl_cfg = cfg['rl']
    df = load_subject_table(test, cfg, max_subjects)
    eids = df.index.to_numpy()
    split_idx = make_split(eids, cfg, seed=rl_cfg['seed'])
    arrays = extract_arrays(df, test, fc_type, cfg)
    prepared, preprocessors = prepare_splits(arrays, split_idx, cfg)
    halves = split_halves(len(split_idx['train']), rl_cfg['agent_frac'], rl_cfg['seed'])

    sets = {name: SubjectTensors.from_prepared(prepared[name], device) for name in SPLITS}
    for name, pos in halves.items():
        sets[name] = sets['train'].subset(torch.as_tensor(pos, device=device))

    if save:
        out = stage_dir(cfg, fc_type, test)
        os.makedirs(os.path.join(out, 'splits'), exist_ok=True)
        for name in RL_SPLITS:
            np.savetxt(os.path.join(out, 'splits', f'{test}_{name}_eids.txt'), sets[name].eid.cpu().numpy(), fmt='%d')
        joblib.dump(preprocessors, os.path.join(out, 'preprocessors.pkl'))

    print(f"[{fc_type}|{test}] " + ", ".join(f"{name}: {len(sets[name])}" for name in RL_SPLITS))
    return RLData(sets=sets, edge_list=torch.as_tensor(arrays['edge_list'], dtype=torch.long, device=device),
                  node_names=arrays['node_names'], num_nodes=cfg['fc_types'][fc_type]['num_nodes'],
                  preprocessors=preprocessors, halves=halves)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
