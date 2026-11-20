"""common/tuning.py search-space sampling and common/config.py `base:` inheritance."""
import os

import optuna
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from common.config import load_config
from common.tuning import config_from_params, flatten_values, make_pruner, remaining_trials, suggest_config


def load_optuna_cfg():
    return load_config(os.path.join(ROOT, 'configs', 'topk_optuna.yaml'))


def test_base_config_inheritance(tmp_path):
    (tmp_path / 'parent.yaml').write_text(yaml.safe_dump({'a': {'x': 1, 'y': 2}, 'b': 3}))
    (tmp_path / 'child.yaml').write_text(yaml.safe_dump({'base': 'parent.yaml', 'a': {'y': 20}, 'c': 4}))
    cfg = load_config(tmp_path / 'child.yaml', ['a.x=10'])
    assert cfg == {'a': {'x': 10, 'y': 20}, 'b': 3, 'c': 4}

    # The real optuna config inherits topk.yaml and only changes the output dir
    cfg = load_optuna_cfg()
    base = load_config(os.path.join(ROOT, 'configs', 'topk.yaml'))
    assert cfg['model'] == base['model'] and cfg['training'] == base['training']
    assert cfg['sweep']['keep_fracs'] == base['sweep']['keep_fracs']                # merged, not replaced
    assert cfg['paths']['output_dir'] != base['paths']['output_dir']
    assert cfg['sweep']['fc'] == ['FC55'] and cfg['sweep']['tests'] == ['GF', 'DSST']
    assert cfg['sweep']['masks'] == ['personal', 'shared']
    assert base['model']['edge_dropout'] == 0.0
    assert 'base' not in cfg


def test_sampled_configs_are_valid():
    cfg = load_optuna_cfg()
    space = cfg['optuna']['search_space']
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    emb_equals_hidden = 0
    for _ in range(200):
        trial = study.ask()
        trial_cfg, values = suggest_config(trial, cfg, space)
        m, t = trial_cfg['model'], trial_cfg['training']
        assert m['hidden_channels'] in (16, 32, 64, 128, 256)
        assert m['embedding_dim'] in (8, 16, 32, 64) and m['embedding_dim'] <= m['hidden_channels']
        emb_equals_hidden += m['embedding_dim'] == m['hidden_channels']
        assert 1 <= m['gnn_num_layers'] <= 4
        dims = m['classifier_hidden_dims']
        assert 1 <= len(dims) <= 3
        assert all(a > b for a, b in zip(dims, dims[1:])), dims          # strictly decreasing
        assert dims[0] <= m['hidden_channels'] // 2 and dims[-1] >= 4    # one power below hidden, >= 2**low
        assert all(d & (d - 1) == 0 for d in dims)                       # powers of 2
        assert m['pooling'] == 'attention' and 'model.pooling' not in values     # fixed, not searched
        assert m['demo_embed_dim'] in (2, 4, 8)
        for key in ('node_dropout', 'classifier_dropout', 'edge_dropout'):
            assert 0.0 <= m[key] <= 0.7
        assert 1e-4 <= t['lr'] <= 1e-2 and 1e-5 <= t['weight_decay'] <= 1e-1
        assert trial_cfg['graph'] == cfg['graph']                         # untouched sections
        study.tell(trial, 0.0)
    assert emb_equals_hidden > 0                                           # at_most is inclusive
    assert cfg['model']['hidden_channels'] == 64                          # original config not mutated

    # Rebuilding from params gives the same config
    best = study.trials[-1]
    rebuilt, values2 = config_from_params(cfg, space, best.params)
    assert values2 == values and rebuilt['model'] == trial_cfg['model']
    flat = flatten_values(values)
    assert isinstance(flat['classifier_hidden_dims'], str) and 'lr' in flat


def test_at_most_inclusive_below_strict():
    cfg = {'model': {'hidden_channels': 32}}
    space = {'model.emb': {'type': 'pow2', 'low': 1, 'high': 10, 'at_most': 'model.hidden_channels'},
             'model.dims': {'type': 'decreasing_pow2', 'min_layers': 1, 'max_layers': 1, 'low': 1,
                            'below': 'model.hidden_channels'}}
    trial_cfg, _ = suggest_config(optuna.trial.FixedTrial({'log2_emb': 5, 'dims_num_layers': 1, 'dims_log2_l0': 4}),
                                  cfg, space)
    assert trial_cfg['model']['emb'] == 32 and trial_cfg['model']['dims'] == [16]
    # The sampled ranges are exactly [low, 5] for at_most and [low, 4] for below
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    suggest_config(trial, cfg, space)
    assert trial.distributions['log2_emb'].high == 5
    assert trial.distributions['dims_log2_l0'].high == 4


def test_duplicate_param_names_rejected():
    space = {'model.lr': {'type': 'float', 'low': 0.1, 'high': 1.0},
             'training.lr': {'type': 'float', 'low': 0.1, 'high': 1.0}}
    with pytest.raises(ValueError):
        suggest_config(optuna.trial.FixedTrial({'lr': 0.5}), {}, space)


def test_remaining_trials_counts_pruned():
    study = optuna.create_study()
    for i in range(3):
        t = study.ask()
        study.tell(t, float(i))
    t = study.ask()
    study.tell(t, state=optuna.trial.TrialState.PRUNED)
    t = study.ask()
    study.tell(t, state=optuna.trial.TrialState.FAIL)
    assert remaining_trials(study, 10) == 6   # failed trials don't count
    assert remaining_trials(study, 2) == 0


def test_make_pruner_switch():
    hb = {'type': 'hyperband', 'min_resource': 5, 'reduction_factor': 2}
    assert isinstance(make_pruner({**hb, 'enabled': False}, 100), optuna.pruners.NopPruner)
    assert isinstance(make_pruner({**hb, 'enabled': True}, 100), optuna.pruners.HyperbandPruner)
    assert isinstance(make_pruner(hb, 100), optuna.pruners.HyperbandPruner)          # enabled defaults to true
    assert isinstance(make_pruner({'type': 'none'}, 100), optuna.pruners.NopPruner)
    assert isinstance(make_pruner(None, 100), optuna.pruners.NopPruner)
    assert load_optuna_cfg()['optuna']['pruner']['enabled'] is False                # off by default
