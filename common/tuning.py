"""Method-agnostic Optuna helpers: config-driven search spaces, studies, pruning and study outputs.

Used by topk/run_optuna.py and intended for the RL stage. A search space is a mapping from dotted
config keys to specs (see configs/topk_optuna.yaml):

    model.hidden_channels: {type: pow2, low: 5, high: 7}          # 2**5 .. 2**7, searched as an int exponent
    model.embedding_dim:   {type: pow2, low: 3, high: 6, at_most: model.hidden_channels}   # <= hidden (inclusive)
    model.gnn_num_layers:  {type: int, low: 1, high: 4}
    training.lr:           {type: float, low: 1.0e-4, high: 3.0e-2, log: true}
    model.node_dropout:    {type: float, low: 0.0, high: 0.6, step: 0.05}
    model.pooling:         {type: categorical, choices: [attention, mean]}
    model.classifier_hidden_dims:                                    # strictly decreasing powers of 2
        {type: decreasing_pow2, min_layers: 1, max_layers: 3, low: 2, below: model.hidden_channels}

`pow2` and `decreasing_pow2` take an optional dynamic upper bound from an earlier key: `at_most:` (inclusive,
value <= cfg[key]) or `below:` (strict, value < cfg[key]). Specs are sampled in the order they are listed,
so these must come after the key they refer to.
"""
import copy
import json
import os
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna

from common.config import get_dotted, set_dotted

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


# ---------------------------------------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------------------------------------

def _param_name(key):
    return key.split('.')[-1]


def check_search_space(search_space):
    """Raise if two dotted keys would map to the same Optuna parameter name."""
    names = [_param_name(k) for k in search_space]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"Search space keys must have unique last components; duplicated: {sorted(dupes)}")


def _upper_exponent(spec, cfg):
    """Largest allowed exponent: min of `high`, floor(log2(cfg[at_most])) (inclusive bound) and the largest
    exponent strictly below log2(cfg[below]) (strict bound), using whichever keys are present."""
    upper = spec.get('high', float('inf'))
    if 'at_most' in spec:
        upper = min(upper, int(get_dotted(cfg, spec['at_most'])).bit_length() - 1)
    if 'below' in spec:
        upper = min(upper, (int(get_dotted(cfg, spec['below'])) - 1).bit_length() - 1)
    if upper == float('inf'):
        raise ValueError(f"Spec needs `high`, `at_most` or `below`: {spec}")
    return int(upper)


def _suggest_decreasing_pow2(trial, name, spec, cfg):
    """Strictly decreasing powers of 2 (as in the nocv notebook's classifier dims).

    Layer i's exponent is in [low + layers_left_after_i, previous exponent - 1]; the first layer's upper
    bound comes from _upper_exponent. The number of layers is capped so a valid sequence always exists.
    """
    low = spec.get('low', 1)
    upper = _upper_exponent(spec, cfg)
    max_layers = min(spec['max_layers'], max(0, upper - low + 1))
    min_layers = min(spec.get('min_layers', 1), max_layers)
    n_layers = trial.suggest_int(f'{name}_num_layers', min_layers, max_layers)
    dims = []
    for i in range(n_layers):
        lo = low + (n_layers - i - 1)
        exp = trial.suggest_int(f'{name}_log2_l{i}', lo, upper)
        dims.append(2 ** exp)
        upper = exp - 1
    return dims


def suggest_value(trial, key, spec, cfg):
    name = _param_name(key)
    kind = spec['type']
    if kind == 'int':
        return trial.suggest_int(name, spec['low'], spec['high'], step=spec.get('step', 1), log=spec.get('log', False))
    if kind == 'float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']),
                                   step=spec.get('step'), log=spec.get('log', False))
    if kind == 'categorical':
        return trial.suggest_categorical(name, list(spec['choices']))
    if kind == 'pow2':
        return 2 ** trial.suggest_int(f'log2_{name}', spec['low'], _upper_exponent(spec, cfg))
    if kind == 'decreasing_pow2':
        return _suggest_decreasing_pow2(trial, name, spec, cfg)
    raise ValueError(f"Unknown search space type '{kind}' for {key}")


def suggest_config(trial, cfg, search_space):
    """Deep copy of cfg with every search-space key sampled from `trial`.

    Returns (trial_cfg, values) where values maps each dotted key to its resolved value (e.g. 64 for a
    pow2 spec, [32, 16] for decreasing_pow2). Works with optuna.trial.FixedTrial to rebuild a config
    from a finished trial's params.
    """
    check_search_space(search_space)
    trial_cfg = copy.deepcopy(cfg)
    values = {}
    for key, spec in search_space.items():
        value = suggest_value(trial, key, spec, trial_cfg)
        set_dotted(trial_cfg, key, value)
        values[key] = value
    return trial_cfg, values


def config_from_params(cfg, search_space, params):
    """Rebuild (trial_cfg, values) from a finished trial's Optuna params."""
    return suggest_config(optuna.trial.FixedTrial(params), cfg, search_space)


# ---------------------------------------------------------------------------------------------------------
# Studies and pruning
# ---------------------------------------------------------------------------------------------------------

def pruner_enabled(pruner_cfg):
    """Off when the block is missing, `enabled: false`, or `type: none`; `enabled` defaults to true."""
    return bool(pruner_cfg) and pruner_cfg.get('enabled', True) and pruner_cfg.get('type', 'none') != 'none'


def make_pruner(pruner_cfg, max_resource):
    if not pruner_enabled(pruner_cfg):
        return optuna.pruners.NopPruner()
    kind = pruner_cfg['type']
    if kind == 'hyperband':
        return optuna.pruners.HyperbandPruner(min_resource=pruner_cfg.get('min_resource', 1),
                                              max_resource=max_resource,
                                              reduction_factor=pruner_cfg.get('reduction_factor', 3))
    if kind == 'median':
        return optuna.pruners.MedianPruner(n_startup_trials=pruner_cfg.get('n_startup_trials', 5),
                                           n_warmup_steps=pruner_cfg.get('n_warmup_steps', 10))
    raise ValueError(f"Unknown pruner type '{kind}'")


def make_study(db_path, study_name, optuna_cfg, max_resource):
    """Create or resume a study stored in its own SQLite file."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    sampler_cfg = optuna_cfg.get('sampler', {})
    sampler = optuna.samplers.TPESampler(seed=optuna_cfg.get('seed', 0),
                                         multivariate=sampler_cfg.get('multivariate', True),
                                         group=sampler_cfg.get('group', True),
                                         n_startup_trials=sampler_cfg.get('n_startup_trials', 10))
    return optuna.create_study(study_name=study_name, direction=optuna_cfg.get('direction', 'minimize'),
                               sampler=sampler, pruner=make_pruner(optuna_cfg.get('pruner'), max_resource),
                               storage=f'sqlite:///{os.path.abspath(db_path)}', load_if_exists=True)


FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def count_states(study):
    counts = Counter(t.state for t in study.trials)
    State = optuna.trial.TrialState
    return {'complete': counts[State.COMPLETE], 'pruned': counts[State.PRUNED], 'failed': counts[State.FAIL]}


def remaining_trials(study, n_trials):
    """Trials still needed to reach n_trials finished (complete + pruned) trials, so reruns only top up."""
    finished = sum(t.state in FINISHED for t in study.trials)
    return max(0, n_trials - finished)


def pruning_callback(trial):
    """epoch_callback for common.train.fit: report val loss each epoch and prune when Optuna says so."""
    def callback(epoch, train_loss, val_loss):
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return callback


def is_new_best(study, value):
    """True if `value` beats every completed trial so far (call from inside the objective)."""
    try:
        best = study.best_value
    except ValueError:   # no completed trials yet
        return True
    return value < best if study.direction == optuna.study.StudyDirection.MINIMIZE else value > best


def progress_callback(label):
    def callback(study, trial):
        msg = f"  [{label}] trial {trial.number} {trial.state.name.lower()}"
        if trial.value is not None:
            msg += f" val={trial.value:.4f} test_r2={trial.user_attrs.get('test_r2', float('nan')):.4f}"
        try:
            best = study.best_trial
            msg += f" | best #{best.number} val={best.value:.4f} test_r2={best.user_attrs.get('test_r2', float('nan')):.4f}"
        except ValueError:
            pass
        print(msg, flush=True)
    return callback


# ---------------------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------------------

def common_params(study):
    """Params present in every completed trial (conditional params such as later classifier layers are left out)."""
    complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not complete:
        return []
    names = set(complete[0].params)
    for t in complete[1:]:
        names &= set(t.params)
    return [p for p in complete[0].params if p in names]


def _hover_text(trial):
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    a = trial.user_attrs
    params = ''.join(f"<br>  {k}: {fmt(v)}" for k, v in trial.params.items())
    return (f"<br>Best epoch: {a.get('best_epoch', 'N/A')} | Time: {fmt(a.get('minutes', 'N/A'))}m"
            f"<br>Train R²: {fmt(a.get('train_r2', 'N/A'))} | R² corr: {fmt(a.get('train_r2_corr', 'N/A'))}"
            f"<br>Val R²: {fmt(a.get('val_r2', 'N/A'))} | R² corr: {fmt(a.get('val_r2_corr', 'N/A'))}"
            f"<br>Test R²: {fmt(a.get('test_r2', 'N/A'))} | R² corr: {fmt(a.get('test_r2_corr', 'N/A'))}"
            f"<br><b>Params:</b>{params}")


def _optimization_history_plotly(study):
    fig = optuna.visualization.plot_optimization_history(study)
    by_number = {t.number: t for t in study.trials}
    for trace in fig.data:
        if trace.mode == 'markers' and trace.x is not None:
            trace.customdata = [_hover_text(by_number[int(x)]) for x in trace.x]
            trace.hovertemplate = (trace.hovertemplate or "Trial: %{x}<br>Objective: %{y}") + "%{customdata}"
    return fig


def _save_matplotlib(ax, path):
    if hasattr(ax, 'flat'):
        ax = ax.flat[0]
    fig = ax.figure
    fig.set_size_inches(max(fig.get_size_inches()[0], 14), max(fig.get_size_inches()[1], 7))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_study_outputs(study, out_dir, importance_seed=0):
    """Write trials.csv, best_params.json, param_importances.json and plots/ (HTML via plotly, PNG via matplotlib).

    Plots: optimization_history, parallel_coordinate, param_importances, slice. Each is attempted
    independently, so a failure (e.g. importances with < 2 completed trials) only skips that plot.
    """
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    study.trials_dataframe().to_csv(os.path.join(out_dir, 'trials.csv'), index=False)
    with open(os.path.join(out_dir, 'best_params.json'), 'w') as f:
        json.dump({'trial_number': study.best_trial.number, 'value': study.best_value,
                   'params': study.best_trial.params, 'user_attrs': study.best_trial.user_attrs}, f, indent=2)

    params = common_params(study)
    evaluator = lambda: optuna.importance.FanovaImportanceEvaluator(seed=importance_seed)
    try:
        importances = optuna.importance.get_param_importances(study, evaluator=evaluator())
        with open(os.path.join(out_dir, 'param_importances.json'), 'w') as f:
            json.dump({k: float(v) for k, v in importances.items()}, f, indent=2)
    except Exception as e:   # noqa: BLE001 - needs >= 2 completed trials with varying params
        print(f"  Skipping param importances: {e!r}")

    vis, mvis = optuna.visualization, optuna.visualization.matplotlib
    plots = {
        'optimization_history': (lambda: _optimization_history_plotly(study),
                                 lambda: mvis.plot_optimization_history(study)),
        'parallel_coordinate': (lambda: vis.plot_parallel_coordinate(study, params=params),
                                lambda: mvis.plot_parallel_coordinate(study, params=params)),
        # no params= here: Optuna refuses it for dynamic search spaces and uses the shared params itself
        'param_importances': (lambda: vis.plot_param_importances(study, evaluator=evaluator()),
                              lambda: mvis.plot_param_importances(study, evaluator=evaluator())),
        'slice': (lambda: vis.plot_slice(study, params=params),
                  lambda: mvis.plot_slice(study, params=params)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, (make_html, make_png) in plots.items():
            try:
                make_html().write_html(os.path.join(plot_dir, f'{name}.html'))
            except Exception as e:   # noqa: BLE001
                print(f"  Skipping {name}.html: {e!r}")
            try:
                _save_matplotlib(make_png(), os.path.join(plot_dir, f'{name}.png'))
            except Exception as e:   # noqa: BLE001
                plt.close('all')
                print(f"  Skipping {name}.png: {e!r}")


def flatten_values(values):
    """Resolved search-space values as CSV-friendly scalars: lists become '64-32', keys lose their prefix."""
    out = {}
    for key, value in values.items():
        name = _param_name(key)
        out[name] = '-'.join(str(v) for v in value) if isinstance(value, (list, tuple)) else value
    return out

