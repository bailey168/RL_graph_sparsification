"""Training the GATv2 on masked graphs, shared by the reward predictor and the final GNN.

rl.gnn_tuning picks the mode: 'fixed' (default) trains one model with cfg['model'] / cfg['training']; 'optuna' runs
one Optuna study.

The Optuna mode uses the same machinery and outputs as topk/run_optuna.py (common/tuning.py search space, TPE, one
SQLite file per study, best-trial checkpoint, study plots). In both modes batches come from rl.masking.MaskedLoader: either a fresh random mask per
subject per batch (mask augmentation, predictor) or fixed per-subject masks (agent-stopped graphs, final GNN).
"""
import os
import time

import numpy as np
import optuna
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common.config import snapshot
from common.models import build_model
from common.train import fit
from common.tuning import (config_from_params, count_states, flatten_values, is_new_best, make_study,
                           progress_callback, pruning_callback, remaining_trials, save_study_outputs, suggest_config)
from common.utils import calc_r2_corr, set_deterministic_training
from rl.masking import MaskedLoader, masked_batch
from topk.run_optuna import read_best_trial, write_best_artifacts


def make_masked_loaders(sets, masks, data, batch_size, seed, n_steps):
    """Train (shuffled), val and test MaskedLoaders. Seeds everything, so call it right before building the model."""
    generator = set_deterministic_training(seed)
    mask_generator = torch.Generator().manual_seed(seed + 1)
    return {name: MaskedLoader(sets[name], data.edge_list, data.num_nodes, batch_size, shuffle=name == 'train',
                               generator=generator if name == 'train' else None, masks=masks[name],
                               n_steps=n_steps, mask_generator=mask_generator)
            for name in ('train', 'val', 'test')}


def run_masked_study(cfg, study_dir, label, data, sets, masks, device, n_trials):
    """Run (or resume) one study and write its outputs into study_dir.

    sets: {'train', 'val', 'test'} -> SubjectTensors; masks: same keys -> (n, E) bool tensor, or None (random
    masks per batch). Returns a dict with the best trial's saved metrics and study bookkeeping.
    """
    opt_cfg = cfg['optuna']
    space = opt_cfg['search_space']
    seed = cfg['rl']['seed']
    n_steps = cfg['rl']['n_steps']
    y_pre = data.preprocessors['y']
    os.makedirs(study_dir, exist_ok=True)

    def train_trial(trial_cfg, epoch_callback=None):
        loaders = make_masked_loaders(sets, masks, data, trial_cfg['training']['batch_size'], seed, n_steps)
        model = build_model(trial_cfg['model'], data.num_nodes, data.demo_dim)
        return model, fit(model, loaders, trial_cfg['training'], device, y_pre, epoch_callback=epoch_callback)

    def objective(trial):
        trial_cfg, _ = suggest_config(trial, cfg, space)
        try:
            model, result = train_trial(trial_cfg, pruning_callback(trial))
        except FloatingPointError as e:   # non-finite loss: a bad region of the search space
            trial.set_user_attr('error', repr(e))
            raise optuna.TrialPruned() from e
        for key, value in result['metrics'].items():
            trial.set_user_attr(key, value)
        for key in ('best_epoch', 'epochs_run', 'minutes'):
            trial.set_user_attr(key, result[key])
        value = result['best_val_loss']
        if is_new_best(trial.study, value):
            write_best_artifacts(study_dir, trial.number, model, result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return value

    study = make_study(os.path.join(study_dir, 'study.db'), label, opt_cfg, cfg['training']['max_epochs'])
    todo = remaining_trials(study, n_trials)
    start = time.time()
    if todo:
        print(f"  {todo} trials to run ({n_trials - todo} already finished)")
        study.optimize(objective, n_trials=todo, callbacks=[progress_callback(label)], gc_after_trial=True)
    states = count_states(study)
    if states['complete'] == 0:
        raise RuntimeError(f"No completed trials ({states})")

    best = study.best_trial
    best_cfg, best_values = config_from_params(cfg, space, best.params)
    saved = read_best_trial(study_dir)
    if saved is None or saved['trial_number'] != best.number:
        print(f"  Checkpoint does not match best trial #{best.number}; retraining it (same seed)")
        model, result = train_trial(best_cfg)
        write_best_artifacts(study_dir, best.number, model, result)
        saved = read_best_trial(study_dir)

    save_study_outputs(study, study_dir, importance_seed=seed)
    with open(os.path.join(study_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(snapshot(best_cfg, label=label, best_trial=best.number), f, sort_keys=False)
    return {'saved': saved, 'best_trial': best.number, 'best_val_loss': best.value, 'states': states,
            'best_values': flatten_values(best_values), 'study_minutes': (time.time() - start) / 60.0}


GNN_TUNING_MODES = ('fixed', 'optuna')


def gnn_tuning_mode(cfg):
    """rl.gnn_tuning: 'fixed' (one GATv2 with cfg['model'] / cfg['training'], the default) or 'optuna'."""
    mode = cfg['rl'].get('gnn_tuning', 'fixed')
    if mode not in GNN_TUNING_MODES:
        raise ValueError(f"rl.gnn_tuning must be one of {GNN_TUNING_MODES}, got {mode!r}")
    return mode


def run_fixed_training(cfg, out_dir, label, data, sets, masks, device):
    """Train one GNN with the fixed cfg['model'] / cfg['training'] (no Optuna) and write its outputs into out_dir.

    Same files as the best trial of run_masked_study (model.pt, losses.csv, test_predictions.csv, best_trial.json,
    config.yaml), minus the study files. Returns a dict with the same keys as run_masked_study.
    """
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    loaders = make_masked_loaders(sets, masks, data, cfg['training']['batch_size'], cfg['rl']['seed'],
                                  cfg['rl']['n_steps'])
    model = build_model(cfg['model'], data.num_nodes, data.demo_dim)
    result = fit(model, loaders, cfg['training'], device, data.preprocessors['y'])
    write_best_artifacts(out_dir, None, model, result)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(snapshot(cfg, label=label, gnn_tuning='fixed'), f, sort_keys=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'saved': read_best_trial(out_dir), 'best_trial': None, 'best_val_loss': result['best_val_loss'],
            'states': {'complete': 1, 'pruned': 0, 'failed': 0}, 'best_values': {},
            'study_minutes': (time.time() - start) / 60.0}


def train_stage_gnn(cfg, out_dir, label, data, sets, masks, device, n_trials):
    """The stage's GNN: one fixed-architecture run or an Optuna study, depending on rl.gnn_tuning."""
    if gnn_tuning_mode(cfg) == 'fixed':
        return run_fixed_training(cfg, out_dir, label, data, sets, masks, device)
    return run_masked_study(cfg, out_dir, label, data, sets, masks, device, n_trials)


@torch.no_grad()   # not inference_mode: the env updates these predictions in place
def predict_masked(model, subjects, mask, data, batch_size=4096):
    """Scaled predictions (n,) of `model` for SubjectTensors `subjects` with (n, E) bool masks."""
    model.eval()
    outs = []
    for start in range(0, len(subjects), batch_size):
        sl = slice(start, start + batch_size)
        batch = masked_batch(subjects.edge_vals[sl], mask[sl].to(subjects.y.device), data.edge_list, data.num_nodes,
                             subjects.demo[sl])
        outs.append(model(batch).view(-1))
    return torch.cat(outs)


def original_scale_metrics(y_scaled, pred_scaled, y_pre, prefix=''):
    """R², R²-corr, MAE, RMSE on the original target scale (same definitions as common.train.evaluate)."""
    inv = lambda t: y_pre.inverse_transform(t.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y, p = inv(y_scaled), inv(pred_scaled)
    metrics = {'r2': float(r2_score(y, p)), 'r2_corr': calc_r2_corr(y, p),
               'mae': float(mean_absolute_error(y, p)), 'rmse': float(np.sqrt(mean_squared_error(y, p)))}
    return {f'{prefix}{k}': v for k, v in metrics.items()}


def load_trained_gnn(study_dir, num_nodes, demo_dim, device):
    """The study's best model (config.yaml + model.pt), frozen in eval mode."""
    with open(os.path.join(study_dir, 'config.yaml')) as f:
        model_cfg = yaml.safe_load(f)['model']
    model = build_model(model_cfg, num_nodes, demo_dim)
    model.load_state_dict(torch.load(os.path.join(study_dir, 'model.pt'), map_location='cpu'))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
