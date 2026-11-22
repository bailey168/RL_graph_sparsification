"""Rollouts, GAE and the PPO clipped update for the pruning agent, plus greedy evaluation episodes."""
from dataclasses import dataclass

import torch
import torch.nn as nn

from rl.study import original_scale_metrics


@dataclass
class Rollout:
    """Flattened transitions of one batch of episodes. subj indexes the SubjectTensors the episodes ran on."""
    subj: torch.Tensor     # (T,)
    mask: torch.Tensor     # (T, E) bool, state before the action
    t: torch.Tensor        # (T,)
    stop: torch.Tensor     # (T,) bool
    order: torch.Tensor    # (T, M) edges removed in draw order, -1 padded
    logp: torch.Tensor     # (T,)
    value: torch.Tensor    # (T,)
    adv: torch.Tensor      # (T,)
    ret: torch.Tensor      # (T,)
    stats: dict

    def __len__(self):
        return self.t.shape[0]


def compute_gae(rewards, values, active, ended, gamma=1.0, lam=0.95):
    """GAE over K synchronous steps for B episodes. All inputs (K, B); `ended` = the episode ended at this step
    (the next state is terminal, value 0); entries where `active` is False are ignored. Returns (adv, returns)."""
    K, B = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    next_value = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for k in reversed(range(K)):
        stop = ended[k] | ~active[k]
        nv = torch.where(stop, torch.zeros_like(next_value), next_value)
        delta = rewards[k] + gamma * nv - values[k]
        last = delta + gamma * lam * torch.where(stop, torch.zeros_like(last), last)
        last = last * active[k]
        adv[k] = last
        next_value = values[k]
    return adv, (adv + values) * active


def collect_rollout(policy, env, subjects, generator, gamma=1.0, lam=0.95):
    """Run one stochastic episode per subject and return the flattened transitions with advantages."""
    env.reset(subjects)
    B, E = len(subjects), env.data.num_edges
    device = subjects.y.device
    K = env.n_steps
    max_m = max(env.schedule[k] - env.schedule[k + 1] for k in range(K))
    R = torch.zeros(K, B, device=device)
    V = torch.zeros(K, B, device=device)
    A = torch.zeros(K, B, dtype=torch.bool, device=device)
    D = torch.zeros(K, B, dtype=torch.bool, device=device)
    steps = []
    policy.eval()
    for k in range(K):
        rows = (~env.done).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            break
        m = env.n_remove
        mask_before = env.mask[rows].clone()
        stop, order, logp, value = policy.act(subjects.edge_vals[rows], mask_before, k, subjects.demo[rows], m,
                                              generator=generator)
        stop_all = torch.zeros(B, dtype=torch.bool, device=device)
        stop_all[rows] = stop
        remove = torch.zeros(B, E, dtype=torch.bool, device=device)
        remove[rows] = torch.zeros(len(rows), E, dtype=torch.bool, device=device).scatter_(1, order, True)
        reward, done = env.step(stop_all, remove)
        R[k, rows], V[k, rows], A[k, rows], D[k, rows] = reward[rows], value, True, done[rows]
        padded = torch.full((len(rows), max_m), -1, dtype=torch.long, device=device)
        padded[:, :m] = order
        steps.append((rows, mask_before, torch.full_like(rows, k), stop, padded, logp, value))

    adv, ret = compute_gae(R, V, A, D, gamma, lam)
    cat = lambda i: torch.cat([s[i] for s in steps])
    t = cat(2)
    rows_all = cat(0)
    episode_return = env.err0 - env.err
    stats = {'transitions': len(t), 'return_mean': episode_return.mean().item(),
             'stop_step_mean': env.stop_step.float().mean().item(), 'keep_frac_mean': env.keep_frac.mean().item(),
             'mse_full': env.err0.mean().item(), 'mse_stop': env.err.mean().item()}
    return Rollout(subj=rows_all, mask=cat(1), t=t, stop=cat(3), order=cat(4), logp=cat(5), value=cat(6),
                   adv=adv[t, rows_all], ret=ret[t, rows_all], stats=stats)


def ppo_update(policy, optimizer, rollout, subjects, ppo_cfg, generator=None):
    """Clipped PPO update over `epochs` passes of shuffled minibatches; stops early when approx KL > target_kl."""
    policy.train()
    T = len(rollout)
    mb = ppo_cfg['minibatch_size']
    clip, target_kl = ppo_cfg['clip'], ppo_cfg.get('target_kl')
    totals = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'approx_kl': 0.0, 'clip_frac': 0.0}
    n_updates, epochs_run = 0, 0
    for _ in range(ppo_cfg['epochs']):
        epochs_run += 1
        perm = torch.randperm(T, generator=generator).to(rollout.t.device)
        epoch_kl, epoch_batches = 0.0, 0
        for start in range(0, T, mb):
            idx = perm[start:start + mb]
            subj = rollout.subj[idx]
            logp, entropy, value = policy.evaluate_actions(subjects.edge_vals[subj], rollout.mask[idx], rollout.t[idx],
                                                           subjects.demo[subj], rollout.stop[idx], rollout.order[idx])
            log_ratio = logp - rollout.logp[idx]
            ratio = log_ratio.exp()
            adv = rollout.adv[idx]
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            value_loss = ((value - rollout.ret[idx]) ** 2).mean()
            loss = policy_loss + ppo_cfg['value_coef'] * value_loss - ppo_cfg['entropy_coef'] * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), ppo_cfg['max_grad_norm'])
            optimizer.step()

            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean().item()
                totals['policy_loss'] += policy_loss.item()
                totals['value_loss'] += value_loss.item()
                totals['entropy'] += entropy.mean().item()
                totals['approx_kl'] += kl
                totals['clip_frac'] += ((ratio - 1).abs() > clip).float().mean().item()
            epoch_kl += kl
            epoch_batches += 1
            n_updates += 1
        if target_kl is not None and epoch_kl / epoch_batches > target_kl:
            break
    return {**{k: v / max(n_updates, 1) for k, v in totals.items()}, 'ppo_epochs': epochs_run}


@torch.no_grad()
def run_greedy(policy, env, subjects, batch_size=2048):
    """Greedy episodes for every subject. Returns CPU tensors: final mask (n, E), removal_step (n, E) int8 (-1 = kept),
    stop_step (n,), pred0/pred (scaled predictions on the full and the stopped graph), err0/err."""
    policy.eval()
    keys = ('mask', 'removal_step', 'stop_step', 'pred0', 'pred', 'err0', 'err')
    out = {k: [] for k in keys}
    for start in range(0, len(subjects), batch_size):
        sub = subjects.subset(torch.arange(start, min(start + batch_size, len(subjects)), device=subjects.y.device))
        env.reset(sub)
        B, E = len(sub), env.data.num_edges
        for k in range(env.n_steps):
            rows = (~env.done).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                break
            stop, order, _, _ = policy.act(sub.edge_vals[rows], env.mask[rows], k, sub.demo[rows], env.n_remove,
                                           greedy=True, with_value=False)
            stop_all = torch.zeros(B, dtype=torch.bool, device=sub.y.device)
            stop_all[rows] = stop
            remove = torch.zeros(B, E, dtype=torch.bool, device=sub.y.device)
            remove[rows] = torch.zeros(len(rows), E, dtype=torch.bool, device=sub.y.device).scatter_(1, order, True)
            env.step(stop_all, remove)
        for k in keys:
            out[k].append(getattr(env, k).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def greedy_metrics(result, subjects, y_pre, num_edges, prefix):
    """Frozen-predictor metrics on the agent-stopped graphs (and on the full graphs for reference)."""
    keep = result['mask'].sum(1).float() / num_edges
    y = subjects.y.cpu()
    return {f'{prefix}mse': result['err'].mean().item(), f'{prefix}mse_full': result['err0'].mean().item(),
            **original_scale_metrics(y, result['pred'], y_pre, prefix),
            f'{prefix}r2_full': original_scale_metrics(y, result['pred0'], y_pre)['r2'],
            f'{prefix}keep_frac_mean': keep.mean().item(), f'{prefix}keep_frac_sd': keep.std().item() if len(keep) > 1 else 0.0,
            f'{prefix}stop_step_mean': result['stop_step'].float().mean().item()}
