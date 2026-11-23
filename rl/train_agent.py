"""Stage 2: train the PPO pruning agent on half B of the train split, rewarded by the frozen stage-1 predictor.

Each iteration runs one stochastic episode for `rl.ppo.subjects_per_iter` half-B subjects, computes GAE and does a
clipped PPO update. Every `eval_every` iterations the greedy agent is evaluated on the val subjects (and on a same-size
half-B subset, for the train/val gap); the checkpoint with the lowest val MSE on agent-stopped graphs (frozen predictor,
scaled target) is kept, and training stops after `patience` evaluations without improvement.

Output in {output_dir}/{FC}/{TEST}/agent/: log.csv (per iteration, incl. rollout/update seconds), evals.csv (greedy
evaluations, incl. eval seconds), best.pt (policy), last.pt (resume state), config.yaml, plots/learning_curves.png,
timing.json (minutes per component, accumulated across resumes), metrics.json (written last; val/test
frozen-predictor metrics of the best agent, plus time_*_min). A run without metrics.json resumes from last.pt.

Example:
    python rl/train_agent.py --tests GF --max-subjects 2000 --iterations 20
"""
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*is deprecated", category=FutureWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import snapshot
from common.utils import set_deterministic_training
from rl.env import PruningEnv
from rl.policy import ActorCritic
from rl.ppo import collect_rollout, greedy_metrics, ppo_update, run_greedy
from rl.predictor import clear_stage
from rl.setup import base_parser, get_device, load_cli_config, load_rl_data, stage_dir
from rl.study import load_trained_gnn
from rl.timing import TIMING_FILE, StageTimer
from topk.run_sweep import append_row, free_memory, write_json

AGENT_FILES = ('metrics.json', 'best.pt', 'last.pt', 'log.csv', 'evals.csv', TIMING_FILE)


def build_policy(cfg, data, device):
    return ActorCritic(cfg['rl']['policy'], data.num_nodes, data.demo_dim, data.edge_list,
                       cfg['rl']['n_steps']).to(device)


def load_agent(cfg, data, agent_dir, device):
    policy = build_policy(cfg, data, device)
    policy.load_state_dict(torch.load(os.path.join(agent_dir, 'best.pt'), map_location=device))
    return policy.eval()


def load_frozen_predictor(cfg, fc_type, test, data, device):
    pred_dir = stage_dir(cfg, fc_type, test, 'predictor')
    if not os.path.exists(os.path.join(pred_dir, 'metrics.json')):
        raise FileNotFoundError(f"Run rl/predictor.py first (no {pred_dir}/metrics.json)")
    return load_trained_gnn(pred_dir, data.num_nodes, data.demo_dim, device)


def truncate_csv(path, max_iteration):
    """Drop rows logged after the iteration a resume restarts from."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        df[df['iteration'] <= max_iteration].to_csv(path, index=False)


def plot_learning_curves(agent_dir):
    log = pd.read_csv(os.path.join(agent_dir, 'log.csv'))
    evals = pd.read_csv(os.path.join(agent_dir, 'evals.csv'))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ('return_mean', 'Episode return (e0 − e_stop), rollouts', log),
        ('keep_frac_mean', 'Mean edges kept, rollouts', log),
        ('entropy', 'Entropy bonus term', log),
        ('approx_kl', 'Approx. KL per update', log),
    ]
    for ax, (col, title, df) in zip(axes.flat, panels):
        ax.plot(df['iteration'], df[col])
        ax.set_title(title)
    ax = axes.flat[4]
    ax.plot(evals['iteration'], evals['val_mse'], label='val (agent-stopped)')
    ax.plot(evals['iteration'], evals['train_mse'], label='half B (agent-stopped)')
    ax.plot(evals['iteration'], evals['val_mse_full'], '--', label='val (full graph)')
    ax.set_title('Greedy MSE (scaled target), frozen predictor')
    ax.legend()
    ax = axes.flat[5]
    ax.plot(evals['iteration'], evals['val_keep_frac_mean'], label='val')
    ax.plot(evals['iteration'], evals['train_keep_frac_mean'], label='half B')
    ax.set_title('Greedy mean edges kept')
    ax.legend()
    for ax in axes.flat:
        ax.set_xlabel('PPO iteration')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(agent_dir, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(agent_dir, 'plots', 'learning_curves.png'), dpi=150)
    plt.close(fig)


def train_agent(cfg, fc_type, test, data, predictor, agent_dir, device, iterations, timer):
    rl_cfg, ppo_cfg = cfg['rl'], cfg['rl']['ppo']
    seed, n_steps = rl_cfg['seed'], rl_cfg['n_steps']
    os.makedirs(agent_dir, exist_ok=True)
    setup_start = time.perf_counter()
    with open(os.path.join(agent_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(snapshot(cfg, fc=fc_type, test=test, stage='agent'), f, sort_keys=False)

    set_deterministic_training(seed)
    policy = build_policy(cfg, data, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_cfg['lr'])
    cpu_gen = torch.Generator().manual_seed(seed)
    dev_gen = torch.Generator(device=device).manual_seed(seed)
    env = PruningEnv(predictor, data, n_steps)
    agent_set, val_set = data.sets['train_agent'], data.sets['val']
    n_check = min(len(val_set), len(agent_set))
    check_idx = torch.randperm(len(agent_set), generator=torch.Generator().manual_seed(seed + 4))[:n_check]
    train_check = agent_set.subset(check_idx.to(device))
    y_pre = data.preprocessors['y']
    log_path, evals_path = os.path.join(agent_dir, 'log.csv'), os.path.join(agent_dir, 'evals.csv')
    last_path, best_path = os.path.join(agent_dir, 'last.pt'), os.path.join(agent_dir, 'best.pt')

    state = {'iteration': 0, 'best_val_mse': float('inf'), 'best_iteration': 0, 'evals_since_best': 0}
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt['policy'])
        optimizer.load_state_dict(ckpt['optimizer'])
        cpu_gen.set_state(ckpt['cpu_gen'])
        dev_gen.set_state(ckpt['dev_gen'])
        state = ckpt['state']
        truncate_csv(log_path, state['iteration'])
        truncate_csv(evals_path, state['iteration'])
        print(f"  Resuming from iteration {state['iteration']} (best val MSE {state['best_val_mse']:.4f})")
    timer.add('setup', time.perf_counter() - setup_start)

    start = time.time()
    batch_size = ppo_cfg['eval_batch_size']
    pending = {'ppo_rollout': 0.0, 'ppo_update': 0.0}   # added to the timer at each checkpoint, in step with last.pt
    for it in range(state['iteration'] + 1, iterations + 1):
        timer.sync()
        t0 = time.perf_counter()
        idx = torch.randperm(len(agent_set), generator=cpu_gen)[:ppo_cfg['subjects_per_iter']].to(device)
        subjects = agent_set.subset(idx)
        roll = collect_rollout(policy, env, subjects, dev_gen, ppo_cfg['gamma'], ppo_cfg['gae_lambda'])
        timer.sync()
        t1 = time.perf_counter()
        stats = ppo_update(policy, optimizer, roll, subjects, ppo_cfg, cpu_gen)
        timer.sync()
        t2 = time.perf_counter()
        pending['ppo_rollout'] += t1 - t0
        pending['ppo_update'] += t2 - t1
        append_row(log_path, {'iteration': it, **roll.stats, **stats, 'seconds': t2 - t0,
                              'rollout_seconds': t1 - t0, 'update_seconds': t2 - t1})

        if it % ppo_cfg['eval_every'] == 0 or it == iterations:
            val = greedy_metrics(run_greedy(policy, env, val_set, batch_size), val_set, y_pre, data.num_edges, 'val_')
            tr = greedy_metrics(run_greedy(policy, env, train_check, batch_size), train_check, y_pre, data.num_edges,
                                'train_')
            improved = val['val_mse'] < state['best_val_mse']
            if improved:
                state.update(best_val_mse=val['val_mse'], best_iteration=it, evals_since_best=0)
                torch.save(policy.state_dict(), best_path)
            else:
                state['evals_since_best'] += 1
            state['iteration'] = it
            torch.save({'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(), 'state': state,
                        'cpu_gen': cpu_gen.get_state(), 'dev_gen': dev_gen.get_state()}, last_path)
            eval_seconds = time.perf_counter() - t2
            append_row(evals_path, {'iteration': it, **val, **tr, 'best': improved, 'eval_seconds': eval_seconds})
            for name, seconds in pending.items():
                timer.add(name, seconds, save=False)
            timer.add('ppo_eval', eval_seconds)
            pending = dict.fromkeys(pending, 0.0)
            print(f"  it {it}: return={roll.stats['return_mean']:.4f} keep={roll.stats['keep_frac_mean']:.2f} | "
                  f"greedy val MSE={val['val_mse']:.4f} (full {val['val_mse_full']:.4f}) "
                  f"R²={val['val_r2']:.4f} keep={val['val_keep_frac_mean']:.2f} | half B MSE={tr['train_mse']:.4f}"
                  f"{' *' if improved else ''} ({(time.time() - start) / 60:.1f}m)", flush=True)
            if state['evals_since_best'] >= ppo_cfg['patience']:
                print(f"  Early stop: no val improvement in {ppo_cfg['patience']} evaluations")
                break

    with timer.section('final_eval'):
        plot_learning_curves(agent_dir)
        policy = load_agent(cfg, data, agent_dir, device)
        metrics = {}
        for name in ('val', 'test'):
            metrics.update(greedy_metrics(run_greedy(policy, env, data.sets[name], batch_size), data.sets[name],
                                          y_pre, data.num_edges, f'{name}_'))
    timer.finish()
    row = {'fc': fc_type, 'test': test, 'stage': 'agent', 'best_iteration': state['best_iteration'],
           'iterations_run': state['iteration'], 'n_agent': len(agent_set), 'minutes': (time.time() - start) / 60,
           **metrics, **timer.flat()}
    write_json(os.path.join(agent_dir, 'metrics.json'), row)   # written last: marks the stage as complete
    return row


def main():
    parser = base_parser("Stage 2: PPO pruning agent")
    parser.add_argument('--iterations', type=int, help='Override rl.ppo.iterations')
    args = parser.parse_args()
    cfg, fc_types, tests = load_cli_config(args)
    iterations = args.iterations or cfg['rl']['ppo']['iterations']
    device = get_device()
    print(f"Using device: {device} | up to {iterations} PPO iterations")

    failed = False
    for test in tests:
        for fc_type in fc_types:
            agent_dir = stage_dir(cfg, fc_type, test, 'agent')
            if args.overwrite:
                clear_stage(agent_dir, AGENT_FILES)
            if os.path.exists(os.path.join(agent_dir, 'metrics.json')):
                print(f"[{fc_type}|{test}] Agent complete, skipping")
                continue
            print(f"\n=== {fc_type} | {test} | PPO agent ===")
            try:
                timer = StageTimer(agent_dir, device)
                with timer.section('load_data'):
                    data = load_rl_data(cfg, test, fc_type, device, args.max_subjects, save=False)
                with timer.section('load_predictor'):
                    predictor = load_frozen_predictor(cfg, fc_type, test, data, device)
                row = train_agent(cfg, fc_type, test, data, predictor, agent_dir, device, iterations, timer)
                print(f"  best iteration {row['best_iteration']}: test MSE={row['test_mse']:.4f} "
                      f"(full {row['test_mse_full']:.4f}) R²={row['test_r2']:.4f} (full {row['test_r2_full']:.4f}) "
                      f"keep={row['test_keep_frac_mean']:.2f}±{row['test_keep_frac_sd']:.2f} | "
                      f"{row['time_total_min']:.1f} min")
            except Exception:
                traceback.print_exc()
                failed = True
            finally:
                free_memory()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
