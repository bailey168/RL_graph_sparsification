"""Pruning env rewards, GAE, rollouts/PPO update, and a planted toy problem the agent must solve."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from rl.env import PruningEnv
from rl.masking import SubjectTensors, removal_schedule
from rl.policy import ActorCritic
from rl.ppo import collect_rollout, compute_gae, ppo_update, run_greedy

POLICY_CFG = {'embedding_dim': 8, 'demo_embed_dim': 4, 'hidden_channels': 16, 'heads': 2, 'gnn_num_layers': 2,
              'mlp_hidden': 32, 'init_stop_prob': 0.05}
PPO_CFG = {'epochs': 4, 'minibatch_size': 256, 'lr': 3e-3, 'clip': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01,
           'max_grad_norm': 0.5, 'gamma': 1.0, 'gae_lambda': 0.95, 'target_kl': 0.05}


class SumPredictor(nn.Module):
    """Frozen toy predictor: prediction = sum of the kept edges' values (each edge appears in both directions)."""

    def forward(self, batch):
        per_graph = scatter(batch.edge_attr.view(-1), batch.batch[batch.edge_index[0]], dim=0,
                            dim_size=batch.num_graphs, reduce='sum')
        return per_graph / 2


def toy_problem(n=512, num_nodes=8, n_signal=6, seed=0, signal=None):
    """y = sum of the signal edges; the predictor also sums the noise edges, so removing noise helps and removing
    signal hurts. The optimal policy removes noise edges first and stops before touching the signal."""
    g = torch.Generator().manual_seed(seed)
    edge_list = torch.combinations(torch.arange(num_nodes), 2)
    E = len(edge_list)
    if signal is None:
        signal = torch.zeros(E, dtype=torch.bool)
        signal[torch.randperm(E, generator=g)[:n_signal]] = True
    edge_vals = torch.randn(n, E, generator=g)
    y = (edge_vals * signal).sum(1)
    subjects = SubjectTensors(edge_vals, torch.randn(n, 2, generator=g), y, torch.arange(n))
    data = SimpleNamespace(edge_list=edge_list, num_nodes=num_nodes, num_edges=E)
    return subjects, data, signal


def test_env_rewards_telescope_and_termination():
    subjects, data, _ = toy_problem(n=6)
    env = PruningEnv(SumPredictor(), data, n_steps=20).reset(subjects)
    E, sched = data.num_edges, removal_schedule(data.num_edges)
    total = torch.zeros(6)
    g = torch.Generator().manual_seed(0)
    stop_at = torch.tensor([0, 3, 20, 20, 7, 20])   # rows that never stop run to 0 edges
    for k in range(20):
        stop = stop_at == k
        remove = torch.zeros(6, E, dtype=torch.bool)
        for i in range(6):
            kept = env.mask[i].nonzero().view(-1)
            remove[i, kept[torch.randperm(len(kept), generator=g)[:env.n_remove]]] = True
        before = env.mask.clone()
        reward, _ = env.step(stop, remove)
        assert torch.equal(env.mask[stop_at < k], before[stop_at < k])
        total += reward
    assert env.done.all()
    assert torch.allclose(total, env.err0 - env.err, atol=1e-5)
    assert torch.equal(env.stop_step, stop_at)
    assert (env.mask.sum(1) == torch.tensor([sched[s] for s in stop_at])).all()
    assert (env.mask.sum(1)[stop_at == 20] == 0).all()
    assert ((env.removal_step >= 0) == ~env.mask).all()
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(6, dtype=torch.bool), torch.zeros(6, E, dtype=torch.bool))


def test_gae_hand_example():
    # One episode of 3 steps (ends at step 2), one that ends at step 0; gamma = 1, lambda = 0.5
    rewards = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 0.0]])
    values = torch.tensor([[0.5, 1.0], [1.0, 0.0], [2.0, 0.0]])
    active = torch.tensor([[True, True], [True, False], [True, False]])
    ended = torch.tensor([[False, True], [False, False], [True, False]])
    adv, ret = compute_gae(rewards, values, active, ended, gamma=1.0, lam=0.5)
    d2 = 3.0 - 2.0
    d1 = 0.0 + 2.0 - 1.0
    d0 = 1.0 + 1.0 - 0.5
    a1 = d1 + 0.5 * d2
    assert adv[:, 0].tolist() == pytest.approx([d0 + 0.5 * a1, a1, d2])
    assert adv[:, 1].tolist() == pytest.approx([2.0 - 1.0, 0.0, 0.0])
    assert ret[0, 0].item() == pytest.approx(adv[0, 0].item() + 0.5)


def make_agent(data, demo_dim=2, seed=0):
    torch.manual_seed(seed)
    return ActorCritic(POLICY_CFG, data.num_nodes, demo_dim, data.edge_list, n_steps=20)


def test_rollout_and_update_run():
    subjects, data, _ = toy_problem(n=64)
    env = PruningEnv(SumPredictor(), data, n_steps=20)
    policy = make_agent(data)
    g = torch.Generator().manual_seed(0)
    roll = collect_rollout(policy, env, subjects, g)
    assert len(roll) == roll.stats['transitions'] >= 64
    assert torch.isfinite(roll.adv).all() and torch.isfinite(roll.ret).all()
    # Undiscounted returns of first transitions equal the episode return e0 - e_stop (GAE with lambda = 1 check)
    roll1 = collect_rollout(policy, env, subjects, torch.Generator().manual_seed(1), lam=1.0)
    first = roll1.t == 0
    assert torch.allclose(roll1.ret[first], (env.err0 - env.err)[roll1.subj[first]], atol=1e-4)
    opt = torch.optim.Adam(policy.parameters(), lr=PPO_CFG['lr'])
    stats = ppo_update(policy, opt, roll, subjects, PPO_CFG, generator=g)
    assert all(map(lambda v: v == v, stats.values()))   # no NaNs
    res = run_greedy(policy, env, subjects, batch_size=40)
    assert res['mask'].shape == (64, data.num_edges)
    assert (res['mask'].sum(1) == torch.tensor([removal_schedule(data.num_edges)[s] for s in res['stop_step']])).all()


@pytest.mark.slow
def test_agent_learns_planted_toy_problem():
    """PPO must beat both 'never prune' and random-order pruning, and learn to keep the signal edges (~1-2 min CPU).

    Uses gamma = 0.8: with gamma = 1 and a policy that prunes to the end, the return e_0 - e_stop no longer depends
    on the removal order, so the order gets no gradient and the agent stays stuck pruning everything.
    """
    gamma = 0.8
    subjects, data, signal = toy_problem(n=256, num_nodes=7, n_signal=5)
    val, _, _ = toy_problem(n=256, num_nodes=7, seed=9, signal=signal)
    env = PruningEnv(SumPredictor(), data, n_steps=20)
    policy = make_agent(data)
    cfg = {**PPO_CFG, 'minibatch_size': 512, 'gamma': gamma}
    opt = torch.optim.Adam(policy.parameters(), lr=cfg['lr'])
    g = torch.Generator().manual_seed(0)
    for _ in range(60):
        roll = collect_rollout(policy, env, subjects, g, gamma=gamma)
        ppo_update(policy, opt, roll, subjects, cfg, generator=g)
    res = run_greedy(policy, env, val)
    full_mse = res['err0'].mean().item()
    agent_mse = res['err'].mean().item()
    # Random-order pruning stopped at the agent's own keep levels
    rand_mask = torch.zeros_like(res['mask'])
    for i, k in enumerate(res['mask'].sum(1).tolist()):
        rand_mask[i, torch.randperm(data.num_edges)[:k]] = True
    rand_mse = (((val.edge_vals * rand_mask).sum(1) - val.y) ** 2).mean().item()
    signal_kept = res['mask'][:, signal].float().mean().item()
    noise_kept = res['mask'][:, ~signal].float().mean().item()
    msg = f"full={full_mse:.3f} agent={agent_mse:.3f} random={rand_mse:.3f} signal={signal_kept:.2f} noise={noise_kept:.2f}"
    assert agent_mse < 0.5 * full_mse, msg
    assert agent_mse < 0.5 * rand_mse, msg
    assert signal_kept > noise_kept + 0.3, msg
