"""Plackett–Luce removal distribution and the actor-critic."""
import itertools
import math

import pytest
import torch

from rl.policy import ActorCritic, NEG, removal_entropy, removal_log_prob, sample_removal

POLICY_CFG = {'embedding_dim': 8, 'demo_embed_dim': 4, 'hidden_channels': 16, 'heads': 2, 'gnn_num_layers': 2,
              'mlp_hidden': 16, 'init_stop_prob': 0.05}


def brute_force_ordered(logits, kept, order):
    p, remaining = 1.0, [i for i in range(len(logits)) if kept[i]]
    for i in order:
        w = torch.tensor([logits[j] for j in remaining]).exp()
        p *= (math.exp(logits[i]) / w.sum()).item()
        remaining.remove(i)
    return math.log(p)


def test_removal_log_prob_matches_enumeration():
    logits = torch.tensor([[0.3, -1.0, 2.0, 0.5, 1.1]])
    kept = torch.tensor([[True, True, True, False, True]])
    masked = logits.masked_fill(~kept, NEG)
    total = 0.0
    for order in itertools.permutations([0, 1, 2, 4], 2):
        lp = removal_log_prob(masked, torch.tensor([order]), kept).item()
        assert lp == pytest.approx(brute_force_ordered(logits[0].tolist(), kept[0].tolist(), order), abs=1e-5)
        total += math.exp(lp)
    assert total == pytest.approx(1.0, abs=1e-5)   # a distribution over ordered pairs of kept edges


def test_removal_log_prob_padding():
    logits = torch.tensor([[0.3, -1.0, 2.0, 0.5], [1.0, 0.0, -0.5, 0.2]])
    kept = torch.ones(2, 4, dtype=torch.bool)
    full = removal_log_prob(logits, torch.tensor([[2, 0, 1], [0, 3, 1]]), kept)
    padded = removal_log_prob(logits, torch.tensor([[2, 0, -1], [0, 3, 1]]), kept)
    two = removal_log_prob(logits[:1], torch.tensor([[2, 0]]), kept[:1])
    assert padded[0].item() == pytest.approx(two.item(), abs=1e-6)
    assert padded[1].item() == pytest.approx(full[1].item(), abs=1e-6)


def test_gumbel_top_m_frequencies():
    logits = torch.tensor([[1.0, 0.0, -1.0, 0.5]]).repeat(40000, 1)
    g = torch.Generator().manual_seed(0)
    order = sample_removal(logits, 1, generator=g)
    freq = torch.bincount(order[:, 0], minlength=4).float() / len(order)
    assert torch.allclose(freq, logits[0].softmax(0), atol=0.01)
    order2 = sample_removal(logits[:20000], 2, generator=g)
    first_two = (order2[:, 0] == 0) & (order2[:, 1] == 3)
    expected = math.exp(removal_log_prob(logits[:1], torch.tensor([[0, 3]]), torch.ones(1, 4, dtype=torch.bool)))
    assert first_two.float().mean().item() == pytest.approx(expected, abs=0.01)
    greedy = sample_removal(logits[:1], 2, greedy=True)
    assert greedy.tolist() == [[0, 3]]


def test_removal_entropy_range():
    kept = torch.tensor([[True, True, True, True], [True, True, False, False]])
    uniform = torch.zeros(2, 4)
    h = removal_entropy(uniform, kept)
    assert torch.allclose(h, torch.ones(2), atol=1e-5)
    peaked = torch.tensor([[50.0, 0, 0, 0], [50.0, 0, 0, 0]])
    assert (removal_entropy(peaked, kept) < 1e-3).all()


@pytest.fixture
def policy_setup(fc21):
    edge_list, X = fc21
    torch.manual_seed(0)
    policy = ActorCritic(POLICY_CFG, 21, 3, torch.tensor(edge_list), n_steps=20)
    edge_vals = torch.tensor(X)
    demo = torch.randn(len(X), 3)
    return policy, edge_vals, demo, len(edge_list)


def test_act_and_evaluate_agree(policy_setup):
    policy, edge_vals, demo, E = policy_setup
    mask = torch.ones(len(edge_vals), E, dtype=torch.bool)
    mask[:, :20] = False
    g = torch.Generator().manual_seed(0)
    stop, order, logp, value = policy.act(edge_vals, mask, 2, demo, 10, generator=g)
    assert order.shape == (len(edge_vals), 10) and value.shape == (len(edge_vals),)
    assert (order >= 20).all()                                     # never picks an already-removed edge
    assert all(len(set(r)) == 10 for r in order.tolist())          # without replacement
    lp2, ent, v2 = policy.evaluate_actions(edge_vals, mask, torch.full((len(edge_vals),), 2), demo, stop, order)
    assert torch.allclose(logp, lp2, atol=1e-5) and torch.allclose(value, v2, atol=1e-5)
    assert torch.isfinite(ent).all()
    # Initial STOP probability is close to init_stop_prob
    _, stop_logit = policy.policy_logits(edge_vals, mask, 2, demo)
    assert torch.allclose(torch.sigmoid(stop_logit), torch.full_like(stop_logit, 0.05), atol=0.02)


def test_forced_last_step_and_gradients(policy_setup):
    policy, edge_vals, demo, E = policy_setup
    B = len(edge_vals)
    mask = torch.zeros(B, E, dtype=torch.bool)
    mask[:, :10] = True                       # last step: remove all 10 remaining edges
    stop = torch.zeros(B, dtype=torch.bool)
    order = torch.arange(10).repeat(B, 1)
    _, stop_logit = policy.policy_logits(edge_vals, mask, 19, demo)
    logp, ent, value = policy.evaluate_actions(edge_vals, mask, torch.full((B,), 19), demo, stop, order)
    assert torch.allclose(logp, torch.nn.functional.logsigmoid(-stop_logit), atol=1e-6)
    # Mixed minibatch (different steps, padded orders) backpropagates finite gradients
    mask2 = torch.ones(B, E, dtype=torch.bool)
    order2 = torch.full((B, 12), -1)
    order2[:, :10] = torch.arange(10)
    t = torch.tensor([0, 19] * (B // 2))
    m = torch.where(t[:, None] == 19, mask, mask2)
    stop2 = torch.tensor([True, False] * (B // 2))
    lp, ent, v = policy.evaluate_actions(edge_vals, m, t, demo, stop2, order2)
    (lp.sum() + ent.sum() + v.sum()).backward()
    for p in policy.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
