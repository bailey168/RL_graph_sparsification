"""PPO actor-critic for iterative edge pruning.

State: the current pruned graph (kept edges carrying z-scored FC values), the step t and the subject's demographics.
Each trunk (actor, critic) embeds node IDs and runs GATv2 layers over the kept edges (edge value as a 1-d edge_attr).
Demographics and t/n_steps are NOT node features: each trunk builds a per-subject context [projected demo, t/n_steps]
that is used only at the heads (both are one global value per subject, so message passing adds nothing).

Action: STOP, or remove m edges (m = the next 5% of the original edges). The actor gives a STOP logit (pooled graph
embedding ⊕ context) and a removal logit per kept edge (symmetric MLP of [h_i + h_j, h_i ⊙ h_j, w_ij, context]).
The critic's value head sees its own pooled graph embedding ⊕ context.
The m edges are drawn without replacement from a Plackett–Luce distribution over the removal logits (Gumbel-top-m),
and the log-probability of the ordered draw is the PPO surrogate:
    log π = log(1 - p_stop) + Σ_j [s_(j) - logsumexp(s over kept edges not yet drawn)],   or log p_stop for STOP.
When m equals the number of kept edges (the last step) the removal is forced and contributes log-prob 0.
Greedy action: STOP if p_stop >= 0.5, else remove the m highest-logit edges.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation

from rl.masking import masked_batch

NEG = -1e9   # logit for edges that cannot be removed (finite, so no NaN gradients)


class Trunk(nn.Module):
    """Node-ID embedding -> GATv2 stack (LayerNorm, ELU) -> node embeddings, attention-pooled graph and the
    head-only context [projected demographics, t]."""

    def __init__(self, num_nodes, demo_dim, embedding_dim, demo_embed_dim, hidden_channels, heads, gnn_num_layers):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.node_embedding.weight, std=0.1)
        self.demo_projection = nn.Linear(demo_dim, demo_embed_dim)
        in_dim = embedding_dim   # node ID only (demographics and t go to the heads)
        self.convs = nn.ModuleList([
            GATv2Conv(in_dim if i == 0 else hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=1,
                      residual=True, add_self_loops=True, fill_value='mean')
            for i in range(gnn_num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(gnn_num_layers)])
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ELU(), nn.Linear(hidden_channels // 2, 1)))
        self.context_dim = demo_embed_dim + 1

    def forward(self, edge_vals, mask, t_frac, demo, edge_list):
        """Returns node embeddings (B, N, H), graph embedding (B, H) and context [demo, t_frac] (B, De + 1)."""
        B, N = mask.shape[0], self.num_nodes
        context = torch.cat([self.demo_projection(demo), t_frac.unsqueeze(1)], dim=1)
        x = self.node_embedding.weight.expand(B, N, -1).reshape(B * N, -1)
        g = masked_batch(edge_vals, mask, edge_list, N, demo)
        for conv, norm in zip(self.convs, self.norms):
            x = F.elu(norm(conv(x, g.edge_index, edge_attr=g.edge_attr)))
        return x.view(B, N, -1), self.pool(x, g.batch, dim_size=B), context


def mlp(in_dim, hidden, out_dim=1, layers=2):
    mods, d = [], in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.ELU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def removal_log_prob(logits, order, kept):
    """Plackett–Luce log-probability of drawing `order` (B, M; -1 = padding at the end) in that order from `kept`.

    logits: (B, E) removal logits (only kept entries matter); kept: (B, E) bool.
    Denominator of draw j: logsumexp over kept edges not drawn before j = logaddexp(never-drawn mass, suffix of the
    drawn logits from j on).
    """
    valid = order >= 0
    idx = order.clamp(min=0)
    sel = logits.gather(1, idx).masked_fill(~valid, NEG)
    drawn = torch.zeros_like(logits).scatter_add(1, idx, valid.to(logits.dtype)) > 0
    rest = logits.masked_fill(~kept | drawn, NEG).logsumexp(dim=1, keepdim=True)
    suffix = sel.flip(1).logcumsumexp(dim=1).flip(1)
    return ((sel - torch.logaddexp(rest, suffix)) * valid).sum(dim=1)


def removal_entropy(logits, kept):
    """Entropy of one removal draw (softmax over kept edges), divided by log(#kept) so it lies in [0, 1]."""
    logp = logits.masked_fill(~kept, NEG).log_softmax(dim=1)
    h = -(logp.exp() * logp).masked_fill(~kept, 0.0).sum(dim=1)
    return h / torch.log(kept.sum(dim=1).clamp(min=2).to(logits.dtype))


def bernoulli_entropy(logit):
    p = torch.sigmoid(logit)
    return -(p * F.logsigmoid(logit) + (1 - p) * F.logsigmoid(-logit))


def sample_removal(logits, m, greedy=False, generator=None):
    """(B, m) edge indices in draw order: Gumbel-top-m (= a Plackett–Luce sample), or the top-m logits if greedy."""
    keys = logits
    if not greedy:
        u = torch.rand(logits.shape, generator=generator, device=logits.device).clamp_(1e-10, 1 - 1e-10)
        keys = logits - torch.log(-torch.log(u))
    return keys.topk(m, dim=1).indices


class ActorCritic(nn.Module):
    def __init__(self, policy_cfg, num_nodes, demo_dim, edge_list, n_steps):
        super().__init__()
        c = policy_cfg
        self.n_steps = n_steps
        self.register_buffer('edge_list', edge_list.clone(), persistent=False)
        trunk = lambda: Trunk(num_nodes, demo_dim, c['embedding_dim'], c['demo_embed_dim'], c['hidden_channels'],
                              c['heads'], c['gnn_num_layers'])
        self.actor, self.critic = trunk(), trunk()
        H, ctx = c['hidden_channels'], self.actor.context_dim
        self.edge_head = mlp(2 * H + 1 + ctx, c['mlp_hidden'], layers=3)
        self.stop_head = mlp(H + ctx, c['mlp_hidden'])
        self.value_head = mlp(H + ctx, c['mlp_hidden'])
        p0 = c['init_stop_prob']
        with torch.no_grad():   # start near p_stop = init_stop_prob regardless of the input
            self.stop_head[-1].weight.mul_(0.01)
            self.stop_head[-1].bias.fill_(math.log(p0 / (1 - p0)))

    def _t_frac(self, t, B, device):
        t = torch.as_tensor(t, device=device)
        return (t.expand(B) if t.dim() == 0 else t).float() / self.n_steps

    def policy_logits(self, edge_vals, mask, t, demo):
        """(removal logits (B, E) with NEG at removed edges, STOP logit (B,))."""
        B = mask.shape[0]
        h, g, context = self.actor(edge_vals, mask, self._t_frac(t, B, mask.device), demo, self.edge_list)
        b, e = mask.nonzero(as_tuple=True)
        hi, hj = h[b, self.edge_list[e, 0]], h[b, self.edge_list[e, 1]]
        scores = self.edge_head(torch.cat([hi + hj, hi * hj, edge_vals[b, e].unsqueeze(1), context[b]], dim=1)).squeeze(1)
        logits = edge_vals.new_full(mask.shape, NEG).index_put((b, e), scores)
        return logits, self.stop_head(torch.cat([g, context], dim=1)).squeeze(1)

    def value(self, edge_vals, mask, t, demo):
        B = mask.shape[0]
        _, g, context = self.critic(edge_vals, mask, self._t_frac(t, B, mask.device), demo, self.edge_list)
        return self.value_head(torch.cat([g, context], dim=1)).squeeze(1)

    @torch.no_grad()
    def act(self, edge_vals, mask, t, demo, m, greedy=False, generator=None, with_value=True):
        """One action per row, all rows at the same step t removing m edges.

        Returns stop (B,) bool, order (B, m) long, log-prob (B,) and value (B,) (None if with_value=False).
        """
        logits, stop_logit = self.policy_logits(edge_vals, mask, t, demo)
        if greedy:
            stop = stop_logit >= 0
        else:
            u = torch.rand(stop_logit.shape, generator=generator, device=stop_logit.device)
            stop = u < torch.sigmoid(stop_logit)
        order = sample_removal(logits, m, greedy, generator)
        logp = self._log_prob(logits, stop_logit, mask, stop, order)
        value = self.value(edge_vals, mask, t, demo) if with_value else None
        return stop, order, logp, value

    def _log_prob(self, logits, stop_logit, mask, stop, order):
        forced = (order >= 0).sum(1) >= mask.sum(1)   # removing every kept edge: no choice
        logp_remove = torch.where(forced, torch.zeros_like(stop_logit), removal_log_prob(logits, order, mask))
        return torch.where(stop, F.logsigmoid(stop_logit), F.logsigmoid(-stop_logit) + logp_remove)

    def evaluate_actions(self, edge_vals, mask, t, demo, stop, order):
        """Log-prob, entropy bonus (H(STOP) + normalised H(one removal draw)) and value for stored transitions.

        order: (B, M) padded with -1 (rows from different steps remove different numbers of edges).
        """
        logits, stop_logit = self.policy_logits(edge_vals, mask, t, demo)
        logp = self._log_prob(logits, stop_logit, mask, stop, order)
        forced = (order >= 0).sum(1) >= mask.sum(1)
        entropy = bernoulli_entropy(stop_logit) + torch.where(forced, torch.zeros_like(stop_logit),
                                                              removal_entropy(logits, mask))
        return logp, entropy, self.value(edge_vals, mask, t, demo)
