"""Edge masks and masked minibatches for the RL stage.

Every subject's graph has the same E candidate edges (edge_list), so a sparsified graph is just a (E,) bool mask.
masked_batch turns a (B, E) mask on the GPU into the batched `edge_attr` graph GATv2Net expects, with no Python loop
over subjects, and MaskedLoader wraps it as a drop-in loader for common.train.fit.
"""
from dataclasses import dataclass

import torch

from common.loader import TensorBatch


def removal_schedule(num_edges, n_steps=20):
    """Kept-edge counts k_0..k_n after t removal steps: k_t = max(1, floor((n-t)*E/n)), and k_n = 0.

    Same rule as topk.sparsify.num_kept_edges at keep = (n-t)/n, in integer arithmetic (no float rounding).
    Needs num_edges >= n_steps so that every step removes at least one edge.
    """
    if num_edges < n_steps:
        raise ValueError(f"num_edges={num_edges} < n_steps={n_steps}: some steps would remove no edges")
    return [max(1, (n_steps - t) * num_edges // n_steps) if t < n_steps else 0 for t in range(n_steps + 1)]


def masks_for_counts(counts, num_edges, generator=None):
    """(n, E) bool masks keeping counts[i] edges of row i, chosen uniformly at random. counts: (n,) long, on CPU."""
    scores = torch.rand(len(counts), num_edges, generator=generator)
    ranks = scores.argsort(dim=1).argsort(dim=1)                  # each row: a random permutation of 0..E-1
    return ranks < counts.view(-1, 1)


def random_masks(n, num_edges, n_steps=20, generator=None):
    """Mask augmentation: per row, a keep level uniform over the n_steps+1 schedule levels, then random edges."""
    schedule = torch.tensor(removal_schedule(num_edges, n_steps))
    levels = torch.randint(0, n_steps + 1, (n,), generator=generator)
    return masks_for_counts(schedule[levels], num_edges, generator)


def stratified_masks(n, num_edges, n_steps=20, generator=None):
    """Fixed evaluation masks: the n_steps+1 levels are balanced across rows (in random order), random edges."""
    schedule = torch.tensor(removal_schedule(num_edges, n_steps))
    levels = (torch.arange(n) % (n_steps + 1))[torch.randperm(n, generator=generator)]
    return masks_for_counts(schedule[levels], num_edges, generator)


def level_masks(n, num_edges, level, n_steps=20, generator=None):
    """n random masks all at schedule level `level` (keep = k_level edges)."""
    k = removal_schedule(num_edges, n_steps)[level]
    return masks_for_counts(torch.full((n,), k, dtype=torch.long), num_edges, generator)


def masked_batch(edge_vals, mask, edge_list, num_nodes, demographics, y=None, eid=None):
    """Batched GATv2 graph (`edge_attr` format, as graphs.build_graphs(graph_format='edge_attr')) from masks.

    edge_vals: (B, E) signed edge values; mask: (B, E) bool kept edges; edge_list: (E, 2) long node pairs;
    demographics: (B, D). Kept edges appear in both directions with their value as a 1-d edge_attr; no self-loops
    (GATv2Conv adds them). Kept-edge counts may differ between rows. All tensors on one device.
    """
    B = mask.shape[0]
    device = mask.device
    b, e = mask.nonzero(as_tuple=True)
    offsets = b * num_nodes
    src, dst = edge_list[e, 0] + offsets, edge_list[e, 1] + offsets
    w = edge_vals[b, e]
    graph_ids = torch.arange(B, device=device)
    return TensorBatch(
        x=torch.arange(num_nodes, device=device).repeat(B),
        edge_index=torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]),
        edge_attr=torch.cat([w, w]).unsqueeze(-1),
        batch=graph_ids.repeat_interleave(num_nodes),
        demographics=demographics,
        y=y,
        eid=eid,
        num_graphs=B,
    )


@dataclass
class SubjectTensors:
    """One subject set's arrays on the device: edge_vals (n, E), demo (n, D), y (n,) scaled target, eid (n,)."""
    edge_vals: torch.Tensor
    demo: torch.Tensor
    y: torch.Tensor
    eid: torch.Tensor

    def __len__(self):
        return self.y.shape[0]

    def subset(self, idx):
        return SubjectTensors(self.edge_vals[idx], self.demo[idx], self.y[idx], self.eid[idx])

    @classmethod
    def from_prepared(cls, split, device):
        """From one split of common.data.prepare_splits (edge_vals, demo, y, eids numpy arrays)."""
        to = lambda a, dtype: torch.as_tensor(a, dtype=dtype).to(device)
        return cls(to(split['edge_vals'], torch.float32), to(split['demo'], torch.float32),
                   to(split['y'], torch.float32), to(split['eids'].astype('int64'), torch.long))


class MaskedLoader:
    """Minibatches of masked graphs, usable wherever common.train.fit expects a loader.

    masks: fixed (n, E) bool masks (any device), or None to draw a fresh random mask per subject per batch
    (random_masks: mask augmentation). `generator` shuffles; `mask_generator` draws the random masks.
    """

    def __init__(self, data, edge_list, num_nodes, batch_size, shuffle=False, generator=None, masks=None,
                 n_steps=20, mask_generator=None):
        self.dataset = data
        self.edge_list = edge_list
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.masks = None if masks is None else masks.to(data.y.device)
        self.n_steps = n_steps
        self.mask_generator = mask_generator
        self.num_edges = edge_list.shape[0]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        order = torch.randperm(n, generator=self.generator) if self.shuffle else torch.arange(n)
        device = self.dataset.y.device
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size].to(device)
            if self.masks is None:
                mask = random_masks(len(idx), self.num_edges, self.n_steps, self.mask_generator).to(device)
            else:
                mask = self.masks[idx]
            d = self.dataset
            yield masked_batch(d.edge_vals[idx], mask, self.edge_list, self.num_nodes, d.demo[idx], d.y[idx],
                               d.eid[idx])
