import numpy as np


def num_kept_edges(num_edges, keep_frac):
    """k = max(1, floor(p*E)), except keep_frac == 0 keeps no edges (self-loops only)."""
    if keep_frac <= 0:
        return 0
    return max(1, int(np.floor(keep_frac * num_edges)))


def is_redundant(mask, keep_frac):
    """At 0% and 100% the shared mask gives the same graphs as the personal mask, so it is not run."""
    return mask == 'shared' and (keep_frac <= 0 or keep_frac >= 1)


def topk_edge_indices(scores, keep_frac):
    """Per-subject top-K edge selection.

    scores: (N, E) non-negative edge scores (e.g. |raw FC|).
    Returns (N, k) int array of kept edge indices for each subject. This is the
    interface a learned (e.g. RL) sparsifier should also produce.
    """
    num_subjects, num_edges = scores.shape
    k = num_kept_edges(num_edges, keep_frac)
    if k == 0:
        return np.empty((num_subjects, 0), dtype=np.int64)
    if k >= num_edges:
        return np.tile(np.arange(num_edges), (num_subjects, 1))
    return np.argpartition(scores, -k, axis=1)[:, -k:]


def edge_retention(top_idx, num_edges):
    """Fraction of subjects that kept each edge, shape (E,)."""
    counts = np.bincount(top_idx.ravel().astype(np.int64), minlength=num_edges)
    return counts / top_idx.shape[0]


def shared_edge_indices(train_scores, keep_frac):
    """One top-K edge set for the whole group, derived from training subjects only.

    train_scores: (N_train, E) |raw FC|. Edges are ranked by how often they are in a subject's
    personal top-K (the consensus of the personal rule), with ties broken by mean |w|.
    Returns (k,) int array of kept edge indices.
    """
    num_edges = train_scores.shape[1]
    k = num_kept_edges(num_edges, keep_frac)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    freq = edge_retention(topk_edge_indices(train_scores, keep_frac), num_edges)
    order = np.lexsort((train_scores.mean(axis=0), freq))     # last key is primary
    return np.sort(order[-k:])


def tile_mask(edge_idx, num_subjects):
    """Repeat a shared (k,) edge set into the per-subject (N, k) layout used by build_graphs."""
    return np.tile(edge_idx, (num_subjects, 1))
