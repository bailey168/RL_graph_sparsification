import numpy as np


def num_kept_edges(num_edges, keep_frac):
    return max(1, int(np.floor(keep_frac * num_edges)))


def topk_edge_indices(scores, keep_frac):
    """Per-subject top-K edge selection.

    scores: (N, E) non-negative edge scores (e.g. |raw FC|).
    Returns (N, k) int array of kept edge indices for each subject. This is the
    interface a learned (e.g. RL) sparsifier should also produce.
    """
    num_subjects, num_edges = scores.shape
    k = num_kept_edges(num_edges, keep_frac)
    if k >= num_edges:
        return np.tile(np.arange(num_edges), (num_subjects, 1))
    return np.argpartition(scores, -k, axis=1)[:, -k:]


def edge_retention(top_idx, num_edges):
    """Fraction of subjects that kept each edge, shape (E,)."""
    counts = np.bincount(top_idx.ravel(), minlength=num_edges)
    return counts / top_idx.shape[0]
