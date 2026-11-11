import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import parse_region_map
from common.graphs import build_graphs
from topk.sparsify import (edge_retention, is_redundant, num_kept_edges, shared_edge_indices, tile_mask,
                           topk_edge_indices)

REGION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'region_names')


@pytest.fixture
def fc21():
    rng = np.random.default_rng(0)
    _, edge_list, _ = parse_region_map(os.path.join(REGION_DIR, 'FC21_regions.txt'))
    X = rng.normal(0, 0.5, size=(8, len(edge_list))).astype(np.float32)
    return edge_list, X


def test_topk_selects_largest_abs(fc21):
    _, X = fc21
    scores = np.abs(X)
    top_idx = topk_edge_indices(scores, 0.1)
    k = num_kept_edges(X.shape[1], 0.1)
    assert top_idx.shape == (X.shape[0], k)
    for i in range(X.shape[0]):
        kept_min = scores[i, top_idx[i]].min()
        dropped = np.setdiff1d(np.arange(X.shape[1]), top_idx[i])
        assert kept_min >= scores[i, dropped].max()


def test_full_keep_returns_all_edges(fc21):
    _, X = fc21
    top_idx = topk_edge_indices(np.abs(X), 1.0)
    assert top_idx.shape == X.shape
    assert all(len(np.unique(row)) == X.shape[1] for row in top_idx)


def test_shared_mask_shape(fc21):
    _, X = fc21
    scores = np.abs(X)
    k = num_kept_edges(X.shape[1], 0.2)
    idx = shared_edge_indices(scores, 0.2)
    assert idx.shape == (k,) and len(np.unique(idx)) == k
    assert np.array_equal(shared_edge_indices(scores, 1.0), np.arange(X.shape[1]))
    tiled = tile_mask(idx, 5)
    assert tiled.shape == (5, k) and (tiled == idx).all()


def test_shared_frequency_matches_count(fc21):
    _, X = fc21
    scores = np.abs(X)
    keep = 0.1
    k = num_kept_edges(X.shape[1], keep)
    freq = edge_retention(topk_edge_indices(scores, keep), X.shape[1])
    idx = shared_edge_indices(scores, keep)
    dropped = np.setdiff1d(np.arange(X.shape[1]), idx)
    assert freq[idx].min() >= freq[dropped].max()
    assert len(idx) == k


def test_keep_zero_gives_self_loops_only(fc21):
    edge_list, X = fc21
    num_nodes, n = 21, X.shape[0]
    assert num_kept_edges(X.shape[1], 0.0) == 0
    assert num_kept_edges(X.shape[1], 0.001) == 1   # any positive fraction still keeps >= 1 edge
    top_idx = topk_edge_indices(np.abs(X), 0.0)
    assert top_idx.shape == (n, 0)
    assert shared_edge_indices(np.abs(X), 0.0).shape == (0,)
    assert np.all(edge_retention(top_idx, X.shape[1]) == 0)

    graphs = build_graphs(X, top_idx, np.zeros((n, 3), dtype=np.float32), np.zeros(n), np.arange(n),
                          edge_list, num_nodes, self_loop_weight=1.0)
    for g in graphs:
        assert g.edge_index.shape == (2, num_nodes)
        assert (g.edge_index[0] == g.edge_index[1]).all()             # self-loops only
        assert np.allclose(g.edge_weight.numpy(), 1.0)                # s / deg_i with deg_i = s
        assert (g.edge_index == graphs[0].edge_index).all()           # identical graph for every subject


def test_is_redundant():
    assert is_redundant('shared', 0.0) and is_redundant('shared', 1.0)
    assert not is_redundant('shared', 0.5)
    assert not is_redundant('personal', 0.0) and not is_redundant('personal', 1.0)
