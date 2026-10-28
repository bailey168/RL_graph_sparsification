import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import parse_region_map
from topk.sparsify import num_kept_edges, topk_edge_indices

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
