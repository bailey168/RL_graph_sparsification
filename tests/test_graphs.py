import os
import sys

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import parse_region_map
from common.graphs import build_graphs, dense_gcn_adj, signed_gcn_norm
from common.models import GCNNet
from topk.sparsify import topk_edge_indices

REGION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'region_names')


@pytest.fixture
def fc21():
    rng = np.random.default_rng(0)
    node_names, edge_list, _ = parse_region_map(os.path.join(REGION_DIR, 'FC21_regions.txt'))
    n_subj = 8
    X = rng.normal(0, 0.5, size=(n_subj, len(edge_list))).astype(np.float32)
    return node_names, edge_list, X


def test_region_maps():
    for name, nodes, edges in [('FC21', 21, 210), ('FC55', 55, 1485)]:
        node_names, edge_list, cols = parse_region_map(os.path.join(REGION_DIR, f'{name}_regions.txt'))
        assert len(node_names) == nodes
        assert edge_list.shape == (edges, 2)
        assert len(set(map(tuple, np.sort(edge_list, axis=1)))) == edges  # no duplicate pairs


@pytest.mark.parametrize('keep_frac', [0.1, 0.5, 1.0])
def test_graph_structure(fc21, keep_frac):
    _, edge_list, X = fc21
    num_nodes = 21
    top_idx = topk_edge_indices(np.abs(X), keep_frac)
    k = top_idx.shape[1]
    graphs = build_graphs(X, top_idx, np.zeros((len(X), 3), np.float32), np.zeros(len(X), np.float32),
                          np.arange(len(X)), edge_list, num_nodes, self_loop_weight=1.0)
    for i, g in enumerate(graphs):
        assert g.edge_index.shape == (2, 2 * k + num_nodes)
        assert torch.isfinite(g.edge_weight).all()
        # signs preserved for the non-self-loop edges
        raw = np.concatenate([X[i, top_idx[i]], X[i, top_idx[i]]])
        assert np.array_equal(np.sign(g.edge_weight[:2 * k].numpy()), np.sign(raw))
        # symmetric: (u, v, w) present iff (v, u, w) present
        fwd = {(u, v): w for u, v, w in zip(*g.edge_index.tolist(), g.edge_weight.tolist())}
        for (u, v), w in fwd.items():
            assert fwd[(v, u)] == pytest.approx(w)


def test_signed_norm_values():
    # Two nodes, one edge of weight -0.5, plus an isolated third node
    ei, ew = signed_gcn_norm(np.array([0]), np.array([1]), np.array([-0.5]), 3, self_loop_weight=1.0)
    deg = np.array([1.5, 1.5, 1.0])
    expected = {(0, 1): -0.5 / 1.5, (1, 0): -0.5 / 1.5, (0, 0): 1 / 1.5, (1, 1): 1 / 1.5, (2, 2): 1.0}
    got = {(u, v): w for u, v, w in zip(*ei.tolist(), ew.tolist())}
    assert got.keys() == expected.keys()
    for key, val in expected.items():
        assert got[key] == pytest.approx(val)
    assert np.all(deg > 0)


def test_no_self_loops_isolated_nodes_finite():
    ei, ew = signed_gcn_norm(np.array([0]), np.array([1]), np.array([0.3]), 4, self_loop_weight=0.0)
    assert ei.shape == (2, 2)
    assert np.isfinite(ew).all()


@pytest.mark.parametrize('pooling,include_demo', [('attention', True), ('mean', False)])
def test_model_forward(fc21, pooling, include_demo):
    _, edge_list, X = fc21
    demo = np.random.default_rng(1).normal(size=(len(X), 5)).astype(np.float32)
    graphs = build_graphs(X, topk_edge_indices(np.abs(X), 0.3), demo, np.zeros(len(X), np.float32),
                          np.arange(len(X)), edge_list, 21)
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs))))
    model = GCNNet(num_nodes=21, embedding_dim=8, hidden_channels=16, gnn_num_layers=2,
                   classifier_hidden_dims=[8], node_dropout=0.1, classifier_dropout=0.1,
                   pooling=pooling, include_demo=include_demo, demo_dim=5, demo_embed_dim=4)
    model.eval()
    out = model(batch)
    assert out.shape == (len(graphs),)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('keep_frac,self_loop_weight', [(0.1, 1.0), (0.5, 0.5), (1.0, 1.0), (0.05, 0.0)])
def test_dense_adj_matches_sparse_norm(fc21, keep_frac, self_loop_weight):
    _, edge_list, X = fc21
    top_idx = topk_edge_indices(np.abs(X), keep_frac)
    mask = np.zeros_like(X, dtype=bool)
    np.put_along_axis(mask, top_idx, True, axis=1)
    adj = dense_gcn_adj(torch.from_numpy(X).double(), torch.from_numpy(edge_list).long(), 21,
                        self_loop_weight, mask=torch.from_numpy(mask))
    for i in range(len(X)):
        ei, ew = signed_gcn_norm(edge_list[top_idx[i], 0], edge_list[top_idx[i], 1],
                                 X[i, top_idx[i]].astype(np.float64), 21, self_loop_weight)
        expected = np.zeros((21, 21))
        np.add.at(expected, (ei[0], ei[1]), ew)
        np.testing.assert_allclose(adj[i].numpy(), expected, atol=1e-12)


@pytest.mark.parametrize('pooling,include_demo,keep_frac', [('attention', True, 0.3), ('mean', False, 1.0),
                                                            ('attention', False, 0.05)])
def test_forward_dense_matches_forward(fc21, pooling, include_demo, keep_frac):
    _, edge_list, X = fc21
    demo = np.random.default_rng(1).normal(size=(len(X), 5)).astype(np.float32)
    top_idx = topk_edge_indices(np.abs(X), keep_frac)
    graphs = build_graphs(X, top_idx, demo, np.zeros(len(X), np.float32), np.arange(len(X)), edge_list, 21)
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs))))
    torch.manual_seed(0)
    model = GCNNet(num_nodes=21, embedding_dim=8, hidden_channels=16, gnn_num_layers=3,
                   classifier_hidden_dims=[8], node_dropout=0.1, classifier_dropout=0.1,
                   pooling=pooling, include_demo=include_demo, demo_dim=5, demo_embed_dim=4)
    # Non-trivial BatchNorm running stats, so eval-mode normalization is actually exercised
    for bn in model.modules():
        if isinstance(bn, torch.nn.BatchNorm1d):
            bn.running_mean.uniform_(-0.5, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
    model.eval()

    mask = np.zeros_like(X, dtype=bool)
    np.put_along_axis(mask, top_idx, True, axis=1)
    adj = dense_gcn_adj(torch.from_numpy(X), torch.from_numpy(edge_list).long(), 21, 1.0,
                        mask=torch.from_numpy(mask))
    with torch.no_grad():
        expected = model(batch)
        got = model.forward_dense(adj, torch.from_numpy(demo))
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
