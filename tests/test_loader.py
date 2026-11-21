"""The device-resident tensor loader must reproduce the PyG DataLoader exactly (batches and training)."""
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from common.graphs import build_graphs
from common.loader import GraphTensors, TensorLoader
from common.models import build_model
from common.train import fit
from common.utils import set_deterministic_training
from topk.run_sweep import make_loaders, stage_graphs
from topk.sparsify import shared_edge_indices, tile_mask, topk_edge_indices

FORMATS = {'gcn': 'gcn_norm', 'gatv2': 'edge_attr'}


def _graphs(fc21, n, keep_frac, mask, graph_format, seed=0):
    edge_list, _ = fc21
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.5, size=(n, len(edge_list))).astype(np.float32)
    demo = rng.normal(size=(n, 5)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    if mask == 'personal':
        top_idx = topk_edge_indices(np.abs(X), keep_frac)
    else:
        top_idx = tile_mask(shared_edge_indices(np.abs(X), keep_frac), n)
    return build_graphs(X, top_idx, demo, y, np.arange(1000, 1000 + n), edge_list, 21, 1.0,
                        graph_format=graph_format)


FIELDS = ('x', 'edge_index', 'batch', 'demographics', 'y', 'eid')


@pytest.mark.parametrize('graph_format', ['gcn_norm', 'edge_attr'])
@pytest.mark.parametrize('keep_frac,mask', [(0.0, 'personal'), (0.5, 'personal'), (0.5, 'shared'), (1.0, 'personal')])
def test_batches_match_pyg(fc21, graph_format, keep_frac, mask):
    graphs = _graphs(fc21, 37, keep_frac, mask, graph_format)
    tensors = GraphTensors(graphs, 'cpu')
    feat = 'edge_weight' if graph_format == 'gcn_norm' else 'edge_attr'
    pyg = DataLoader(graphs, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(3))
    ours = TensorLoader(tensors, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(3))
    assert len(ours) == len(pyg) and len(ours.dataset) == len(pyg.dataset)
    for _ in range(2):   # the shuffle order must also match across epochs
        for a, b in zip(pyg, ours, strict=True):
            assert a.num_graphs == b.num_graphs
            for f in FIELDS + (feat,):
                assert a[f].dtype == getattr(b, f).dtype, f
                assert torch.equal(a[f], getattr(b, f)), f


def test_unequal_edge_counts_rejected(fc21):
    graphs = _graphs(fc21, 4, 0.5, 'personal', 'edge_attr')
    graphs[0] = _graphs(fc21, 1, 0.2, 'personal', 'edge_attr')[0]
    with pytest.raises(ValueError, match='equal node and edge counts'):
        GraphTensors(graphs, 'cpu')


def _train_cfg(loader):
    return {'batch_size': 16, 'max_epochs': 6, 'lr': 0.01, 'weight_decay': 0.01, 'warmup_epochs': 2,
            'early_stop_patience': 3, 'early_stop_min_delta': 0.0, 'compile': False, 'num_workers': 0,
            'loader': loader}


def _model_cfg(kind):
    return {'type': kind, 'heads': 2, 'embedding_dim': 8, 'hidden_channels': 16, 'gnn_num_layers': 2,
            'classifier_hidden_dims': [8], 'node_dropout': 0.3, 'edge_dropout': 0.2, 'classifier_dropout': 0.3,
            'pooling': 'attention', 'include_demo': True, 'demo_embed_dim': 4}


def _fit(graphs, kind, loader):
    train_cfg = _train_cfg(loader)
    staged = stage_graphs(graphs, train_cfg, torch.device('cpu'))
    generator = set_deterministic_training(0)
    loaders = make_loaders(staged, train_cfg['batch_size'], 0, generator)
    model = build_model(_model_cfg(kind), 21, demo_dim=5)
    y_pre = StandardScaler().fit(np.random.default_rng(0).normal(size=(50, 1)))
    result = fit(model, loaders, train_cfg, torch.device('cpu'), y_pre)
    return model, result


@pytest.mark.parametrize('kind', ['gcn', 'gatv2'])
@pytest.mark.parametrize('keep_frac', [0.0, 0.3])
def test_fit_identical_to_pyg(fc21, kind, keep_frac):
    full = _graphs(fc21, 90, keep_frac, 'personal', FORMATS[kind])
    graphs = {'train': full[:60], 'val': full[60:75], 'test': full[75:]}
    model_a, a = _fit(graphs, kind, 'pyg')
    model_b, b = _fit(graphs, kind, 'tensor')
    assert a['history'] == b['history']
    assert a['best_epoch'] == b['best_epoch'] and a['metrics'] == b['metrics']
    for name in ('eid', 'actual', 'predicted'):
        assert np.array_equal(a['test_predictions'][name], b['test_predictions'][name])
    sa, sb = model_a.state_dict(), model_b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_unknown_loader(fc21):
    with pytest.raises(ValueError, match='training.loader'):
        stage_graphs({}, {'loader': 'nope'}, torch.device('cpu'))
