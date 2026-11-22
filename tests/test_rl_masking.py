"""RL masks, masked batches and MaskedLoader."""
import numpy as np
import torch
from torch_geometric.data import Batch

from common.graphs import build_graphs
from common.models import GATv2Net
from rl.masking import (MaskedLoader, SubjectTensors, level_masks, masked_batch, random_masks, removal_schedule,
                        stratified_masks)
from topk.sparsify import num_kept_edges


def test_removal_schedule_matches_topk_rule():
    for E in (210, 1485, 45):
        sched = removal_schedule(E, 20)
        assert len(sched) == 21 and sched[0] == E and sched[-1] == 0
        assert all(a > b for a, b in zip(sched, sched[1:]))
        for t, k in enumerate(sched[:-1]):
            assert k == max(1, (20 - t) * E // 20)
            # Same as top-K's k wherever float rounding doesn't bite
            assert abs(k - num_kept_edges(E, (20 - t) / 20)) <= 1
    assert removal_schedule(1485)[1] == 1410   # FC55: 75 removed at the first step


def test_random_and_stratified_mask_counts():
    E, g = 210, torch.Generator().manual_seed(0)
    sched = set(removal_schedule(E))
    m = random_masks(500, E, generator=g)
    assert m.dtype == torch.bool and m.shape == (500, E)
    assert set(m.sum(1).tolist()) <= sched
    s = stratified_masks(42, E, generator=g)
    counts = sorted(s.sum(1).tolist())
    assert counts == sorted(removal_schedule(E) * 2)   # every level exactly twice
    lm = level_masks(10, E, 3, generator=g)
    assert (lm.sum(1) == removal_schedule(E)[3]).all()
    # Masks differ across rows (random edges, not a fixed subset)
    assert (lm.float().std(0) > 0).any()


def make_subjects(fc21, n=8, demo_dim=3):
    edge_list, X = fc21
    rng = np.random.default_rng(1)
    demo = rng.normal(size=(n, demo_dim)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    eids = np.arange(100, 100 + n)
    return SubjectTensors(torch.tensor(X[:n]), torch.tensor(demo), torch.tensor(y), torch.tensor(eids)), demo, y, eids


def test_masked_batch_matches_build_graphs(fc21):
    edge_list, X = fc21
    subjects, demo, y, eids = make_subjects(fc21)
    E, N = len(edge_list), 21
    g = torch.Generator().manual_seed(0)
    k = 60
    mask = level_masks(len(eids), E, 0, generator=g)
    mask[:] = False
    top_idx = np.stack([np.sort(torch.randperm(E, generator=g)[:k].numpy()) for _ in eids])
    for i, idx in enumerate(top_idx):
        mask[i, idx] = True

    batch = masked_batch(subjects.edge_vals, mask, torch.tensor(edge_list), N, subjects.demo, subjects.y, subjects.eid)
    ref = Batch.from_data_list(build_graphs(X[:len(eids)], top_idx, demo, y, eids, edge_list, N,
                                            graph_format='edge_attr'))
    as_set = lambda ei, ea: sorted(zip(ei[0].tolist(), ei[1].tolist(), ea.view(-1).tolist()))
    assert as_set(batch.edge_index, batch.edge_attr) == as_set(ref.edge_index, ref.edge_attr)
    assert torch.equal(batch.batch, ref.batch) and torch.equal(batch.x, ref.x)

    torch.manual_seed(0)
    model = GATv2Net(N, 8, 16, 2, [8], 0.0, 0.0, heads=2, demo_dim=demo.shape[1]).eval()
    with torch.no_grad():
        assert torch.allclose(model(batch), model(ref), atol=1e-5)


def test_masked_batch_variable_counts_and_empty(fc21):
    edge_list, _ = fc21
    subjects, *_ = make_subjects(fc21, n=3)
    mask = torch.zeros(3, len(edge_list), dtype=torch.bool)
    mask[0, :5] = True        # 5 edges, then empty, then all
    mask[2] = True
    b = masked_batch(subjects.edge_vals, mask, torch.tensor(edge_list), 21, subjects.demo)
    assert b.edge_index.shape[1] == 2 * (5 + len(edge_list))
    graph_of_edge = b.edge_index[0] // 21
    assert set(graph_of_edge.tolist()) == {0, 2}
    model = GATv2Net(21, 8, 16, 2, [8], 0.0, 0.0, heads=2, demo_dim=3).eval()
    with torch.no_grad():
        assert torch.isfinite(model(b)).all()


def test_masked_loader_random_and_fixed(fc21):
    edge_list, _ = fc21
    subjects, *_ = make_subjects(fc21)
    E = len(edge_list)
    loader = MaskedLoader(subjects, torch.tensor(edge_list), 21, batch_size=3, shuffle=True,
                          generator=torch.Generator().manual_seed(0), mask_generator=torch.Generator().manual_seed(1))
    batches = list(loader)
    assert len(loader) == len(batches) == 3 and len(loader.dataset) == 8
    assert sorted(torch.cat([b.eid for b in batches]).tolist()) == subjects.eid.tolist()
    sched = set(removal_schedule(E))
    for b in batches:
        counts = torch.bincount(b.edge_index[0] // 21, minlength=b.num_graphs) // 2
        assert set(counts.tolist()) <= sched
    fixed = torch.zeros(8, E, dtype=torch.bool)
    fixed[:, :7] = True
    b = next(iter(MaskedLoader(subjects, torch.tensor(edge_list), 21, batch_size=8, masks=fixed)))
    assert b.edge_index.shape[1] == 8 * 7 * 2
