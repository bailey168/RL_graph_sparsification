"""Device-resident minibatching for fixed-size graphs, as a drop-in replacement for the PyG DataLoader.

Every graph in a split shares the node set 0..N-1 and has the same number of edges m (top-K keeps a fixed k
per subject), so a split can be stored as stacked tensors on the GPU and a minibatch assembled with a few
indexing ops instead of PyG's per-batch Python collation (Batch.from_data_list) plus a host->device copy.

Batches are bit-identical to what torch_geometric.loader.DataLoader yields for the same graphs: the shuffle
order comes from a torch DataLoader over range(n) with the same generator (same RandomSampler, same generator
draws per epoch), and the fields are built the way Batch.from_data_list concatenates them.
"""
import torch
from torch.utils.data import DataLoader as IndexLoader

EDGE_FEATURES = ('edge_attr', 'edge_weight')


class TensorBatch:
    """The fields of a PyG Batch that the models and common/train.py use."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def to(self, device):   # already on the device
        return self


class GraphTensors:
    """One split's graphs (a list of Data from graphs.build_graphs) stacked into tensors on `device`."""

    def __init__(self, graphs, device):
        if not graphs:
            raise ValueError("GraphTensors needs at least one graph")
        first = graphs[0]
        self.num_nodes = int(first.num_nodes)
        if self.num_nodes >= 2 ** 15:
            raise ValueError(f"num_nodes={self.num_nodes} does not fit the int16 edge index")
        num_edges = {g.edge_index.shape[1] for g in graphs}
        if len(num_edges) != 1 or any(int(g.num_nodes) != self.num_nodes for g in graphs):
            raise ValueError("GraphTensors needs graphs with equal node and edge counts; use training.loader=pyg")
        self.edge_feature = next((k for k in EDGE_FEATURES if k in first), None)
        self.device = torch.device(device)

        to = lambda t: t.to(self.device)
        self.edge_index = to(torch.stack([g.edge_index for g in graphs]).to(torch.int16))          # (n, 2, m)
        if self.edge_feature is not None:
            self.edge_values = to(torch.stack([g[self.edge_feature] for g in graphs]))             # (n, m[, F])
        self.demographics = to(torch.cat([g.demographics for g in graphs]))                         # (n, D)
        self.y = to(torch.cat([g.y for g in graphs]))                                               # (n,)
        self.eid = to(torch.cat([g.eid for g in graphs]))                                           # (n,)
        self.node_ids = to(torch.arange(self.num_nodes, dtype=torch.long))

    def __len__(self):
        return self.y.shape[0]

    def batch(self, idx):
        """TensorBatch for graph indices idx (1-D long tensor), matching Batch.from_data_list."""
        idx = idx.to(self.device)
        B, N = idx.numel(), self.num_nodes
        graph_ids = torch.arange(B, device=self.device)
        offsets = (graph_ids * N).view(B, 1, 1)
        edge_index = (self.edge_index[idx].long() + offsets).permute(1, 0, 2).reshape(2, -1)
        fields = dict(
            x=self.node_ids.repeat(B),
            edge_index=edge_index,
            batch=graph_ids.repeat_interleave(N),
            demographics=self.demographics[idx],
            y=self.y[idx],
            eid=self.eid[idx],
            num_graphs=B,
        )
        if self.edge_feature is not None:
            values = self.edge_values[idx]
            fields[self.edge_feature] = values.reshape(-1, *values.shape[2:])
        return TensorBatch(**fields)


class TensorLoader:
    """Iterates TensorBatches over a GraphTensors; same batch order as the PyG DataLoader with these args."""

    def __init__(self, tensors, batch_size, shuffle=False, generator=None):
        self.dataset = tensors
        self.index_loader = IndexLoader(range(len(tensors)), batch_size=batch_size, shuffle=shuffle,
                                        generator=generator)

    def __len__(self):
        return len(self.index_loader)

    def __iter__(self):
        for idx in self.index_loader:
            yield self.dataset.batch(idx)
