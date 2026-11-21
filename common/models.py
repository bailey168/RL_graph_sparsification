import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation


class _GraphNet(torch.nn.Module):
    """Shared skeleton: node-ID embedding -> conv stack -> graph pooling -> (+ demographics) -> MLP head.

    Mirrors GATv2ConvNet in others/gnn/architectures_normal.py. Subclasses supply the conv layer through
    `make_conv(in_dim, out_dim)` and implement forward(). Module creation order (embedding, convs, norms,
    pool, demographics, head) and parameter names are fixed so existing GCN checkpoints still load.
    """

    def __init__(self, make_conv, num_nodes, embedding_dim, hidden_channels, gnn_num_layers,
                 classifier_hidden_dims, node_dropout, classifier_dropout, pooling='attention',
                 include_demo=True, demo_dim=0, demo_embed_dim=8, out_dim=1, edge_dropout=0.0):
        super().__init__()
        self.gnn_num_layers = gnn_num_layers
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.include_demo = include_demo
        self.out_dim = out_dim
        self.pooling = pooling

        self.node_embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)

        self.conv_layers = torch.nn.ModuleList([
            make_conv(embedding_dim if i == 0 else hidden_channels, hidden_channels)
            for i in range(gnn_num_layers)
        ])
        # Norm + activation after every conv layer except the last
        self.batch_norms = torch.nn.ModuleList([
            pyg_nn.norm.BatchNorm(hidden_channels) for _ in range(gnn_num_layers - 1)
        ])

        if pooling == 'attention':
            attention_gate = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels // 2),
                torch.nn.ELU(),
                torch.nn.Linear(hidden_channels // 2, 1)
            )
            self.pool = AttentionalAggregation(gate_nn=attention_gate)
        elif pooling == 'mean':
            self.pool = None
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        total_features_dim = hidden_channels
        if include_demo:
            self.demo_projection = torch.nn.Linear(demo_dim, demo_embed_dim)
            total_features_dim += demo_embed_dim

        classifier_modules = []
        in_dim = total_features_dim
        for hidden_dim in classifier_hidden_dims:
            classifier_modules.extend([
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU(),
                torch.nn.Dropout(classifier_dropout)
            ])
            in_dim = hidden_dim
        classifier_modules.append(torch.nn.Linear(in_dim, out_dim))
        self.classifier = torch.nn.Sequential(*classifier_modules)

        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.normal_(self.node_embedding.weight, mean=0.0, std=0.02)

        custom_modules = [self.classifier]
        if self.pool is not None:
            custom_modules.append(self.pool)
        if self.include_demo:
            custom_modules.append(self.demo_projection)

        for custom_mod in custom_modules:
            for module in custom_mod.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

    def _post_conv(self, i, x):
        """BatchNorm -> ELU -> dropout after every conv layer except the last. x: (nodes, H)."""
        if i == self.gnn_num_layers - 1:
            return x
        x = F.elu(self.batch_norms[i](x))
        return F.dropout(x, p=self.node_dropout, training=self.training)

    def _pool(self, x, batch):
        return self.pool(x, batch) if self.pool is not None else global_mean_pool(x, batch)

    def _readout(self, graph_features, demographics):
        """(+ projected demographics) -> MLP head."""
        if self.include_demo:
            graph_features = torch.cat([graph_features, self.demo_projection(demographics)], dim=1)
        out = self.classifier(graph_features)
        return out.squeeze(-1) if self.out_dim == 1 else out


class GCNNet(_GraphNet):
    """GCNConv model. Edge weights are expected to be pre-normalized (see graphs.signed_gcn_norm), so
    GCNConv runs with normalize=False and add_self_loops=False."""

    def __init__(self, num_nodes, embedding_dim, hidden_channels, gnn_num_layers,
                 classifier_hidden_dims, node_dropout, classifier_dropout, pooling='attention',
                 include_demo=True, demo_dim=0, demo_embed_dim=8, out_dim=1, edge_dropout=0.0):
        super().__init__(
            lambda i, o: GCNConv(i, o, normalize=False, add_self_loops=False),
            num_nodes, embedding_dim, hidden_channels, gnn_num_layers, classifier_hidden_dims, node_dropout,
            classifier_dropout, pooling, include_demo, demo_dim, demo_embed_dim, out_dim, edge_dropout)

    def drop_edges(self, edge_index, edge_weight):
        """DropEdge as inverted dropout on the normalized weights: during training, each non-self-loop entry
        is zeroed with probability edge_dropout and survivors are scaled by 1/(1-p). Self-loops are always
        kept. Sampled once per forward pass (all conv layers see the same graph)."""
        if not self.training or self.edge_dropout <= 0:
            return edge_weight
        is_edge = edge_index[0] != edge_index[1]
        return torch.where(is_edge, F.dropout(edge_weight, p=self.edge_dropout, training=True), edge_weight)

    def forward(self, data):
        x = self.node_embedding(data.x)
        edge_weight = self.drop_edges(data.edge_index, data.edge_weight)

        for i, conv in enumerate(self.conv_layers):
            x = self._post_conv(i, conv(x, data.edge_index, edge_weight=edge_weight))

        return self._readout(self._pool(x, data.batch), data.demographics)

    def forward_dense(self, adj, demographics=None):
        """Same computation as forward(), on a dense normalized adjacency (B, N, N).

        Every graph shares the node set 0..N-1, so the node embedding is one (N, emb) matrix
        broadcast across the batch. Build adj with graphs.dense_gcn_adj; uses the same weights as forward().
        """
        B, N, _ = adj.shape
        x = self.node_embedding.weight
        if self.training and self.edge_dropout > 0:   # same DropEdge as forward(); diagonal = self-loops kept
            off_diag = ~torch.eye(N, dtype=torch.bool, device=adj.device)
            adj = torch.where(off_diag, F.dropout(adj, p=self.edge_dropout, training=True), adj)

        for i, conv in enumerate(self.conv_layers):
            x = adj @ conv.lin(x) + conv.bias          # (B, N, H); layer 0 broadcasts (N, H)
            x = self._post_conv(i, x.reshape(B * N, -1)).reshape(B, N, -1)   # BatchNorm wants (nodes, H)

        if self.pool is not None:
            gate = torch.softmax(self.pool.gate_nn(x), dim=1)   # (B, N, 1)
            graph_features = (gate * x).sum(dim=1)
        else:
            graph_features = x.mean(dim=1)
        return self._readout(graph_features, demographics)


class GATv2Net(_GraphNet):
    """GATv2Conv model: GATv2ConvNet from others/gnn/architectures_normal.py without inject or jumping knowledge.

    Each edge carries its (z-scored) value as a 1-d `edge_attr`, which enters the attention logits only.
    Graphs come without self-loops (graphs.build_graphs(graph_format='edge_attr')); GATv2Conv adds them,
    using the mean incoming edge_attr as their attribute (0 for isolated nodes, e.g. everywhere at keep 0.0).
    Heads are averaged (concat=False), every layer has a residual connection, and `edge_dropout` is GATv2's
    attention dropout. There is no dense forward: the RL inference path supports GCN only for now.
    """

    def __init__(self, num_nodes, embedding_dim, hidden_channels, gnn_num_layers,
                 classifier_hidden_dims, node_dropout, classifier_dropout, heads=4, pooling='attention',
                 include_demo=True, demo_dim=0, demo_embed_dim=8, out_dim=1, edge_dropout=0.0):
        super().__init__(
            lambda i, o: GATv2Conv(i, o, heads=heads, concat=False, edge_dim=1, residual=True,
                                   dropout=edge_dropout, add_self_loops=True, fill_value='mean'),
            num_nodes, embedding_dim, hidden_channels, gnn_num_layers, classifier_hidden_dims, node_dropout,
            classifier_dropout, pooling, include_demo, demo_dim, demo_embed_dim, out_dim, edge_dropout)
        self.heads = heads

    def forward(self, data):
        x = self.node_embedding(data.x)
        for i, conv in enumerate(self.conv_layers):
            x = self._post_conv(i, conv(x, data.edge_index, edge_attr=data.edge_attr))
        return self._readout(self._pool(x, data.batch), data.demographics)


MODEL_TYPES = {'gcn': GCNNet, 'gatv2': GATv2Net}
GRAPH_FORMATS = {'gcn': 'gcn_norm', 'gatv2': 'edge_attr'}


def model_type(model_cfg):
    kind = model_cfg.get('type', 'gcn')
    if kind not in MODEL_TYPES:
        raise ValueError(f"Unknown model.type '{kind}' (expected one of {sorted(MODEL_TYPES)})")
    return kind


def graph_format(model_cfg):
    """Edge format build_graphs must produce for this model: 'gcn_norm' (signed-normalized edge_weight with
    self-loops) or 'edge_attr' (raw edge values as edge_attr, no self-loops)."""
    return GRAPH_FORMATS[model_type(model_cfg)]


def build_model(model_cfg, num_nodes, demo_dim):
    kind = model_type(model_cfg)
    kwargs = dict(
        num_nodes=num_nodes,
        embedding_dim=model_cfg['embedding_dim'],
        hidden_channels=model_cfg['hidden_channels'],
        gnn_num_layers=model_cfg['gnn_num_layers'],
        classifier_hidden_dims=model_cfg['classifier_hidden_dims'],
        node_dropout=model_cfg['node_dropout'],
        classifier_dropout=model_cfg['classifier_dropout'],
        pooling=model_cfg['pooling'],
        include_demo=model_cfg['include_demo'],
        demo_dim=demo_dim,
        demo_embed_dim=model_cfg['demo_embed_dim'],
        edge_dropout=model_cfg.get('edge_dropout', 0.0),
    )
    if kind == 'gatv2':
        kwargs['heads'] = model_cfg['heads']
    return MODEL_TYPES[kind](**kwargs)
