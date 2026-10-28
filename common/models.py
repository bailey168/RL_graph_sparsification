import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation


class GCNNet(torch.nn.Module):
    """Node-ID embedding -> GCNConv stack -> graph pooling -> (+ demographics) -> MLP head.

    Mirrors GATv2ConvNet in others/gnn/architectures_normal.py. Edge weights are expected to be
    pre-normalized (see graphs.signed_gcn_norm), so GCNConv runs with normalize=False.
    """

    def __init__(self, num_nodes, embedding_dim, hidden_channels, gnn_num_layers,
                 classifier_hidden_dims, node_dropout, classifier_dropout, pooling='attention',
                 include_demo=True, demo_dim=0, demo_embed_dim=8, out_dim=1):
        super().__init__()
        self.gnn_num_layers = gnn_num_layers
        self.node_dropout = node_dropout
        self.include_demo = include_demo
        self.out_dim = out_dim
        self.pooling = pooling

        self.node_embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)

        self.conv_layers = torch.nn.ModuleList([
            GCNConv(embedding_dim if i == 0 else hidden_channels, hidden_channels,
                    normalize=False, add_self_loops=False)
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

    def forward(self, data):
        x = self.node_embedding(data.x)

        for i, conv in enumerate(self.conv_layers):
            x = conv(x, data.edge_index, edge_weight=data.edge_weight)
            if i < self.gnn_num_layers - 1:
                x = self.batch_norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.node_dropout, training=self.training)

        if self.pool is not None:
            graph_features = self.pool(x, data.batch)
        else:
            graph_features = global_mean_pool(x, data.batch)

        if self.include_demo:
            graph_features = torch.cat([graph_features, self.demo_projection(data.demographics)], dim=1)

        out = self.classifier(graph_features)
        return out.squeeze(-1) if self.out_dim == 1 else out

    def forward_dense(self, adj, demographics=None):
        """Same computation as forward(), on a dense normalized adjacency (B, N, N).

        Every graph shares the node set 0..N-1, so the node embedding is one (N, emb) matrix
        broadcast across the batch. Build adj with graphs.dense_gcn_adj; uses the same weights as forward().
        """
        B, N, _ = adj.shape
        x = self.node_embedding.weight

        for i, conv in enumerate(self.conv_layers):
            x = adj @ conv.lin(x) + conv.bias          # (B, N, H); layer 0 broadcasts (N, H)
            if i < self.gnn_num_layers - 1:
                x = self.batch_norms[i](x.reshape(B * N, -1)).reshape(B, N, -1)
                x = F.elu(x)
                x = F.dropout(x, p=self.node_dropout, training=self.training)

        if self.pool is not None:
            gate = torch.softmax(self.pool.gate_nn(x), dim=1)   # (B, N, 1)
            graph_features = (gate * x).sum(dim=1)
        else:
            graph_features = x.mean(dim=1)

        if self.include_demo:
            graph_features = torch.cat([graph_features, self.demo_projection(demographics)], dim=1)

        out = self.classifier(graph_features)
        return out.squeeze(-1) if self.out_dim == 1 else out


def build_model(model_cfg, num_nodes, demo_dim):
    return GCNNet(
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
    )
