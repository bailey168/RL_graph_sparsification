import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import JumpingKnowledge
import torch.nn.functional as F

# GATv2Conv
class GATv2ConvNet(torch.nn.Module):
    def __init__(self, out_dim, in_channels, gnn_num_layers, classifier_hidden_dims,
                hidden_channels, embedding_dim, demo_dim, edge_dropout_rate, node_dropout_rate,
                hidden_heads, demo_embed_dim, classifier_dropout_rate, bias=True, include_demo=True, 
                concat=False, include_jk=False):
        super(GATv2ConvNet, self).__init__()
        self.gnn_num_layers = gnn_num_layers
        # Store hidden dims instead of number of layers
        self.classifier_hidden_dims = classifier_hidden_dims
        self.out_dim = out_dim  
        self.include_demo = include_demo
        self.demo_dim = demo_dim
        self.edge_dropout_rate = edge_dropout_rate
        self.node_dropout_rate = node_dropout_rate
        self.classifier_dropout_rate = classifier_dropout_rate
        self.demo_embed_dim = demo_embed_dim
        self.concat = concat
        self.include_jk = include_jk

        self.node_embedding = torch.nn.Embedding(
            num_embeddings=in_channels,
            embedding_dim=embedding_dim
        )

        conv_out_channels = hidden_channels // hidden_heads if self.concat else hidden_channels

        # All conv layers have same input/output dimensions
        self.conv_layers = torch.nn.ModuleList([
            GATv2Conv(
                in_channels=embedding_dim,
                out_channels=conv_out_channels,
                heads=hidden_heads,
                bias=bias,
                edge_dim=1,
                residual=True,
                dropout=self.edge_dropout_rate,
                concat=self.concat
            )] + \
        [
            GATv2Conv(
                in_channels=hidden_channels,
                out_channels=conv_out_channels,
                heads=hidden_heads,
                bias=bias,
                edge_dim=1,
                residual=True,
                dropout=self.edge_dropout_rate,
                concat=self.concat
            ) for _ in range(self.gnn_num_layers - 1)
        ])

        # Determine how many normalization/activation layers block to use
        num_norm_layers = self.gnn_num_layers if self.include_jk else self.gnn_num_layers - 1

        if self.include_demo:
            # 1-layer MLP for demographic data at each layer (except the last)
            self.demo_mlps = torch.nn.ModuleList([
                torch.nn.Linear(self.demo_dim, self.demo_embed_dim) for _ in range(self.gnn_num_layers - 1)
            ])

            # 2-layer MLP to downsize concatenated features (except the last)
            self.downsize_linear1 = torch.nn.ModuleList([
                torch.nn.Linear(hidden_channels + self.demo_embed_dim, hidden_channels * 2) for _ in range(self.gnn_num_layers - 1)
            ])
            self.downsize_norms = torch.nn.ModuleList([
                pyg_nn.norm.BatchNorm(hidden_channels * 2) for _ in range(self.gnn_num_layers - 1)
            ])
            self.downsize_linear2 = torch.nn.ModuleList([
                torch.nn.Linear(hidden_channels * 2, hidden_channels) for _ in range(self.gnn_num_layers - 1)
            ])
            
        # Add batch normalization and activation layers
        self.batch_norms = torch.nn.ModuleList([
            pyg_nn.norm.BatchNorm(hidden_channels) for _ in range(num_norm_layers)
        ])
        self.activations = torch.nn.ModuleList([
            torch.nn.ELU() for _ in range(num_norm_layers)
        ])
        
        # Jumping Knowledge setup
        if self.include_jk:
            self.jk = JumpingKnowledge(mode='lstm', channels=hidden_channels, num_layers=self.gnn_num_layers)

        # Use AttentionalAggregation with the final layer output
        attention_gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_channels // 2, 1)
        )
        self.global_attention_pool = AttentionalAggregation(gate_nn=attention_gate)

        # Build dynamic classifier
        classifier_modules = []
        current_in_channels = hidden_channels
        
        for hidden_dim in self.classifier_hidden_dims:
            classifier_modules.extend([
                torch.nn.Linear(current_in_channels, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU(),
                torch.nn.Dropout(self.classifier_dropout_rate)
            ])
            current_in_channels = hidden_dim
            
        classifier_modules.append(torch.nn.Linear(current_in_channels, out_dim))
        self.classifier = torch.nn.Sequential(*classifier_modules)
                
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize Embeddings
        torch.nn.init.normal_(self.node_embedding.weight, mean=0.0, std=0.02)
        
        # List all the custom non-PyG modules to initialize
        custom_modules = [self.classifier, self.global_attention_pool]
        if self.include_demo:
            custom_modules.extend([self.demo_mlps, self.downsize_linear1, self.downsize_linear2])

        # Safely apply Kaiming Initialization to only custom Linear layers
        for custom_mod in custom_modules:
            for module in custom_mod.modules():
                if isinstance(module, torch.nn.Linear):
                    # Use kaiming normal with non-linearity set to 'relu' (works well for ELU too)
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

    def forward(self, data):
        # Extract to local variable to prevent modifying the PyG Data object inplace
        x = self.node_embedding(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        xs = []

        for i in range(self.gnn_num_layers):
            # Apply graph convolution
            x = self.conv_layers[i](x, edge_index, edge_attr=edge_attr)
            
            # Only concatenate demographics for layers except the last one
            if self.include_demo and hasattr(data, 'demographics') and i < self.gnn_num_layers - 1:
                # Process demographics through 1-layer MLP
                demo_features = self.demo_mlps[i](data.demographics)
                
                # Expand demographics to match number of nodes
                nodes_per_graph = torch.bincount(data.batch)
                demo_expanded = torch.repeat_interleave(demo_features, nodes_per_graph, dim=0)
                
                # Concatenate graph features with demographic features
                combined = torch.cat([x, demo_expanded], dim=1)
                
                # Unpacked Downsize MLP with BatchNorm:
                x = self.downsize_linear1[i](combined)
                x = self.downsize_norms[i](x)
                x = F.elu(x)
                x = self.downsize_linear2[i](x)

            # Apply normalization and activation if it's not the last layer OR if JK is enabled
            if i < self.gnn_num_layers - 1 or self.include_jk:
                x = self.batch_norms[i](x)
                x = self.activations[i](x)
                x = F.dropout(x, p=self.node_dropout_rate, training=self.training)
                
            xs.append(x)

        if self.include_jk:
            x = self.jk(xs)

        # Use attentional aggregation instead of global mean pooling
        graph_features = self.global_attention_pool(x, data.batch)
        x = self.classifier(graph_features)

        # Changed for regression:
        if self.out_dim == 1:
            return x.squeeze(-1)  # For single-output regression, return scalar values
        else:
            return x  # For multi-output regression, return raw values (no softmax)