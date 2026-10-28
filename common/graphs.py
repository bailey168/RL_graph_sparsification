import numpy as np
import torch
from torch_geometric.data import Data


def signed_gcn_norm(src, dst, w, num_nodes, self_loop_weight=1.0):
    """Symmetric GCN normalization that keeps edge signs.

    Degrees use |w| so they stay positive: deg_i = s + sum_j |w_ij|, and
    w_hat_ij = w_ij / sqrt(deg_i * deg_j). Input edges are undirected (one entry per pair);
    the output contains both directions plus self-loops (if self_loop_weight > 0).
    """
    abs_w = np.abs(w)
    deg = np.full(num_nodes, self_loop_weight, dtype=np.float64)
    deg += np.bincount(src, weights=abs_w, minlength=num_nodes)
    deg += np.bincount(dst, weights=abs_w, minlength=num_nodes)
    with np.errstate(divide='ignore'):
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    src_all = [src, dst]
    dst_all = [dst, src]
    w_all = [w, w]
    if self_loop_weight > 0:
        nodes = np.arange(num_nodes)
        src_all.append(nodes)
        dst_all.append(nodes)
        w_all.append(np.full(num_nodes, self_loop_weight))
    src_all = np.concatenate(src_all)
    dst_all = np.concatenate(dst_all)
    w_all = np.concatenate(w_all)

    edge_weight = deg_inv_sqrt[src_all] * w_all * deg_inv_sqrt[dst_all]
    edge_index = np.stack([src_all, dst_all])
    return edge_index, edge_weight


def build_graphs(edge_vals, top_idx, demo, y, eids, edge_list, num_nodes, self_loop_weight=1.0):
    """Build one PyG Data object per subject from its kept edges.

    edge_vals: (N, E) signed values carried by edges; top_idx: (N, k) kept edge indices;
    demo: (N, D); y: (N,) scaled target; edge_list: (E, 2) node index pairs.
    """
    x_node = torch.arange(num_nodes, dtype=torch.long)
    demo_t = torch.from_numpy(demo).float().unsqueeze(1)   # (N, 1, D)
    y_t = torch.from_numpy(y).float().unsqueeze(1)          # (N, 1)
    src_global, dst_global = edge_list[:, 0], edge_list[:, 1]

    data_list = []
    for i in range(len(eids)):
        idx = top_idx[i]
        edge_index, edge_weight = signed_gcn_norm(
            src_global[idx], dst_global[idx], edge_vals[i, idx].astype(np.float64),
            num_nodes, self_loop_weight
        )
        data_list.append(Data(
            x=x_node,
            edge_index=torch.from_numpy(edge_index).long(),
            edge_weight=torch.from_numpy(edge_weight).float(),
            demographics=demo_t[i],
            y=y_t[i],
            eid=torch.tensor([int(eids[i])], dtype=torch.long),
            num_nodes=num_nodes,
        ))
    return data_list


def dense_gcn_adj(edge_vals, edge_list, num_nodes, self_loop_weight=1.0, mask=None):
    """Batched torch version of signed_gcn_norm, returning dense (B, N, N) normalized adjacencies.

    edge_vals: (B, E) signed values; edge_list: (E, 2) long tensor of node pairs (one per undirected
    edge); mask: optional (B, E) keep mask (bool or 0/1). Dropped edges are zeroed, which is
    equivalent to removing them. Runs on whatever device edge_vals lives on.
    """
    w = edge_vals if mask is None else edge_vals * mask.to(edge_vals.dtype)
    B = w.shape[0]
    src, dst = edge_list[:, 0], edge_list[:, 1]
    adj = w.new_zeros(B, num_nodes, num_nodes)
    adj[:, src, dst] = w
    adj[:, dst, src] = w
    deg = self_loop_weight + adj.abs().sum(dim=-1)                     # (B, N)
    deg_inv_sqrt = torch.where(deg > 0, deg.clamp_min(1e-12).rsqrt(), torch.zeros_like(deg))
    if self_loop_weight > 0:
        adj = adj + self_loop_weight * torch.eye(num_nodes, device=w.device, dtype=w.dtype)
    return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
