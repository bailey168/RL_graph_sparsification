import torch

from common.graphs import dense_gcn_adj


@torch.inference_mode()
def predict_dense(model, edge_vals, edge_list, num_nodes, demographics=None, mask=None,
                  self_loop_weight=1.0, batch_size=None):
    """Scaled predictions for many subjects via GCNNet.forward_dense, with no PyG Data/Batch.

    All tensors should already live on the model's device (keep them resident across calls).
    edge_vals: (B, E); mask: optional (B, E); demographics: (B, D) or None.
    batch_size=None runs everything in one pass; set it only if B×N×N doesn't fit.
    """
    model.eval()
    B = edge_vals.shape[0]
    step = B if batch_size is None else batch_size
    outs = []
    for start in range(0, B, step):
        sl = slice(start, start + step)
        adj = dense_gcn_adj(edge_vals[sl], edge_list, num_nodes, self_loop_weight,
                            mask=None if mask is None else mask[sl])
        outs.append(model.forward_dense(adj, None if demographics is None else demographics[sl]))
    return torch.cat(outs)
