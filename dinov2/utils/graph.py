import torch


def label_graph(gathered_targets, n_global_crops, n_local_crops, device=None):
    """
    Returns: [N_local * bs, N_global * bs] bool matrix
    """
    local_labels  = gathered_targets.repeat(n_local_crops).unsqueeze(0)   # [N_local, 1]
    global_labels = gathered_targets.repeat(n_global_crops).unsqueeze(1)  # [1, N_global]
    G = (global_labels == local_labels).bool()  # [N_local * bs, N_global * bs]

    if device is not None:
        G = G.to(device)
    return G


def semisup_graph(
    labels_graph,
    view_graph,
    gathered_is_supervised,
    n_global_crops,
    n_local_crops,
    device=None
):
    
    batch_size = gathered_is_supervised.shape[0]

    local_mask  = gathered_is_supervised.repeat(n_local_crops).unsqueeze(0).bool()
    global_mask = gathered_is_supervised.repeat(n_global_crops).unsqueeze(1).bool()
    mask = global_mask & local_mask  # [N_local * bs, N_global * bs]

    G = labels_graph.bool() & mask
    G[view_graph.bool()] = True

    if device is not None:
        G = G.to(device)
    return G


def nview_graph(batch_size, n_global_crops, n_local_crops, device=None):
    """
    Returns: [N_local * bs, N_global * bs] bool matrix
    """
    n_rows = n_local_crops * batch_size
    n_cols = n_global_crops * batch_size


    local_base_indices = torch.arange(n_rows).remainder(batch_size)
    global_base_indices = torch.arange(n_cols).remainder(batch_size)

    G = (global_base_indices.view(-1, 1) == local_base_indices.view(1, -1))
    if device is not None:
        G = G.to(device)
    return G

