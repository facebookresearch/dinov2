import torch


def label_graph(gathered_targets, n_views=2, device=None):
    targets = gathered_targets.repeat(n_views).unsqueeze(1)  # [N_global, 1
    G = (targets == targets.T).bool()  # [N_global*N_global]

    if device is not None:
        G = G.to(device)
    return G


def semisup_graph(
    labels_graph, view_graph, gathered_is_supervised, n_views=2, device=None
):
    is_supervised = gathered_is_supervised.repeat(n_views).unsqueeze(1).bool()
    mask = is_supervised & is_supervised.T
    G = labels_graph.bool() & mask  # keep only the semisupervised graph
    G[view_graph.bool()] = True  # add the view graph edges
    if device is not None:
        G = G.to(device)
    return G


def nview_graph(batch_size, n_views=2, device=None):
    size = batch_size * n_views
    G = torch.zeros((size, size), dtype=torch.bool)
    i = torch.arange(0, size).repeat_interleave(n_views - 1)
    j = (i + torch.arange(1, n_views).repeat(size) * batch_size).remainder(size)
    G[i, j] = 1
    G.fill_diagonal_(1)
    if device is not None:
        G = G.to(device)
    return G
