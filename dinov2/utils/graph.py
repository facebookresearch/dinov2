import torch


def label_graph(targets, device=None):
    G = (targets.unsqueeze(0) == targets.unsqueeze(1)).bool()  # [N, N]

    if device is not None:
        G = G.to(device)
    return G


def semisup_graph(labels_graph, view_graph, is_supervised, device=None):
    mask = is_supervised & is_supervised.T
    G = labels_graph.bool() & mask  # keep only the semisupervised graph
    G[view_graph.bool()] = True  # add the view graph edges
    if device is not None:
        G = G.to(device)
    return G


def nview_graph(batch_size, device=None):
    size = batch_size 
    G = torch.zeros((size, size), dtype=torch.bool)
    G.fill_diagonal_(1)
    if device is not None:
        G = G.to(device)
    return G