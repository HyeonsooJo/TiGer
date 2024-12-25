from torch_sparse import SparseTensor

import numpy as np
import torch


def add_new_graph(prev_purified_adj, noise_adj, prev_noise_adj):
    num_nodes = noise_adj.size(0)

    new_edge_set = set()
    curr_edge_set = set()
    for src, dst in zip(noise_adj.coo()[0], noise_adj.coo()[1]):
        src, dst = src.item(), dst.item()
        if src >= dst: continue
        new_edge_set.add((src, dst))
        curr_edge_set.add((src, dst))
    for src, dst in zip(prev_noise_adj.coo()[0], prev_noise_adj.coo()[1]):
        src, dst = src.item(), dst.item()
        if src >= dst: continue
        new_edge_set.remove((src, dst))
        curr_edge_set.remove((src, dst))
    
    for src, dst in zip(prev_purified_adj.coo()[0], prev_purified_adj.coo()[1]):
        src, dst = src.item(), dst.item()
        if src >= dst: continue
        curr_edge_set.add((src, dst))

    curr_edges = np.array([[src, dst] for src, dst in curr_edge_set]).T
    row = torch.LongTensor(np.concatenate([curr_edges[0], curr_edges[1]]))
    col = torch.LongTensor(np.concatenate([curr_edges[1], curr_edges[0]]))
    unpurified_adj = SparseTensor(row=row, col=col, sparse_sizes=[num_nodes, num_nodes])
    return unpurified_adj, new_edge_set
