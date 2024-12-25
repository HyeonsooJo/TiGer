from scipy.sparse.linalg import svds

import networkx as nx
import numpy as np


def get_svd_score(adj_coo, srcs, dsts):
    U, S, V = svds(adj_coo, k=100, random_state=0)
    score = (U[srcs] * V.T[dsts] * S).sum(axis=1)
    return score

def get_aa_score(nx_graph, srcs, dsts):
    scores = []
    ebunch = [(src, dst) for src, dst in zip(srcs, dsts)]
    preds = nx.adamic_adar_index(nx_graph, ebunch)
    for u, v, p in preds:
        scores.append(p)
    return np.array(scores)
