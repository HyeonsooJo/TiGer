import numpy as np
import torch
import os


def load_dataset(root, graph_name, seed):
    # Load dataset
    data_path = os.path.join(root, graph_name, "graph", f"{graph_name}.pt")
    split_path = os.path.join(root, graph_name, "graph", f"{graph_name}_split_{seed}.pt")

    data = torch.load(data_path)
    adjs = data["adjs"]
    feats = data["feats"]   
    labels = data["labels"]

    split = torch.load(split_path)
    train_idxs = split["train_idxs"]
    valid_idxs = split["valid_idxs"]
    test_idxs = split["test_idxs"]

    return adjs, feats, labels, train_idxs, valid_idxs, test_idxs
