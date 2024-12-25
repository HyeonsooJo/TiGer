import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

from TiGer_agent import TiGer_agent
from models import *
from utils import *

from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import csr_matrix
from copy import deepcopy

import torch.nn.functional as F
import numpy as np
import argparse
import warnings
import random
import torch
import json
import pdb

warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def run(
    adj,
    feats,
    labels,
    model_param,
    train_idx,
    valid_idx,
    test_idx,
    seed
):
    ###################################################
    torch.cuda.empty_cache()
    RANDOM_SEED = seed
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    ###################################################

    _, num_feats = feats.shape
    num_labels = len(torch.unique(labels))
    labels_np = labels.cpu().detach().numpy()

    model = GCN(
        nfeat=num_feats,
        nhid=model_param["num_hidden"],
        nclass=num_labels,
        dropout=model_param["dropout"],
    ).cuda()

    optim = torch.optim.Adam(
        model.parameters(), lr=model_param["lr"], weight_decay=model_param["wd"]
    )

    minimum_epoch = model_param["minimum_epoch"]
    num_patiences = 200
    display_step = 20
    best_val_loss = 1e9
    best_val_acc = 0
    count = 0
    for epoch in range(model_param["num_epochs"]):
        model.train()
        logit = model(feats, adj)
        train_loss = F.nll_loss(logit[train_idx], labels[train_idx])

        optim.zero_grad()
        train_loss.backward()
        optim.step()

        model.eval()
        logit = model(feats, adj)
        predict = torch.argmax(logit, dim=1)
        predict_np = predict.cpu().detach().numpy()

        valid_loss = F.nll_loss(logit[valid_idx], labels[valid_idx])
        valid_acc = accuracy_score(labels_np[valid_idx], predict_np[valid_idx])
        valid_f1 = f1_score(
            labels_np[valid_idx], predict_np[valid_idx], average="weighted"
        )

        if (epoch + 1) % display_step == 0:
            print(
                f"Epoch {epoch+1}/{model_param['num_epochs']}  Train Loss: {train_loss.item():.2e}  Valid Loss: {valid_loss:.2e}  Valid Acc: {valid_acc:.2f}  Valid F1: {valid_f1:.2f}"
            )

        if epoch < minimum_epoch:
            best_val_acc = valid_acc
            best_val_loss = valid_loss
            best_state_dict = deepcopy(model.state_dict())
        else:
            if (best_val_loss > valid_loss) or (best_val_acc < valid_acc):
                best_val_acc = valid_acc
                best_val_loss = valid_loss
                best_state_dict = deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

            if count == num_patiences:
                break

    model.load_state_dict(best_state_dict)
    model.eval()
    logit = model(feats, adj)
    predict = torch.argmax(logit, dim=1)
    predict_np = predict.cpu().detach().numpy()

    test_acc = accuracy_score(labels_np[test_idx], predict_np[test_idx])
    test_f1 = f1_score(labels_np[test_idx], predict_np[test_idx], average="weighted")
    print("-------------------------------------------------------------")
    print(f"Test Acc: {test_acc:.2f}  Test F1: {test_f1:.2f}")
    return test_acc, test_f1, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main code for TiGer")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--graph_name", type=str, default="school")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adjs, feats, labels, train_idxs, valid_idxs, test_idxs = load_dataset(
        args.root,
        args.graph_name,
        args.seed
    )
    
    noise_adjs_path = os.path.join(args.root, args.graph_name, "noise_graph", f"{args.graph_name}_{args.seed}.pt")
    noise_adjs = torch.load(noise_adjs_path)

    with open(os.path.join("configs", f"GCN_{args.graph_name}_config.json"), "r") as f:
        GCN_param = json.load(f)
    with open(os.path.join("configs", f"TiGer_{args.graph_name}_config.json"), "r") as f:
        TiGer_param = json.load(f)

    _, num_feats = feats.shape
    num_labels = len(torch.unique(labels))
    num_timesteps = len(adjs)

    agent = TiGer_agent(TiGer_param, args.seed)

    # Training GCN for node classification at time step 0
    acc, f1, model = run(
        adjs[0].cuda(),
        feats[:adjs[0].size(0)].cuda(),
        labels.long().cuda(),
        GCN_param,
        train_idxs[0],
        valid_idxs[0],
        test_idxs[0],
        args.seed
    )

    purified_adjs = [adjs[0]]
    acc_result = [acc]
    for t in range(1, num_timesteps):
        # Purification with TiGer at time step t
        adj = adjs[t]
        noise_adj = noise_adjs[t]
        prev_noise_adj = noise_adjs[t-1]
        prev_purified_adj = purified_adjs[-1]

        unpurified_adj, new_edge_set = add_new_graph(prev_purified_adj, noise_adj, prev_noise_adj)
        
        model.eval()
        logits = torch.exp(model(feats[:unpurified_adj.size(0)].cuda(), unpurified_adj.cuda())).detach().cpu().numpy()
        del model

        num_target = int(adj.sum().item() / 2)

        agent.step = t
        purified_adj = agent.purification(unpurified_adj, prev_purified_adj, new_edge_set, num_target, logits, feats)
        
        acc, f1, model = run(
            purified_adj.cuda(),
            feats[:purified_adj.size(0)].cuda(),
            labels.long().cuda(),
            GCN_param,
            train_idxs[t],
            valid_idxs[t],
            test_idxs[t],
            args.seed,
        )

        acc_result.append(acc)
        print()

        purified_adjs.append(purified_adj)

    purified_adjs_path = os.path.join("result", f"{args.graph_name}_purified_graphs_{args.seed}.pt")
    torch.save(purified_adjs, purified_adjs_path)
    print(
        "[Accuracy]",
        " ".join([f"{acc_result[t]:.3f}" for t in range(num_timesteps)]),
    )