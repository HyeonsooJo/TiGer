from models import *
from utils import *

from torch_geometric.utils import add_random_edge
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from copy import deepcopy

import networkx as nx
import numpy as np
import random


class TiGer_agent():
    def __init__(self, TiGer_config, seed):
        self.TiGer_config = TiGer_config
        self.seed = seed

        self.deviation_stat = dict()
        self.prev_embs = []
        self.step = None

    def kl_divergence(self, a, b):
        if len(b.shape) == 1:
            return np.sum(a * (np.log(a+1e-12) - np.log(b+1e-12)))
        kl_div = np.sum(a * (np.log(a+1e-12) - np.log(b+1e-12)), axis=1)
        return (kl_div.mean(), kl_div.std()) 

    def update_deviation_stat(self, logits, row, col, num_nodes):
        curr_adj_sp = csr_matrix((np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes))
        for node in range(num_nodes):
            neighbors = curr_adj_sp[node].nonzero()[1]
            if len(neighbors) <= 1: 
                self.deviation_stat[node] = (0, 0)
            else:
                mean_kl, std_kl = self.kl_divergence(logits[node], logits[neighbors])
                self.deviation_stat[node] = (mean_kl, std_kl)

    def load_split_sparse_data(self, data):
        ###############        SEED        ###############
        torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        ##################################################

        edge_index = data.edge_index
        undirected_edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        num_edges = undirected_edge_index.shape[1]
        train_index = np.random.choice(np.arange(num_edges),  int(0.8 * num_edges), replace=False)
        valid_index = np.setdiff1d(np.arange(num_edges), train_index)
        train_edge_index = undirected_edge_index[:, train_index]
        valid_edge_index = undirected_edge_index[:, valid_index]
        train_edge_index = torch.hstack([train_edge_index, train_edge_index[[1, 0], :]])
        valid_edge_index = torch.hstack([valid_edge_index, valid_edge_index[[1, 0], :]])
        data.train_edge_index = train_edge_index
        data.valid_edge_index = valid_edge_index
        
        # import pdb; pdb.set_trace()
        num_nodes = edge_index.max().item() + 1
        _, added_edges = add_random_edge(data.edge_index, 1.0, num_nodes=num_nodes, force_undirected=True)
        added_edges = added_edges[:, added_edges[0] < added_edges[1]]
        train_neg_edge_index = added_edges[:, train_index]
        valid_neg_edge_index = added_edges[:, valid_index]
        train_neg_edge_index = torch.hstack([train_neg_edge_index, train_neg_edge_index[[1, 0], :]])
        valid_neg_edge_index = torch.hstack([valid_neg_edge_index, valid_neg_edge_index[[1, 0], :]])
        data.train_neg_edge_index = train_neg_edge_index
        data.valid_neg_edge_index = valid_neg_edge_index
        return data

    def get_score(self, data, srcs, dsts, logits):
        split_graph = self.load_split_sparse_data(data)

        tiger_model = TiGer_model(num_feat=split_graph.x.shape[1], 
                                      logits=logits,
                                      num_hidden=self.TiGer_config["num_hidden"], 
                                      dropout=self.TiGer_config["dropout"], 
                                      aux_loss_weight=self.TiGer_config["aux_loss_weight"],
                                      prox_weight=self.TiGer_config["prox_weight"]).cuda()
        
        # Training TiGer
        test_edge_index = torch.LongTensor(np.vstack([srcs, dsts])).cuda()
        tiger_model, mp_adj = self.train(tiger_model, split_graph, logits, test_edge_index)
        
        # Scoring new edges
        tiger_model.eval()
        scores = tiger_model(split_graph.x, mp_adj, test_edge_index, np.arange(test_edge_index.shape[1]), mode='test', prev_embs=self.prev_embs, test=True)
        self.prev_embs.append(tiger_model.module_l.get_node_embedding(split_graph.x, mp_adj).detach())
        return scores[0].cpu().detach().numpy()
    

    def train(self, tiger_model, split_graph, logits, test_edge_index):
        torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        num_nodes = split_graph.num_nodes
        mp_adj = SparseTensor(row=split_graph.train_edge_index[0], col=split_graph.train_edge_index[1], sparse_sizes=[num_nodes, num_nodes])
        train_pos_edge_index = split_graph.train_edge_index
        train_neg_edge_index = split_graph.train_neg_edge_index
        valid_pos_edge_index = split_graph.valid_edge_index
        valid_neg_edge_index = split_graph.valid_neg_edge_index
        
        train_pos_edge_index = train_pos_edge_index[:, train_pos_edge_index[0] < train_pos_edge_index[1]]
        train_neg_edge_index = train_neg_edge_index[:, train_neg_edge_index[0] < train_neg_edge_index[1]]
        valid_pos_edge_index = valid_pos_edge_index[:, valid_pos_edge_index[0] < valid_pos_edge_index[1]]
        valid_neg_edge_index = valid_neg_edge_index[:, valid_neg_edge_index[0] < valid_neg_edge_index[1]]

        train_edge_index = torch.hstack([train_pos_edge_index, train_neg_edge_index])
        train_labels = torch.Tensor(np.array([1] * train_pos_edge_index.shape[1] + [0] * train_neg_edge_index.shape[1])).cuda()

        valid_edge_index = torch.hstack([valid_pos_edge_index, valid_neg_edge_index])
        valid_labels = torch.Tensor(np.array([1] * valid_pos_edge_index.shape[1] + [0] * valid_neg_edge_index.shape[1])).cuda()

        # Calculate Proximity Scores & Deviation Scores
        srcs = np.concatenate([train_edge_index[0].cpu().numpy(), valid_edge_index[0].cpu().numpy()])
        dsts = np.concatenate([train_edge_index[1].cpu().numpy(), valid_edge_index[1].cpu().numpy()])
        if self.TiGer_config["prox_type"] == "adamic_adar":
            nx_graph = nx.Graph()   
            for i in range(num_nodes): nx_graph.add_node(i)
            for row, col in split_graph.edge_index.T:
                nx_graph.add_edge(row.item(), col.item())
            prox_scores = torch.Tensor(get_aa_score(nx_graph, srcs, dsts)).cuda()
        elif self.TiGer_config["prox_type"] == "svd":
            row, col = [], []
            for row_item, col_item in split_graph.edge_index.T:
                row.append(row_item.item())
                col.append(col_item.item())
            row = np.array(row)
            col = np.array(col)
            data = np.ones_like(row).astype(float)
            adj_coo = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
            prox_scores = torch.Tensor(get_svd_score(adj_coo, srcs, dsts)).cuda()
        deviation_scores = -torch.Tensor(self.get_deviation_scores(srcs, dsts, logits)).cuda()
 
        train_prox_scores = prox_scores[:train_edge_index.shape[1]]
        train_deviation_scores = deviation_scores[:train_edge_index.shape[1]]
        valid_prox_scores = prox_scores[train_edge_index.shape[1]:]
        valid_deviation_scores = deviation_scores[train_edge_index.shape[1]:]

        test_srcs = test_edge_index[0].cpu().numpy()
        test_dsts = test_edge_index[1].cpu().numpy()
        if self.TiGer_config["prox_type"] == "adamic_adar":
            test_prox_scores = torch.Tensor(get_aa_score(nx_graph, test_srcs, test_dsts)).cuda()
        else:
            test_prox_scores = torch.Tensor(get_svd_score(adj_coo, test_srcs, test_dsts)).cuda()
        test_deviation_scores = -torch.Tensor(self.get_deviation_scores(test_srcs, test_dsts, logits)).cuda()

        max_prox_score = max([train_prox_scores.max(), valid_prox_scores.max(), test_prox_scores.max()])
        min_prox_score = min([train_prox_scores.min(), valid_prox_scores.min(), test_prox_scores.min()])
        train_prox_scores = (train_prox_scores - min_prox_score) / (max_prox_score - min_prox_score)
        valid_prox_scores = (valid_prox_scores - min_prox_score) / (max_prox_score - min_prox_score)
        test_prox_scores = (test_prox_scores - min_prox_score) / (max_prox_score - min_prox_score)
        tiger_model.proximity.pre_setting(train_prox_scores, valid_prox_scores, test_prox_scores)

        max_deviation_score = max([train_deviation_scores.max(), valid_deviation_scores.max(), test_deviation_scores.max()])
        min_deviation_score = min([train_deviation_scores.min(), valid_deviation_scores.min(), test_deviation_scores.min()])
        train_deviation_scores = (train_deviation_scores - min_deviation_score) / (max_deviation_score - min_deviation_score)
        valid_deviation_scores = (valid_deviation_scores - min_deviation_score) / (max_deviation_score - min_deviation_score)
        test_deviation_scores = (test_deviation_scores - min_deviation_score) / (max_deviation_score - min_deviation_score)
        tiger_model.module_s.pre_setting(train_deviation_scores, valid_deviation_scores, test_deviation_scores)
 
        tiger_model.module_s.logits = torch.Tensor(logits).cuda()

        batch_size = self.TiGer_config["batch_size"]
        filter_rate = self.TiGer_config["filter_rate"]
        num_epochs = self.TiGer_config["num_epochs"]
        lr = self.TiGer_config["lr"]
        wd = self.TiGer_config["wd"]

        num_train = train_edge_index.shape[1]
        num_train_batch = int(np.ceil(num_train / batch_size))
        optimizer = torch.optim.Adam(params=tiger_model.parameters(), lr=lr, weight_decay=wd)  
        best_val_auroc = 0

        print("TiGer Start Training")
        for epoch in range(num_epochs):
            tiger_model.train()
            shuffle_idx = np.arange(num_train)
            np.random.shuffle(shuffle_idx)
            for batch_idx in range(num_train_batch):
                start_idx = batch_idx * batch_size
                end_idx = np.minimum((batch_idx + 1) * batch_size, num_train)

                indices = shuffle_idx[start_idx:end_idx]

                batch_edge_index = train_edge_index[:, indices]
                batch_labels = train_labels[indices]

                out = tiger_model(split_graph.x, mp_adj, batch_edge_index, indices, 'train', self.prev_embs, y=batch_labels)

                if not filter_rate == 0.0:
                    out_idx = out.detach()[batch_labels==1].sort()[1]>int(out.shape[0]*filter_rate)
                    out = torch.cat((out[batch_labels==0], out[batch_labels==1][out_idx]))
                    batch_labels = torch.cat((batch_labels[batch_labels==0], batch_labels[batch_labels==1][out_idx]))

                main_loss = tiger_model._loss(out, batch_labels)
                reg_loss = tiger_model.auxloss
                loss = main_loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_auroc = self.valid_auroc(tiger_model, split_graph, mp_adj, valid_edge_index, valid_labels, 'valid', batch_size)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch: {epoch+1} Loss: {main_loss.item():.3f}, {reg_loss.item():.3f}  Val AUROC: {val_auroc[0]:.3f}")
                # scores = tiger_model(split_graph.x, mp_adj, test_edge_index, np.arange(test_edge_index.shape[1]), mode='test', prev_embs=self.prev_embs, step=self.step, test=True, inference=True)
                # for score in scores:
                #     num_real_edges = test_label.sum()
                #     test_label[np.argsort(-score)[num_real_edges:]]
                #     print(f"{test_label[np.argsort(-score)[num_real_edges:]].mean():.3f}", end=" / ")
                # print()
                # print()

            if val_auroc[0] > best_val_auroc:
                best_val_auroc = val_auroc[0]
                best_model_state_dict = deepcopy(tiger_model.state_dict())  

        tiger_model.load_state_dict(best_model_state_dict)
        return tiger_model, mp_adj
    
    def valid_auroc(self, model, split_graph, mp_adj, edge_index, labels, mode, batch_size):
        model.eval()

        num_edge_index = edge_index.shape[1]
        num_batch = int(np.ceil(num_edge_index / batch_size))
        outputs = torch.zeros([num_edge_index, 3])
        idx = np.arange(num_edge_index)

        for batch_idx in range(num_batch):
            start_idx = batch_idx * batch_size
            end_idx = np.minimum((batch_idx + 1) * batch_size, num_edge_index)
            indices = idx[start_idx:end_idx]
            batch_edge_index = edge_index[:, indices]
            batch_labels = labels[indices]
            out = model(split_graph.x, mp_adj, batch_edge_index, indices, mode, self.prev_embs, y=batch_labels, test=True)
            for i in range(3):
                outputs[start_idx:end_idx, i] = out[i]

        aurocs = []
        for i in range(3):
            try:
                aurocs.append(roc_auc_score(labels.cpu(), outputs[:, i]))        
            except: continue
        return aurocs

    def valid_loss(self, model, split_graph, mp_adj, edge_index, labels, mode, graph_name):
        model.eval()

        if graph_name == 'school_s':
            batch_size = 1024
        elif graph_name == 'aminer':
            # aminer
            batch_size = 1024
        elif graph_name == 'patent':
            # patent
            batch_size = 16384
        elif graph_name == 'dblp':
            # dblp
            batch_size = 65536
        elif graph_name == 'arXivAI':
            # arXiv
            batch_size = 262144

        num_edge_index = edge_index.shape[1]
        num_batch = int(np.ceil(num_edge_index / batch_size))
        outputs = torch.zeros([num_edge_index, 4])
        idx = np.arange(num_edge_index)
        # for batch_idx in tqdm.tqdm(range(num_batch)):
        for batch_idx in range(num_batch):
            start_idx = batch_idx * batch_size
            end_idx = np.minimum((batch_idx + 1) * batch_size, num_edge_index)
            indices = idx[start_idx:end_idx]
            batch_edge_index = edge_index[:, indices]
            batch_labels = labels[indices]
            out = model(split_graph.x, mp_adj, batch_edge_index, indices, mode, self.prev_embs, batch_labels, test=True, inference=True)
            for i in range(4):
                outputs[start_idx:end_idx, i] = out[i]

        loss = []
        for i in range(4):
            try:
                loss.append(model._loss(outputs[:, i], labels.cpu()))        
            except: continue
        return loss

    def get_deviation_scores(self, srcs, dsts, logit):
        deviation_scores = np.zeros(len(srcs))
        for i, (src, dst) in enumerate(zip(srcs, dsts)):

            if src not in self.deviation_stat or dst not in self.deviation_stat:
                continue
            if self.deviation_stat[src][1] == 0:
                src_score = 0
            else:
                src_dst_kl = self.kl_divergence(logit[src], logit[dst])
                src_score = np.abs(src_dst_kl - self.deviation_stat[src][0]) / self.deviation_stat[src][1]
            if self.deviation_stat[dst][1] == 0:
                dst_score = 0
            else:
                dst_src_kl = self.kl_divergence(logit[dst], logit[src])
                dst_score = np.abs(dst_src_kl - self.deviation_stat[dst][0]) / self.deviation_stat[dst][1]
            
            kl_score = (src_score + dst_score) / 2
            deviation_scores[i] = kl_score
        return deviation_scores    
    

    def purification(self, unpurified_adj, prev_purified_adj, new_edge_set, num_target, logits, feats):        
        row, col, num_nodes = unpurified_adj.coo()[0].numpy(), unpurified_adj.coo()[1].numpy(), unpurified_adj.size(0)
        origin_row = prev_purified_adj.coo()[0].numpy()
        origin_col = prev_purified_adj.coo()[1].numpy()
        origin_srcs = origin_row[origin_row < origin_col]
        origin_dsts = origin_col[origin_row < origin_col]
        self.update_deviation_stat(logits, row, col, num_nodes)  

        srcs = []
        dsts = []
        for src, dst in new_edge_set:
            srcs.append(src)
            dsts.append(dst)
        srcs = np.array(srcs)
        dsts = np.array(dsts)

        edge_index = torch.LongTensor(np.vstack([row, col])).cuda()
        data = Data(x=torch.Tensor(feats).cuda()[:num_nodes], edge_index=edge_index, num_nodes=num_nodes)
        score = self.get_score(data, srcs, dsts, logits)

        num_purify= int(len(row) / 2) - num_target
        
        print(f"Num current edges:{int(len(row) / 2)}  Num purified edges: {num_purify}   after purificaiton: {num_target}")
        purify_mask = np.argsort(-score)[:-num_purify]
        purify_srcs = np.concatenate([origin_srcs, srcs[purify_mask]])
        purify_dsts = np.concatenate([origin_dsts, dsts[purify_mask]])
        row = torch.LongTensor(np.concatenate([purify_srcs, purify_dsts]))
        col = torch.LongTensor(np.concatenate([purify_dsts, purify_srcs]))
        purified_adj = SparseTensor(row=row, col=col, sparse_sizes=[num_nodes, num_nodes])
        return purified_adj

