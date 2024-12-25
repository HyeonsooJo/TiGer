from torch_geometric.nn import GCNConv

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class Module_L(nn.Module):
    def __init__(self, num_feat, num_hidden, dropout):
        super(Module_L, self).__init__()
        self.model_name = 'leo_gnn_d'
        self.gc1 = GCNConv(num_feat, num_hidden)
        self.gc2 = GCNConv(num_hidden, num_hidden)

        self.linear = nn.Linear(2 * num_hidden, num_hidden)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        
        self.weight_lin = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.bias_lin = nn.Parameter(torch.Tensor(num_hidden))
        nn.init.xavier_uniform_(self.weight_lin)
        nn.init.zeros_(self.bias_lin)

        self.w_q = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.w_k = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        self.w_v = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        
        self._loss = nn.BCELoss()
        self.auxloss = 0

    def forward(self, x, mp_adj, edges, prev_embs):
        x = self.tanh(self.gc1(x, mp_adj))
        x = self.dropout(x)
        x = self.tanh(self.gc2(x, mp_adj)) 
        embeddings = self.dropout(x)

        num_current_nodes = embeddings.shape[0]
        num_prev_emb = len(prev_embs)
        if num_prev_emb > 0:
            start_idx = 0 
            attn_result = []
            for i in range(num_prev_emb):
                end_idx = prev_embs[i].shape[0]
                if num_current_nodes == end_idx: break

                current_prev_emb = [prev_embs[j][start_idx: end_idx].unsqueeze(0) for j in range(i, num_prev_emb)]
                all_embs = torch.concatenate(current_prev_emb, dim=0)
                all_embs = torch.concatenate([all_embs, embeddings[start_idx: end_idx].unsqueeze(0)], dim=0)

                query =  embeddings[start_idx: end_idx].unsqueeze(0) @ self.w_q
                key = all_embs @ self.w_k
                value = all_embs @ self.w_v

                attn = torch.bmm(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2)) / np.sqrt(key.shape[-1])
                attn_weights = F.softmax(attn, dim=-1)
                attn_output = torch.bmm(attn_weights, value.transpose(0, 1)).squeeze(1)
                attn_result.append(attn_output)
                start_idx = end_idx
                
            if num_current_nodes != start_idx:
                remain_emb = embeddings[start_idx:] @ self.w_v
                attn_result.append(remain_emb)
            attn_embeddings = torch.vstack(attn_result)
        else:
            attn_embeddings = embeddings @ self.w_v

        embeddings = self.tanh(self.linear(torch.concatenate([embeddings, attn_embeddings], dim=1)))
        self.embeddings = embeddings
        sym_weight = (self.weight_lin + self.weight_lin.T) / 2
        sim = (embeddings[edges[:,0],:] @ sym_weight) * embeddings[edges[:,1],:] + self.bias_lin
        sim = sim.sum(dim=1)
        return self.sigmoid(sim)    
    
    def get_node_embedding(self, x, adj):
        x = self.tanh(self.gc1(x, adj))
        x = self.tanh(self.gc2(x, adj)) 
        return x
    
    def get_embedding(self, edges):
        return (self.embeddings[edges[:,0],:], self.embeddings[edges[:,1],:])
    
class Proximity(nn.Module):
    def __init__(self):
        super(Proximity, self).__init__()
        self._loss = lambda x,y: 0
        self.train_score = None
        self.valid_score = None
        self.test_score = None

    def pre_setting(self, train_scores, valid_scores, test_scores):
        self.train_score = train_scores
        self.valid_score = valid_scores
        self.test_score = test_scores

    def forward(self, index, mode):
        if mode == 'train': return self.train_score[index]
        elif mode == 'valid': return self.valid_score[index]
        elif mode == 'test': return self.test_score[index]


class Module_S(nn.Module):
    def __init__(self):
        super(Module_S, self).__init__()
        self._loss = lambda x,y: 0
        self.logits = None
        self.train_score = None
        self.valid_score = None
        self.test_score = None

    def pre_setting(self, train_scores, valid_scores, test_scores):
        self.train_score = train_scores
        self.valid_score = valid_scores
        self.test_score = test_scores

    def forward(self, index, mode):
        if mode == 'train': return self.train_score[index]
        elif mode == 'valid': return self.valid_score[index]
        elif mode == 'test': return self.test_score[index]

    def get_embedding(self, edges):
        return (self.logits[edges[:,0],:], self.logits[edges[:,1],:])


class TiGer_model(nn.Module):
    def __init__(self, num_feat, logits, num_hidden, dropout, aux_loss_weight, prox_weight):
        super(TiGer_model, self).__init__()
        self.module_l = Module_L(num_feat=num_feat, 
                                      num_hidden=num_hidden, 
                                      dropout=0.1).cuda()
        
        self.module_s = Module_S().cuda()
        self.proximity = Proximity().cuda()
        self.ensemble_model = ensemble(num_hidden, logits.shape[1], dropout, prox_weight)
        
        self._loss = nn.BCELoss()
        self.aux_loss_weight = aux_loss_weight
        
    def forward(self, x, mp_adj, edges, index, mode, prev_embs, y=None, test=False):
        scores = []
        self.auxloss = torch.zeros(1).cuda()

        ml_scores = self.module_l(x, mp_adj, edges.T, prev_embs)
        if not y == None:
            ml_loss = self.module_l._loss(ml_scores, y)
            self.auxloss += ml_loss
        scores.append(ml_scores)

        ms_scores = self.module_s(index, mode)
        scores.append(ms_scores)

        prox_scores = self.proximity(index, mode)
        scores.append(prox_scores)

        ml_embs = self.module_l.get_embedding(edges.T)
        ms_embs = self.module_s.get_embedding(edges.T)

        final_scores = self.ensemble_model(scores, ml_embs, ms_embs)

        self.auxloss *= self.aux_loss_weight
        if test: 
            return final_scores.detach().cpu() , ml_scores.detach().cpu(), ms_scores.detach().cpu(), prox_scores.detach().cpu()
        return final_scores    

            
class ensemble(nn.Module):
    '''
    not attention, weighted sum
    calculate weight from its edge embedding(=node embedding's max||meax)
    weight net = 2 layer / jaccard input = max||mean(dimension_reduction(node feature))
    '''
    def __init__(self, num_hidden, num_logits, dropout, prox_weight):
        super(ensemble, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.prox_weight = prox_weight
        
        self.ml_weight = nn.Sequential(
            nn.Linear(2 * num_hidden, num_hidden), nn.Tanh(), self.dropout,
            nn.Linear(num_hidden, 1), nn.Tanh(),
        )    
        self.ms_weight = nn.Sequential(
            nn.Linear(2 * num_hidden, num_hidden), nn.Tanh(), self.dropout,
            nn.Linear(num_hidden, 1), nn.Tanh(),
        )
        self.ms_dim_reduc = nn.Sequential(
            nn.Linear(num_logits, num_hidden), nn.Tanh(), self.dropout,
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, scores, ml_embs, ms_embs):
        scores = torch.vstack(scores).T

        attn_scores = []
        attn_scores.append(self.ml_weight(self.mean_max_pool(ml_embs)))
        ms_embs = (self.ms_dim_reduc(ms_embs[0]), self.ms_dim_reduc(ms_embs[1]))
        attn_scores.append(self.ms_weight(self.mean_max_pool(ms_embs)))
        attn_scores.append(torch.ones_like(attn_scores[0]).cuda() * self.prox_weight) 
        attn_scores = torch.hstack(attn_scores)
        attn_scores = self.softmax(attn_scores)

        final_scores = torch.mul(scores, attn_scores).sum(dim=1)
        return torch.clamp(final_scores, min=0.0, max=1.0)

    def mean_max_pool(self, embeddings):
        a, b = embeddings
        mean = (a + b) / 2
        maximum = torch.maximum(a, b)
        return torch.cat([mean, maximum], dim=1)

