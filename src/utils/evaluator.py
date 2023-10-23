import torch
from sklearn.metrics import *
import numpy as np
import math
import itertools
import torch.nn.functional as F
import pdb

def evaluate(pred, gt, metrics, item_num):
#     pdb.set_trace()
    evaluation = []
    sorted_p, sorted_gt = sort(pred, gt)
#     pdb.set_trace()
    for metric in metrics:
        k = int(metric.split('@')[-1])
        if metric.startswith('hit@'):
            evaluation.append(hit_at_k(sorted_gt, k))
        elif metric.startswith('precision@'):
            evaluation.append(precision_at_k(sorted_gt, k))
        elif metric.startswith('recall@'):
            evaluation.append(recall_at_k(sorted_gt, k))
        elif metric.startswith('ndcg@'):
            evaluation.append(ndcg_at_k(sorted_gt, k))
        elif metric.startswith('unbiasedhit@'):
            evaluation.append(unbiased_hit_at_k(sorted_gt, k, item_num))
        elif metric.startswith('unbiasedndcg@'):
            evaluation.append(unbiased_ndcg_at_k(sorted_gt, k, item_num))
            
    format_str = []
    for m in evaluation:
        format_str.append('%.4f' % m)
#     pdb.set_trace()
    return evaluation, ','.join(format_str)

def mmr_evaluate(similarity, pred, gt, rec_item, metrics, lambda_para, item_num, K=101):
#     pdb.set_trace()
    evaluation = []
    sorted_gt, _ = mmr_sort(similarity, pred, gt, rec_item, lambda_para, K)
#     pdb.set_trace()
    for metric in metrics:
        k = int(metric.split('@')[-1])
        if metric.startswith('hit@'):
            evaluation.append(hit_at_k(sorted_gt, k))
        elif metric.startswith('precision@'):
            evaluation.append(precision_at_k(sorted_gt, k))
        elif metric.startswith('recall@'):
            evaluation.append(recall_at_k(sorted_gt, k))
        elif metric.startswith('ndcg@'):
            evaluation.append(ndcg_at_k(sorted_gt, k))
        elif metric.startswith('unbiasedhit@'):
            evaluation.append(unbiased_hit_at_k(sorted_gt, k, item_num))
        elif metric.startswith('unbiasedndcg@'):
            evaluation.append(unbiased_ndcg_at_k(sorted_gt, k, item_num))
            
    format_str = []
    for m in evaluation:
        format_str.append('%.4f' % m)
#     pdb.set_trace()
    return evaluation, ','.join(format_str)

def mmr_sort(similarity, pred, gt, rec_item, lambda_para, K):
    sorted_gt = {}
    rec_list = {}
    for i in pred:
        gt_sorted, rec_sorted = [], []
        predi = np.array(pred[i])
        reci = rec_item[i]
        gti = gt[i]
        for j in range(K):
            if j == 0:
                idx = np.argmax(predi)
                gt_sorted.append(gti[idx])
                rec_sorted.append(reci[idx])
            else:
                sim = []
                for item in rec_sorted:
                    sim.append(similarity[item][reci])
                new_pred = predi * lambda_para - (1- lambda_para) * np.max(np.array(sim), axis=0)
                idx = np.argmax(new_pred)
                gt_sorted.append(gti[idx])
                rec_sorted.append(reci[idx])
#                 pdb.set_trace()
            predi = np.delete(predi, idx)
            gti = np.delete(gti, idx)
            reci = np.delete(reci, idx)
        
        sorted_gt[i] = np.array(gt_sorted)
        rec_list[i] = np.array(rec_sorted)
    return sorted_gt, rec_list
    
    
def sort(pred, gt):
    sorted_p, sorted_gt = {}, {}
    for i in pred:
#         pdb.set_trace()
        index = np.argsort(-np.array(pred[i]))
        sorted_p[i] = np.array(pred[i])[index]
        sorted_gt[i] = np.array(gt[i])[index]
    return sorted_p, sorted_gt

def get_rec_list(pred, rec_item, k):
    sorted_p, rec_list = {}, {}
    for i in pred:
        index = np.argsort(-np.array(pred[i]))
        rec_list[i] = np.array(rec_item[i])[index][:k]
    return rec_list

def get_rec_reward(pred, rec_item, gt):
    reward_list, rec_list = {}, {}
    for i in pred:
        index = np.random.choice([i for i in range(len(pred[i]))],size=1, p = softmax(pred[i]))
        rec_list[i] = np.array(rec_item[i])[index]
        reward_list[i] = np.array(gt[i])[index]
    return rec_list, reward_list

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def hit_at_k(sorted_gt, k):
    hit = 0.0
    for user in sorted_gt:
        if np.sum(sorted_gt[user][:k]) > 0:
            hit += 1
    return hit/len(sorted_gt)

def unbiased_hit_at_k(sorted_gt, k, item_num):
    hit = 0.0
    m = 100
    n = item_num
    for user in sorted_gt:
        r = sorted_gt[user].nonzero()[0][0] + 1
        true_r = 1 + np.floor((n-1) * (r-1) / m)
        if true_r <= k:
            hit += 1
    return hit/len(sorted_gt)

def precision_at_k(sorted_gt, k):
    pre = 0.0
    for user in sorted_gt:
        pre += np.sum(sorted_gt[user][:k])/k
    return pre/len(sorted_gt)

def recall_at_k(sorted_gt, k):
    recall = 0.0
    for user in sorted_gt:
        recall += np.sum(sorted_gt[user][:k]) / np.sum(sorted_gt[user])
    return recall / len(sorted_gt)

def ndcg_at_k(sorted_gt, k):
    ndcg = 0.0
    for user in sorted_gt:
        dcg = 0.0
        idcg = 0.0
        for i in range(k):
            if sorted_gt[user][i] > 0:
                dcg += 1. / np.log2(i + 2)
        for i in range(min(k, np.sum(sorted_gt[user]))):
            idcg += 1. / np.log2(i + 2)
        ndcg += dcg / idcg
    return ndcg / len(sorted_gt)

def unbiased_ndcg_at_k(sorted_gt, k, item_num):
    ndcg = 0.0
    m = 100
    n = item_num
    for user in sorted_gt:
        dcg = 0.0
        idcg = 0.0
        r = sorted_gt[user].nonzero()[0][0] + 1
        true_r = 1 + np.floor((n-1) * (r-1) / m)
        if true_r <= k:
            dcg += 1. / np.log2(true_r + 2)
        for i in range(min(k, np.sum(sorted_gt[user]))):
            idcg += 1. / np.log2(i + 2)
        ndcg += dcg / idcg
    return ndcg / len(sorted_gt)


def content_diversity(model, rec_list):
    item_pair = []
    for user in rec_list:
        item_pair.extend([list(pair) for pair in itertools.combinations(rec_list[user], 2)])
    item_pair = np.array(item_pair)
    item_i = torch.tensor(item_pair[:,0]).to(torch.long).to(model.dummy_param.device)
    item_j = torch.tensor(item_pair[:,1]).to(torch.long).to(model.dummy_param.device)
    distance = (model.iid_embeddings(item_i).detach() - model.iid_embeddings(item_j).detach()).pow(2).sum(1).sqrt()
#     distance = (F.normalize(model.iid_embeddings(item_i).detach(),dim=0,p=2) - F.normalize(model.iid_embeddings(item_j).detach(),dim=0,p=2)).pow(2).sum(1).sqrt()
    return distance.mean().detach().cpu().numpy()
    
