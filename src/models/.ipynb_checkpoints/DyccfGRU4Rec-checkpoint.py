from models.GRU4Rec import GRU4Rec
import numpy as np
import os
import torch
import torch.nn.functional as F
import pdb

class DyccfGRU4Rec(GRU4Rec):
    loader = 'DccfHistLoader'
    runner = 'DccfRunner'
    
    def parse_model_args(parser):
        parser.add_argument('--ctf_num', default=5, type=int,
                           help='number of counterfactual histories')
        parser.add_argument('--fact_prob', default=0.3, type=float,
                           help='the probability of factual history')
        return GRU4Rec.parse_model_args(parser)
    
    def __init__(self, ctf_num, fact_prob, *args, **kwargs):
        self.ctf_num = ctf_num
        self.fact_prob = fact_prob
        GRU4Rec.__init__(self, *args, **kwargs)
        
    def predict(self, batch):
        """
        prediction for evaluation
        """
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        hist = batch['history'].to(torch.long).to(self.dummy_param.device)
        label = batch['rating'].to(torch.long).to(self.dummy_param.device).view([-1])
        
        valid_his = hist.abs().gt(0).long()
        pos_his = hist.ge(0).long()
        
        # counterfactual histories
        last_valid_idx = torch.topk(valid_his * torch.tensor([i for i in range(valid_his.shape[1])]).to(self.dummy_param.device),k=1)[1]
        last_valid_mask = F.one_hot(last_valid_idx, num_classes=valid_his.shape[1]).sum(dim=1)
        
        last_pos_idx = torch.topk(pos_his * torch.tensor([i for i in range(valid_his.shape[1])]).to(self.dummy_param.device),k=1)[1]
#         last_pos_mask = F.one_hot(last_pos_idx, num_classes=valid_his.shape[1]).sum(dim=1)
        last_pos_mask = last_valid_mask.clone()
        
        ctf_pref_his = (-1 * last_valid_mask + (1 - last_valid_mask)) * hist
        
        last_pos_iid = (hist * last_pos_mask).sum(dim=1)
        
#         pdb.set_trace()
        
        item_sim = torch.matmul(F.normalize(self.iid_embeddings(last_pos_iid.abs().view([-1])),dim=0,p=2), F.normalize(self.iid_embeddings.weight[1:],dim=0,p=2).t())
        sample_item = torch.topk(-item_sim, self.ctf_num-1).indices+1
        ctf_item_his = sample_item.unsqueeze(dim=2) * last_pos_mask.unsqueeze(1) + (hist * (1-last_pos_mask)).unsqueeze(1)
        
        new_hist = torch.cat([hist, ctf_pref_his, ctf_item_his.reshape(hist.shape[0], -1)], axis=1).reshape(-1, hist.shape[1])
        
        # get prediction for factual histories and counterfactual histories
        valid_his = new_hist.abs().gt(0).long()
        his_length = valid_his.sum(dim=1)
        
        his_pos_neg = new_hist.ge(0).unsqueeze(-1).float()
        
        pos_his_vectors = self.iid_embeddings(new_hist.abs()) * valid_his.unsqueeze(dim=-1).float()
        neg_his_vectors = self.iid_embeddings_neg(new_hist.abs()) * valid_his.unsqueeze(dim=-1).float()
        his_vectors = pos_his_vectors * his_pos_neg + (-his_pos_neg + 1) * neg_his_vectors
        
        # sort
        sorted_his_length, sorted_idx = torch.topk(his_length, k=new_hist.shape[0])
        sorted_his_vectors = his_vectors.index_select(dim=0, index=sorted_idx)
        
        # pack
        packed_his_vectors = torch.nn.utils.rnn.pack_padded_sequence(sorted_his_vectors, sorted_his_length.cpu(), batch_first=True)
        
        # rnn
        out_put, hidden = self.rnn(packed_his_vectors, None)
        sorted_rnn_vectors = self.out(hidden[-1])
        
        # unsort
        unsorted_idx = torch.topk(sorted_idx, k=new_hist.shape[0], largest=False)[1]
        rnn_vector = sorted_rnn_vectors.index_select(dim=0, index=unsorted_idx)
        
        # predict
        item_vec = self.iid_embeddings(torch.cat([iids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))
        prediction = (rnn_vector * item_vec).sum(dim=1).view([-1])
        
        prediction = prediction.reshape(hist.shape[0], -1)
        prob = [self.fact_prob] + [(1-self.fact_prob)/self.ctf_num] * self.ctf_num
        prob = torch.tensor(prob).to(self.dummy_param.device)
        prediction = (prediction * prob).sum(dim=1)
        
        
        
        
        
        
#         pdb.set_trace()
#         his_pos_neg = hist.ge(0).unsqueeze(-1).float()
        
#         hist_list = [hist] * (self.ctf_num + 1)
#         new_hist = torch.cat(hist_list, axis=1).reshape(-1, hist.shape[1])
        
        
        
#         prev_iid = hist[:,-1]
#         hist_list = [hist] * (self.ctf_num + 1)
#         new_hist = torch.cat(hist_list, axis=1).reshape(-1, hist.shape[1])
        
#         item_sim = torch.matmul(self.iid_embeddings(prev_iid), self.iid_embeddings.weight.t())
#         sample_item = torch.topk(item_sim, self.ctf_num+1).indices
#         new_prev_iid = torch.cat((prev_iid.view(-1,1), sample_item[:,1:]), axis=1).reshape(-1)
#         new_hist[:,-1] = new_prev_iid
        
# #         pdb.set_trace()
#         his_vec = self.iid_embeddings(new_hist)
#         output, hidden = self.rnn(his_vec)
#         rnn_vec = self.out(hidden[-1])
#         item_vec = self.iid_embeddings(torch.cat([iids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))
#         prediction = (rnn_vec * item_vec).sum(dim=1).view([-1])
#         prediction = prediction.reshape(hist.shape[0], -1)
#         prob = [self.fact_prob] + [(1-self.fact_prob)/self.ctf_num] * self.ctf_num
#         prob = torch.tensor(prob).to(self.dummy_param.device)
#         prediction = (prediction * prob).sum(dim=1)
        
        out_dict = {'prediction': prediction, 'uid': uids, 'label': label, 'iid': iids}
        return out_dict
        
    def estimate(self, batch):
        """
        estimation for training
        """
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        hist = batch['history'].to(torch.long).to(self.dummy_param.device)
        negs = batch['negative'].to(torch.long).to(self.dummy_param.device).view([-1])
        
        
        valid_his = hist.abs().gt(0).long()
        pos_his = hist.ge(0).long()
        
        # counterfactual histories
        last_valid_idx = torch.topk(valid_his * torch.tensor([i for i in range(valid_his.shape[1])]).to(self.dummy_param.device),k=1)[1]
        last_valid_mask = F.one_hot(last_valid_idx, num_classes=valid_his.shape[1]).sum(dim=1)
        
        last_pos_idx = torch.topk(pos_his * torch.tensor([i for i in range(valid_his.shape[1])]).to(self.dummy_param.device),k=1)[1]
#         last_pos_mask = F.one_hot(last_pos_idx, num_classes=valid_his.shape[1]).sum(dim=1)
        last_pos_mask = last_valid_mask.clone()
        
        ctf_pref_his = (-1 * last_valid_mask + (1 - last_valid_mask)) * hist
        
        last_pos_iid = (hist * last_pos_mask).sum(dim=1)
        
        item_sim = torch.matmul(F.normalize(self.iid_embeddings(last_pos_iid.abs().view([-1])),dim=0,p=2), F.normalize(self.iid_embeddings.weight[1:],dim=0,p=2).t())
        sample_item = torch.topk(-item_sim, self.ctf_num-1).indices+1
        ctf_item_his = sample_item.unsqueeze(dim=2) * last_pos_mask.unsqueeze(1) + (hist * (1-last_pos_mask)).unsqueeze(1)
        
        new_hist = torch.cat([hist, ctf_pref_his, ctf_item_his.reshape(hist.shape[0], -1)], axis=1).reshape(-1, hist.shape[1])
        
        # get prediction for factual histories and counterfactual histories
        valid_his = new_hist.abs().gt(0).long()
        his_length = valid_his.sum(dim=1)
        
        his_pos_neg = new_hist.ge(0).unsqueeze(-1).float()
        
        pos_his_vectors = self.iid_embeddings(new_hist.abs()) * valid_his.unsqueeze(dim=-1).float()
        neg_his_vectors = self.iid_embeddings_neg(new_hist.abs()) * valid_his.unsqueeze(dim=-1).float()
        his_vectors = pos_his_vectors * his_pos_neg + (-his_pos_neg + 1) * neg_his_vectors
        
        # sort
        sorted_his_length, sorted_idx = torch.topk(his_length, k=new_hist.shape[0])
        sorted_his_vectors = his_vectors.index_select(dim=0, index=sorted_idx)
        
        # pack
        try:
            packed_his_vectors = torch.nn.utils.rnn.pack_padded_sequence(sorted_his_vectors, sorted_his_length.cpu(), batch_first=True)
        except:
            pdb.set_trace()
        
        # rnn
        out_put, hidden = self.rnn(packed_his_vectors, None)
        sorted_rnn_vectors = self.out(hidden[-1])
        
        # unsort
        unsorted_idx = torch.topk(sorted_idx, k=new_hist.shape[0], largest=False)[1]
        rnn_vector = sorted_rnn_vectors.index_select(dim=0, index=unsorted_idx)
        
        prob = [self.fact_prob] + [(1-self.fact_prob)/self.ctf_num] * self.ctf_num
        prob = torch.tensor(prob).to(self.dummy_param.device)
#         pdb.set_trace()
        pos_item_vec = self.iid_embeddings(torch.cat([iids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))
        pos_prediction = (rnn_vector * pos_item_vec).sum(dim=1).view([-1])
        pos_prediction = pos_prediction.reshape(hist.shape[0], -1)
        pos_prediction = (pos_prediction * prob).sum(dim=1)
        
        neg_item_vec = self.iid_embeddings(torch.cat([negs.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))
        neg_prediction = (rnn_vector * neg_item_vec).sum(dim=1).view([-1])
        neg_prediction = neg_prediction.reshape(hist.shape[0], -1)
        neg_prediction = (neg_prediction * prob).sum(dim=1)
        
        out_dict = {'pos_prediction': pos_prediction, 'neg_prediction': neg_prediction}
        return out_dict