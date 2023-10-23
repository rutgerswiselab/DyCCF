from models.STAMP import STAMP
import numpy as np
import os
import torch
import torch.nn.functional as F
import pdb


class DyccfSTAMP(STAMP):
    loader = 'DccfHistLoader'
    runner = 'DccfRunner'
    
    def parse_model_args(parser):
        parser.add_argument('--ctf_num', default=5, type=int,
                           help='number of counterfactual histories')
        parser.add_argument('--fact_prob', default=0.3, type=float,
                           help='the probability of factual history')
        return STAMP.parse_model_args(parser)

    def __init__(self, ctf_num, fact_prob, *args, **kwargs):
        self.ctf_num = ctf_num
        self.fact_prob = fact_prob
        STAMP.__init__(self, *args, **kwargs)
        

    def predict(self, batch):
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

        # Prepare Vectors
        xt = his_vectors[range(len(his_length)), his_length - 1, :]  # B * V
        ms = his_vectors.sum(dim=1) / his_length.view([-1, 1]).float()  # B * V

        # Attention
        att_wxi_v = self.attention_wxi(his_vectors)  # B * H * V
        att_wxt_v = self.attention_wxt(xt).unsqueeze(dim=1)  # B * 1 * V
        att_wms_v = self.attention_wms(ms).unsqueeze(dim=1)  # B * 1 * V
        att_v = self.attention_out((att_wxi_v + att_wxt_v + att_wms_v).sigmoid())  # B * H * 1
        ma = (his_vectors * att_v * valid_his.unsqueeze(dim=-1).float()).sum(dim=1)  # B * V

        # Output Layer
        hs = self.mlp_a(ma).tanh()  # B * V
        ht = self.mlp_b(xt).tanh()  # B * V
        item_vec = self.iid_embeddings(torch.cat([iids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))

        pred_vector = hs * ht  # B * V
        prediction = (pred_vector * item_vec).sum(dim=-1).view([-1])
        
        prediction = prediction.reshape(hist.shape[0], -1)
        prob = [self.fact_prob] + [(1-self.fact_prob)/self.ctf_num] * self.ctf_num
        prob = torch.tensor(prob).to(self.dummy_param.device)
        prediction = (prediction * prob).sum(dim=1)

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
        
        # Prepare Vectors
        xt = his_vectors[range(len(his_length)), his_length - 1, :]  # B * V
        ms = his_vectors.sum(dim=1) / his_length.view([-1, 1]).float()  # B * V

        # Attention
        att_wxi_v = self.attention_wxi(his_vectors)  # B * H * V
        att_wxt_v = self.attention_wxt(xt).unsqueeze(dim=1)  # B * 1 * V
        att_wms_v = self.attention_wms(ms).unsqueeze(dim=1)  # B * 1 * V
        att_v = self.attention_out((att_wxi_v + att_wxt_v + att_wms_v).sigmoid())  # B * H * 1
        ma = (his_vectors * att_v * valid_his.unsqueeze(dim=-1).float()).sum(dim=1)  # B * V

        # Output Layer
        hs = self.mlp_a(ma).tanh()  # B * V
        ht = self.mlp_b(xt).tanh()  # B * V

        pred_vector = hs * ht  # B * V
        
        prob = [self.fact_prob] + [(1-self.fact_prob)/self.ctf_num] * self.ctf_num
        prob = torch.tensor(prob).to(self.dummy_param.device)
        
        # predict
        pos_item_vec = self.iid_embeddings(torch.cat([iids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))
        pos_prediction = (pred_vector * pos_item_vec).sum(dim=-1).view([-1])
        pos_prediction = pos_prediction.reshape(hist.shape[0], -1)
        pos_prediction = (pos_prediction * prob).sum(dim=1)
        
        neg_item_vec = self.iid_embeddings(torch.cat([negs.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))
        neg_prediction = (pred_vector * neg_item_vec).sum(dim=-1).view([-1])
        neg_prediction = neg_prediction.reshape(hist.shape[0], -1)
        neg_prediction = (neg_prediction * prob).sum(dim=1)
        
        out_dict = {'pos_prediction': pos_prediction, 'neg_prediction': neg_prediction}
        return out_dict