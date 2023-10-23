import torch
import torch.nn.functional as F
from models.NCR import NCR
from utils import utils
import numpy as np
import pdb

class DyccfNCR(NCR):
    loader = 'DccfHistLoader'
    runner = 'DccfRunner'
    
    def parse_model_args(parser):
        parser.add_argument('--ctf_num', default=5, type=int,
                           help='number of counterfactual histories')
        parser.add_argument('--fact_prob', default=0.3, type=float,
                           help='the probability of factual history')
        return NCR.parse_model_args(parser)

    def __init__(self, ctf_num, fact_prob, *args, **kwargs):
        self.ctf_num = ctf_num
        self.fact_prob = fact_prob
        NCR.__init__(self, *args, **kwargs)
        
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
        
#         pdb.set_trace()
        
        item_sim = torch.matmul(F.normalize(self.iid_embeddings(last_pos_iid.abs().view([-1])),dim=0,p=2), F.normalize(self.iid_embeddings.weight[1:],dim=0,p=2).t())
        sample_item = torch.topk(-item_sim, self.ctf_num-1).indices+1
        ctf_item_his = sample_item.unsqueeze(dim=2) * last_pos_mask.unsqueeze(1) + (hist * (1-last_pos_mask)).unsqueeze(1)
        
        new_hist = torch.cat([hist, ctf_pref_his, ctf_item_his.reshape(hist.shape[0], -1)], axis=1).reshape(-1, hist.shape[1])
        
        #####
        
        valid_his = new_hist.abs().gt(0).long()
#         his_length = valid_his.sum(dim=1)
        his_length = valid_his.shape[1]

        his_pos_neg = new_hist.ge(0).unsqueeze(-1).float()

        # user/item vectors shape: (batch_size, embedding_size)
        user_vectors = self.uid_embeddings(torch.cat([uids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))
        item_vectors = self.iid_embeddings(torch.cat([iids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))

        # concat iids with uids and send to purchase gate to prepare for logic_and gate
        item_vectors = torch.cat((user_vectors, item_vectors), dim=1)
        item_vectors = self.purchase_gate(item_vectors)

        # expand user vector to prepare for concatenating with history item vectors
        uh_vectors = user_vectors.view(user_vectors.size(0), 1, user_vectors.size(1))
        uh_vectors = uh_vectors.expand(his_pos_neg.size(0), his_pos_neg.size(1), uh_vectors.size(2))

        # history item purchase hidden factors shape: (batch, user, embedding)
        his_vectors = self.iid_embeddings(new_hist.abs())

        # concatenate user embedding with history item embeddings
        his_vectors = torch.cat((uh_vectors, his_vectors), dim=2)

        # True/False representation of user item interactions
        his_vectors = self.purchase_gate(his_vectors)
        not_his_vectors = self.logic_not(his_vectors)


        his_vectors = his_pos_neg * his_vectors + (1 - his_pos_neg) * not_his_vectors
        

        tmp_vector = self.logic_not(his_vectors[:, 0])
#         pdb.set_trace()
        shuffled_history_idx = [i for i in range(1, his_length)]
        np.random.shuffle(shuffled_history_idx)
        for i in shuffled_history_idx:
            tmp_vector = self.logic_or(tmp_vector, self.logic_not(his_vectors[:, i]))
        left_vector = tmp_vector

        right_vector = item_vectors
        sent_vector = self.logic_or(left_vector, right_vector)
        prediction = F.cosine_similarity(sent_vector, self.true.view([1, -1])) * 10
        
        prediction = prediction.reshape(hist.shape[0], -1)
        prob = [self.fact_prob] + [(1-self.fact_prob)/self.ctf_num] * self.ctf_num
        prob = torch.tensor(prob).to(self.dummy_param.device)
        prediction = (prediction * prob).sum(dim=1)
        
        out_dict = {'prediction': prediction, 'uid': uids, 'label': label, 'iid': iids}
        return out_dict
    
    def estimate(self, batch):
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
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
        
#         pdb.set_trace()
        
        item_sim = torch.matmul(F.normalize(self.iid_embeddings(last_pos_iid.abs().view([-1])),dim=0,p=2), F.normalize(self.iid_embeddings.weight[1:],dim=0,p=2).t())
        sample_item = torch.topk(-item_sim, self.ctf_num-1).indices+1
        ctf_item_his = sample_item.unsqueeze(dim=2) * last_pos_mask.unsqueeze(1) + (hist * (1-last_pos_mask)).unsqueeze(1)
        
        new_hist = torch.cat([hist, ctf_pref_his, ctf_item_his.reshape(hist.shape[0], -1)], axis=1).reshape(-1, hist.shape[1])
        
        # original model
        
        prob = [self.fact_prob] + [(1-self.fact_prob)/self.ctf_num] * self.ctf_num
        prob = torch.tensor(prob).to(self.dummy_param.device)
        
        valid_his = new_hist.abs().gt(0).long()
        batch_size = valid_his.shape[0]
        his_length = valid_his.shape[1]


        his_pos_neg = new_hist.ge(0).unsqueeze(-1).float()

        # user/item vectors shape: (batch_size, embedding_size)
        user_vectors = self.uid_embeddings(torch.cat([uids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))
        item_vectors = self.iid_embeddings(torch.cat([iids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))

        # concat iids with uids and send to purchase gate to prepare for logic_and gate
        item_vectors = torch.cat((user_vectors, item_vectors), dim=1)
        item_vectors = self.purchase_gate(item_vectors)

        # expand user vector to prepare for concatenating with history item vectors
        uh_vectors = user_vectors.view(user_vectors.size(0), 1, user_vectors.size(1))
        uh_vectors = uh_vectors.expand(his_pos_neg.size(0), his_pos_neg.size(1), uh_vectors.size(2))

        # history item purchase hidden factors shape: (batch, user, embedding)
        his_vectors = self.iid_embeddings(new_hist.abs())

        # concatenate user embedding with history item embeddings
        his_vectors = torch.cat((uh_vectors, his_vectors), dim=2)

        # True/False representation of user item interactions
        his_vectors = self.purchase_gate(his_vectors)
        not_his_vectors = self.logic_not(his_vectors)

        constraint = list([his_vectors])
        constraint.append(not_his_vectors)

        his_vectors = his_pos_neg * his_vectors + (1 - his_pos_neg) * not_his_vectors

        tmp_vector = self.logic_not(his_vectors[:, 0])
        shuffled_history_idx = [i for i in range(1, his_length)]
        np.random.shuffle(shuffled_history_idx)
        for i in shuffled_history_idx:
            tmp_vector = self.logic_or(tmp_vector, self.logic_not(his_vectors[:, i]))
            constraint.append(tmp_vector.view(batch_size, -1, self.emb_size))
        left_vector = tmp_vector
        # constraint.append(left_vector.view(batch_size, -1, self.ui_vector_size))

        right_vector = item_vectors
        constraint.append(right_vector.view(batch_size, -1, self.emb_size))
        sent_vector = self.logic_or(left_vector, right_vector)
        constraint.append(sent_vector.view(batch_size, -1, self.emb_size))
        pos_prediction = F.cosine_similarity(sent_vector, self.true.view([1, -1])) * 10
        pos_prediction = pos_prediction.reshape(hist.shape[0], -1)
        pos_prediction = (pos_prediction * prob).sum(dim=1)
        
        """negative samples"""
        # user/item vectors shape: (batch_size, embedding_size)
        user_vectors = self.uid_embeddings(torch.cat([uids.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))
        item_vectors = self.iid_embeddings(torch.cat([negs.view(-1,1)]*(self.ctf_num+1), axis=1).reshape(-1))

        # concat iids with uids and send to purchase gate to prepare for logic_and gate
        item_vectors = torch.cat((user_vectors, item_vectors), dim=1)
        item_vectors = self.purchase_gate(item_vectors)

        # expand user vector to prepare for concatenating with history item vectors
        uh_vectors = user_vectors.view(user_vectors.size(0), 1, user_vectors.size(1))
        uh_vectors = uh_vectors.expand(his_pos_neg.size(0), his_pos_neg.size(1), uh_vectors.size(2))

        # history item purchase hidden factors shape: (batch, user, embedding)
        his_vectors = self.iid_embeddings(new_hist.abs())

        # concatenate user embedding with history item embeddings
        his_vectors = torch.cat((uh_vectors, his_vectors), dim=2)

        # True/False representation of user item interactions
        his_vectors = self.purchase_gate(his_vectors)
        not_his_vectors = self.logic_not(his_vectors)

        constraint = list([his_vectors])
        constraint.append(not_his_vectors)

        his_vectors = his_pos_neg * his_vectors + (1 - his_pos_neg) * not_his_vectors

        tmp_vector = self.logic_not(his_vectors[:, 0])
        shuffled_history_idx = [i for i in range(1, his_length)]
        np.random.shuffle(shuffled_history_idx)
        for i in shuffled_history_idx:
            tmp_vector = self.logic_or(tmp_vector, self.logic_not(his_vectors[:, i]))
            constraint.append(tmp_vector.view(batch_size, -1, self.emb_size))
        left_vector = tmp_vector
        # constraint.append(left_vector.view(batch_size, -1, self.ui_vector_size))

        right_vector = item_vectors
        constraint.append(right_vector.view(batch_size, -1, self.emb_size))
        sent_vector = self.logic_or(left_vector, right_vector)
        constraint.append(sent_vector.view(batch_size, -1, self.emb_size))
        neg_prediction = F.cosine_similarity(sent_vector, self.true.view([1, -1])) * 10
        neg_prediction = neg_prediction.reshape(hist.shape[0], -1)
        neg_prediction = (neg_prediction * prob).sum(dim=1)
        
        constraint = torch.cat(constraint, dim=1)
        out_dict = {'pos_prediction': pos_prediction, 'neg_prediction': neg_prediction, 'constraint': constraint}
        return out_dict