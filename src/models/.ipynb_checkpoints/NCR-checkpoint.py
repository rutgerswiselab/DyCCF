import torch
import torch.nn.functional as F
from models.BaseModel import BaseModel
from utils import utils
import numpy as np
import pdb


class NCR(BaseModel):
    loader = 'HistLoader'
    runner = 'BaseRunner'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', default=64, type=int,
                           help='Size of embedding')
        parser.add_argument('--r_weight', type=float, default=10,
                            help='Weight of logic regularizer loss')
        parser.add_argument('--ppl_weight', type=float, default=0,
                            help='Weight of uv interaction prediction loss')
        parser.add_argument('--pos_weight', type=float, default=0,
                            help='Weight of positive purchase loss')
        return BaseModel.parse_model_args(parser)

    def __init__(self, emb_size, r_weight, ppl_weight, pos_weight, *args, **kwargs):
        self.r_weight = r_weight
        self.ppl_weight = ppl_weight
        self.pos_weight = pos_weight
        self.emb_size = emb_size
        BaseModel.__init__(self, *args, **kwargs)

    def _init_weight(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.emb_size)
        self.true = torch.nn.Parameter(torch.tensor(np.random.uniform(0, 0.1, size=self.emb_size).astype(np.float32)), requires_grad=False)
        self.not_layer_1 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.not_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.and_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)
        self.and_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.or_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)
        self.or_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.purchase_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)
        self.purchase_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def logic_not(self, vector):
        vector = F.relu(self.not_layer_1(vector))
        vector = self.not_layer_2(vector)
        return vector

    def logic_and(self, vector1, vector2):
        assert(len(vector1.size()) == len(vector2.size()))
        vector = torch.cat((vector1, vector2), dim=(len(vector1.size()) - 1))
        vector = F.relu(self.and_layer_1(vector))
        vector = self.and_layer_2(vector)
        return vector

    def logic_or(self, vector1, vector2):
        assert (len(vector1.size()) == len(vector2.size()))
        vector = torch.cat((vector1, vector2), dim=(len(vector1.size()) - 1))
        vector = F.relu(self.or_layer_1(vector))
        vector = self.or_layer_2(vector)
        return vector

    def purchase_gate(self, uv_vector):
        uv_vector = F.relu(self.purchase_layer_1(uv_vector))
        uv_vector = self.purchase_layer_2(uv_vector)
        return uv_vector

    # def logic_output(self, vector):
    def mse(self, vector1, vector2):
        return ((vector1 - vector2) ** 2).mean()

    def predict(self, batch):
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        hist = batch['history'].to(torch.long).to(self.dummy_param.device)
        label = batch['rating'].to(torch.long).to(self.dummy_param.device).view([-1])
        
        valid_his = hist.abs().gt(0).long()
#         his_length = valid_his.sum(dim=1)
        his_length = valid_his.shape[1]

        his_pos_neg = hist.ge(0).unsqueeze(-1).float()

        # user/item vectors shape: (batch_size, embedding_size)
        user_vectors = self.uid_embeddings(uids)
        item_vectors = self.iid_embeddings(iids)

        # concat iids with uids and send to purchase gate to prepare for logic_and gate
        item_vectors = torch.cat((user_vectors, item_vectors), dim=1)
        item_vectors = self.purchase_gate(item_vectors)

        # expand user vector to prepare for concatenating with history item vectors
        uh_vectors = user_vectors.view(user_vectors.size(0), 1, user_vectors.size(1))
        uh_vectors = uh_vectors.expand(his_pos_neg.size(0), his_pos_neg.size(1), uh_vectors.size(2))

        # history item purchase hidden factors shape: (batch, user, embedding)
        his_vectors = self.iid_embeddings(hist.abs())

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
        
        out_dict = {'prediction': prediction, 'uid': uids, 'label': label, 'iid': iids}
        return out_dict
    
    def estimate(self, batch):
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        hist = batch['history'].to(torch.long).to(self.dummy_param.device)
        negs = batch['negative'].to(torch.long).to(self.dummy_param.device).view([-1])
        
        
        valid_his = hist.abs().gt(0).long()
        batch_size = valid_his.shape[0]
        his_length = valid_his.shape[1]


        his_pos_neg = hist.ge(0).unsqueeze(-1).float()

        # user/item vectors shape: (batch_size, embedding_size)
        user_vectors = self.uid_embeddings(uids)
        item_vectors = self.iid_embeddings(iids)

        # concat iids with uids and send to purchase gate to prepare for logic_and gate
        item_vectors = torch.cat((user_vectors, item_vectors), dim=1)
        item_vectors = self.purchase_gate(item_vectors)

        # expand user vector to prepare for concatenating with history item vectors
        uh_vectors = user_vectors.view(user_vectors.size(0), 1, user_vectors.size(1))
        uh_vectors = uh_vectors.expand(his_pos_neg.size(0), his_pos_neg.size(1), uh_vectors.size(2))

        # history item purchase hidden factors shape: (batch, user, embedding)
        his_vectors = self.iid_embeddings(hist.abs())

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
        
        
        """negative samples"""
        # user/item vectors shape: (batch_size, embedding_size)
        user_vectors = self.uid_embeddings(uids)
        item_vectors = self.iid_embeddings(negs)

        # concat iids with uids and send to purchase gate to prepare for logic_and gate
        item_vectors = torch.cat((user_vectors, item_vectors), dim=1)
        item_vectors = self.purchase_gate(item_vectors)

        # expand user vector to prepare for concatenating with history item vectors
        uh_vectors = user_vectors.view(user_vectors.size(0), 1, user_vectors.size(1))
        uh_vectors = uh_vectors.expand(his_pos_neg.size(0), his_pos_neg.size(1), uh_vectors.size(2))

        # history item purchase hidden factors shape: (batch, user, embedding)
        his_vectors = self.iid_embeddings(hist.abs())

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
        
        constraint = torch.cat(constraint, dim=1)
        out_dict = {'pos_prediction': pos_prediction, 'neg_prediction': neg_prediction, 'constraint': constraint}
        return out_dict

    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失
        """
        out_dict = self.estimate(feed_dict)
        false = self.logic_not(self.true).view(1, -1)
        constraint = out_dict['constraint']

        # regularizer
        dim = len(constraint.size())-1

        # length constraint
        # r_length = constraint.norm(dim=dim)()

        # not
        r_not_not_true = (1 - F.cosine_similarity(self.logic_not(self.logic_not(self.true)), self.true, dim=0)).sum()
        r_not_not_self = \
            (1 - F.cosine_similarity(self.logic_not(self.logic_not(constraint)), constraint, dim=dim)).mean()
        r_not_self = (1 + F.cosine_similarity(self.logic_not(constraint), constraint, dim=dim)).mean()

        r_not_self = (1 + F.cosine_similarity(self.logic_not(constraint), constraint, dim=dim)).mean()

        r_not_not_not = \
            (1 + F.cosine_similarity(self.logic_not(self.logic_not(constraint)), self.logic_not(constraint), dim=dim)).mean()

        # and
        r_and_true = (1 - F.cosine_similarity(
            self.logic_and(constraint, self.true.expand_as(constraint)), constraint, dim=dim)).mean()
        r_and_false = (1 - F.cosine_similarity(
            self.logic_and(constraint, false.expand_as(constraint)), false.expand_as(constraint), dim=dim)).mean()
        r_and_self = (1 - F.cosine_similarity(self.logic_and(constraint, constraint), constraint, dim=dim)).mean()

        # NEW ADDED REG NEED TO TEST
        r_and_not_self = (1 - F.cosine_similarity(
            self.logic_and(constraint, self.logic_not(constraint)), false.expand_as(constraint), dim=dim)).mean()
        r_and_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_and(self.logic_not(constraint), constraint), false.expand_as(constraint), dim=dim)).mean()

        # or
        r_or_true = (1 - F.cosine_similarity(
            self.logic_or(constraint, self.true.expand_as(constraint)), self.true.expand_as(constraint), dim=dim))\
            .mean()
        r_or_false = (1 - F.cosine_similarity(
            self.logic_or(constraint, false.expand_as(constraint)), constraint, dim=dim)).mean()
        r_or_self = (1 - F.cosine_similarity(self.logic_or(constraint, constraint), constraint, dim=dim)).mean()

        r_or_not_self = (1 - F.cosine_similarity(
            self.logic_or(constraint, self.logic_not(constraint)), self.true.expand_as(constraint), dim=dim)).mean()
        r_or_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_or(self.logic_not(constraint), constraint), self.true.expand_as(constraint), dim=dim)).mean()

        # True/False
        true_false = 1 + F.cosine_similarity(self.true, false.view(-1), dim=0)

        r_loss = r_not_not_true + r_not_not_self + r_not_self + \
                 r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse + \
                 r_or_true + r_or_false + r_or_self + true_false + r_or_not_self + r_or_not_self_inverse + r_not_not_not
        r_loss = r_loss * self.r_weight

        # pos_loss = None
        # recommendation loss
        pos, neg = out_dict['pos_prediction'], out_dict['neg_prediction']
        loss = -(pos - neg).sigmoid().log().sum()

        loss = loss + r_loss 
        out_dict['loss'] = loss
        return out_dict
    
    def copy_params(self, model):
        self.load_state_dict(model.state_dict())