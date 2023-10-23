import torch
import torch.nn.functional as F
from models.GRU4Rec import GRU4Rec


class STAMP(GRU4Rec):

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--attention_size', type=int, default=64,
                            help='Size of attention hidden space.')
        return GRU4Rec.parse_model_args(parser)

    def __init__(self, attention_size, *args, **kwargs):
        self.attention_size = attention_size
        GRU4Rec.__init__(self, *args, **kwargs)

    def _init_weight(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.iid_embeddings_neg = torch.nn.Embedding(self.item_num, self.emb_size)

        self.attention_wxi = torch.nn.Linear(self.emb_size, self.attention_size, bias=True)
        self.attention_wxt = torch.nn.Linear(self.emb_size, self.attention_size, bias=True)
        self.attention_wms = torch.nn.Linear(self.emb_size, self.attention_size, bias=True)
        self.attention_out = torch.nn.Linear(self.attention_size, 1, bias=False)

        self.mlp_a = torch.nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.mlp_b = torch.nn.Linear(self.emb_size, self.emb_size, bias=True)
        
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def predict(self, batch):
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        hist = batch['history'].to(torch.long).to(self.dummy_param.device)
        label = batch['rating'].to(torch.long).to(self.dummy_param.device).view([-1])
        
        valid_his = hist.abs().gt(0).long()
        his_length = valid_his.sum(dim=1)
        
        his_pos_neg = hist.ge(0).unsqueeze(-1).float()
        
        pos_his_vectors = self.iid_embeddings(hist.abs()) * valid_his.unsqueeze(dim=-1).float()
        neg_his_vectors = self.iid_embeddings_neg(hist.abs()) * valid_his.unsqueeze(dim=-1).float()
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
        item_vec = self.iid_embeddings(iids)  # B * V

        pred_vector = hs * ht  # B * V
        prediction = (pred_vector * item_vec).sum(dim=-1).view([-1])

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
        his_length = valid_his.sum(dim=1)
        
        his_pos_neg = hist.ge(0).unsqueeze(-1).float()
        
        pos_his_vectors = self.iid_embeddings(hist.abs()) * valid_his.unsqueeze(dim=-1).float()
        neg_his_vectors = self.iid_embeddings_neg(hist.abs()) * valid_his.unsqueeze(dim=-1).float()
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
        
        # predict
        pos_item_vec = self.iid_embeddings(iids)
        pos_prediction = (pred_vector * pos_item_vec).sum(dim=-1).view([-1])
        neg_item_vec = self.iid_embeddings(negs)
        neg_prediction = (pred_vector * neg_item_vec).sum(dim=-1).view([-1])
        
        out_dict = {'pos_prediction': pos_prediction, 'neg_prediction': neg_prediction}
        return out_dict