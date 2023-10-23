from models.BaseModel import BaseModel
import torch
import torch.nn.functional as F
import pdb

class GRU4Rec(BaseModel):
    loader = 'HistLoader'
    runner = 'BaseRunner'
    
    def parse_model_args(parser):
        parser.add_argument('--hidden_size', default=64, type=int,
                           help='Size of hidden vectors in GRU')
        parser.add_argument('--num_layers', default=1, type=int,
                           help='Number of GRU layers')
        parser.add_argument('--emb_size', default=64, type=int,
                           help='Size of embedding')
        return BaseModel.parse_model_args(parser)
    
    def __init__(self, hidden_size, num_layers, emb_size, *args, **kwargs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb_size = emb_size
        BaseModel.__init__(self, *args, **kwargs)
        
    def _init_weight(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.iid_embeddings_neg = torch.nn.Embedding(self.item_num, self.emb_size)
        self.rnn = torch.nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.out = torch.nn.Linear(self.hidden_size, self.emb_size, bias=False)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        
        return
    
    def copy_params(self, model):
        self.load_state_dict(model.state_dict())
    
    def predict(self, batch):
        """
        prediction for evaluation
        """
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
        
        # sort
        sorted_his_length, sorted_idx = torch.topk(his_length, k=hist.shape[0])
        sorted_his_vectors = his_vectors.index_select(dim=0, index=sorted_idx)
        
        # pack
        packed_his_vectors = torch.nn.utils.rnn.pack_padded_sequence(sorted_his_vectors, sorted_his_length.cpu(), batch_first=True)
        
        # rnn
        out_put, hidden = self.rnn(packed_his_vectors, None)
        sorted_rnn_vectors = self.out(hidden[-1])
        
        # unsort
        unsorted_idx = torch.topk(sorted_idx, k=hist.shape[0], largest=False)[1]
        rnn_vector = sorted_rnn_vectors.index_select(dim=0, index=unsorted_idx)
        
        # predict
        item_vec = self.iid_embeddings(iids)
        prediction = (rnn_vector * item_vec).sum(dim=1).view([-1])
        
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
        
        # sort
        sorted_his_length, sorted_idx = torch.topk(his_length, k=hist.shape[0])
        sorted_his_vectors = his_vectors.index_select(dim=0, index=sorted_idx)
        
        # pack
        packed_his_vectors = torch.nn.utils.rnn.pack_padded_sequence(sorted_his_vectors, sorted_his_length.cpu(), batch_first=True)
        
        # rnn
        out_put, hidden = self.rnn(packed_his_vectors, None)
        sorted_rnn_vectors = self.out(hidden[-1])
        
        # unsort
        unsorted_idx = torch.topk(sorted_idx, k=hist.shape[0], largest=False)[1]
        rnn_vector = sorted_rnn_vectors.index_select(dim=0, index=unsorted_idx)
        
        # predict
        pos_item_vec = self.iid_embeddings(iids)
        pos_prediction = (rnn_vector * pos_item_vec).sum(dim=1).view([-1])
        neg_item_vec = self.iid_embeddings(negs)
        neg_prediction = (rnn_vector * neg_item_vec).sum(dim=1).view([-1])
        
        out_dict = {'pos_prediction': pos_prediction, 'neg_prediction': neg_prediction}
        return out_dict
    
    def forward(self, batch):
        """
        calculate the loss
        """
        out_dict = self.estimate(batch)
        pos, neg = out_dict['pos_prediction'], out_dict['neg_prediction']
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        return out_dict
        