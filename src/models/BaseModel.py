import torch
import torch.nn as nn
import numpy as np
import os
import logging

class BaseModel(nn.Module):
    def parse_model_args(parser):
        parser.add_argument('--model_path', default='../../model/model.pt', type=str,
                           help='model path')
        return parser
    
    def __init__(self, user_num, item_num, model_path, seed):
        super(BaseModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.model_path = model_path
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._init_weight()
        
    def _init_weight(self):
        return
    
    def predict(self, batch):
        return
    
    def estimate(self, batch):
        return
    
    def forward(self, batch):
        return
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save model to ' + model_path)
        
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info('Load model from ' + model_path)
        
        
    def l2(self):
        l2 = 0
        for p in self.parameters():
            l2 += (p ** 2).sum()
            
        return l2