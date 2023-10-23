from runners.BaseRunner import BaseRunner
import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as opt
from time import time
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from collections import defaultdict
from utils import evaluator
from utils import utils

class DyccfRunner(BaseRunner):
    def parse_runner_args(parser):
        parser.add_argument('--sim_coef', default=0.0, type=float,
                           help='The coefficiency of matrix similarity')
        parser.add_argument('--chron', default=1, type=int,
                           help='Chronalogically train or not')
        return BaseRunner.parse_runner_args(parser)
    
    def __init__(self, sim_coef, chron, *args, **kwargs):
        self.chron = chron
        self.sim_coef = sim_coef
        BaseRunner.__init__(self, *args, **kwargs)
        
    def fit(self, model, data):
        losses = []
        if self.chron == 1:
            dataLoader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=16)
        else:
            dataLoader = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=16)
        model.train()
        pbar = tqdm(total=len(data), leave=False, ncols=100, mininterval=1, desc='Predict')
        for i, batchData in enumerate(dataLoader):
            pbar.update(batchData['uid'].shape[0])
            self.optimizer.zero_grad()
            out_dict = model(batchData)
            loss = out_dict['loss'] + model.l2() * self.l2 
            loss += torch.matmul(model.iid_embeddings.weight, model.iid_embeddings.weight.t()).norm() * self.sim_coef
            losses.append(loss.detach().cpu())
            loss.backward()
            self.optimizer.step()
        pbar.close()
        model.eval()
        return np.mean(losses)