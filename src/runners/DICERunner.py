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

class DICERunner(BaseRunner):
    def parse_runner_args(parser):
        parser.add_argument('--int_weight', type=float, default=0.1,
                            help='Weight of interest loss.')
        parser.add_argument('--pop_weight', type=float, default=0.1,
                            help='Weight of conformity loss.')
        parser.add_argument('--dis_weight', type=float, default=0.01,
                            help='Weight of discrepency loss.')
        return BaseRunner.parse_runner_args(parser)
    
    def __init__(self, int_weight, pop_weight, dis_weight, *args, **kwargs):
        self.int_weight = int_weight
        self.pop_weight = pop_weight
        self.dis_weight = dis_weight
        
        BaseRunner.__init__(self, *args, **kwargs)
        
    def fit(self, model, data):
        losses = []
        dataLoader = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=16)
        model.train()
        pbar = tqdm(total=len(data), leave=False, ncols=100, mininterval=1, desc='Predict')
        for i, batchData in enumerate(dataLoader):
            pbar.update(batchData['uid'].shape[0])
            self.optimizer.zero_grad()
            out_dict = model(batchData)
            loss = out_dict['loss'] + model.l2() * self.l2 + out_dict['int_loss'] * self.int_weight + out_dict['pop_loss'] * self.pop_weight - out_dict['discrepency_loss'] * self.dis_weight
            losses.append(loss.detach().cpu())
            loss.backward()
            self.optimizer.step()
        pbar.close()
        model.eval()
        return np.mean(losses)