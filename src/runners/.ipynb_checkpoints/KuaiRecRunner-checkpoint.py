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

import pdb

class KuaiRecRunner(BaseRunner):
    def parse_runner_args(parser):
        parser.add_argument('--max_round', type=int, default=100,
                            help='max round of interaction during the testing.')
        parser.add_argument('--N', type=int, default=2,
                            help='exit mechanism consider recent N items.')
        parser.add_argument('--Nq', type=int, default=1,
                            help='exit if Nq items share the same feature.')
        return BaseRunner.parse_runner_args(parser)
    
    def __init__(self, max_round, N, Nq, *args, **kwargs):
        self.max_round = max_round
        self.N = N
        self.Nq = Nq
        
        BaseRunner.__init__(self, *args, **kwargs)
        
    def train(self, model, trainset, validationset, testset):
        if self.optimizer is None:
            self._build_optimizer(model)
        self._check_time(start=True)
        
        init_valid = self.evaluate(model, validationset)
        # init_test = self.evaluate(model, testset)
#         pdb.set_trace()
        
        logging.info('Init: \t validation= %s [%.1f s]' % (init_valid[1], self._check_time()) + ','.join(self.metrics))
        
        for epoch in range(self.epoch):
            self._check_time()
            trainset.get_neg(validationset.pos_hist_dict)
            
            train_loss = self.fit(model, trainset)
            train_time = self._check_time()
            
            valid = self.evaluate(model, validationset)
            # test = self.evaluate(model, testset)
            test_time = self._check_time()
            
            self.train_results.append(train_loss)
            self.valid_results.append(valid[0][0])
            # self.test_results.append(test[0][0])
            self.valid_msg.append(valid[1])
            # self.test_msg.append(test[1])
            
            logging.info("Epoch %5d [%.1f s] \t train= %s validation= %s [%.1f s]" 
                         % (epoch + 1, train_time, str(train_loss), valid[1], test_time) + ','.join(self.metrics))
            if self.valid_results[-1] == max(self.valid_results):
                model.save_model()
            if self.eva_terminaion() and self.early_stop == 1:
                logging.info("Early stop at %d based on validation result" % (epoch + 1))
                break
            logging.info("")
                
        model.load_model()
        
    def evaluate(self, model, data):
        p, gt, rec_item = self.predict(model, data)
        return evaluator.evaluate(p, gt, self.metrics, model.item_num)
    
    def interaction(self, model, testset):
        user_list = None
        cum_sat = dict()
        length = dict()
        for i in range(self.max_round):
            testset.update_df(user_list)
            if testset.L == 0:
                break
            print(i)
            p, gt, rec_item = self.predict(model, testset)
            interacted_item, item_reward = evaluator.get_rec_reward(p, rec_item, gt)
            cum_sat, length = self.update_results(cum_sat, length, item_reward)
            user_list = self.whether_exit(testset.test_hist, interacted_item, testset.item_feat)
            testset.test_add_hist(interacted_item, item_reward)
            # pdb.set_trace()
        return cum_sat, length
        
    def whether_exit(self, user_hist, interacted_item, item_feat):
        user_list = []
        for user in interacted_item:
            target = interacted_item[user][0]
            last_rec = abs(user_hist[user][-1])
            feat_list = item_feat[target] + item_feat[last_rec]
            if len(set(feat_list)) == len(feat_list):
                user_list.append(user)
        return user_list
    
    def update_results(self, cum_sat, length, item_reward):
        for user in item_reward:
            if user not in cum_sat:
                cum_sat[user] = 0
                length[user] = 0
            cum_sat[user] += item_reward[user][0]
            length[user] += 1
        return cum_sat, length