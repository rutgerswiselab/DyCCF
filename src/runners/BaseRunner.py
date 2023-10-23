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

class BaseRunner(object):
    def parse_runner_args(parser):
        parser.add_argument('--load', default=0, type=int,
                           help='Whether load model and continue to train')
        parser.add_argument('--opt', default='Adam', type=str,
                           help='Select model optimizer')
        parser.add_argument('--lr', default=0.001, type=float,
                           help='learning rate of optimizer')
        parser.add_argument('--l2', default=1e-6, type=float,
                           help='weight of l2-regularizer')
        parser.add_argument('--batch_size', default=512, type=int,
                           help='batch size')
        parser.add_argument('--eval_batch_size', default=256 * 256, type=int,
                           help='batch size')
        parser.add_argument('--epoch', default=100, type=int,
                           help='the max number of epoches')
        parser.add_argument('--metrics', default='nDCG@10,hit@10', type=str,
                           help='evaluation metrics')
        parser.add_argument('--early_stop', default=1, type=int,
                           help='early stop or not')
        
        return parser
    
    def __init__(self, opt, lr, l2, epoch, batch_size, eval_batch_size, metrics, early_stop):
        self.opt_name = opt.lower()
        self.lr = lr
        self.l2 = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.optimizer = None
        self.time = None
        self.train_results, self.valid_results, self.test_results = [], [], []
        self.valid_msg, self.test_msg = [], []
        self.metrics = metrics.lower().split(',')
        self.early_stop = early_stop
        
        
    def _build_optimizer(self, model):
        
        if self.opt_name == 'sgd':
            self.optimizer = opt.SGD(model.parameters(), lr=self.lr)
        elif self.opt_name == 'adagrad':
            self.optimizer = opt.Adagrad(model.parameters(), lr=self.lr)
        elif self.opt_name == 'adam':
            self.optimizer = opt.Adam(model.parameters(), lr=self.lr)
        else:
            print("Unknown Optimizer: " + self.opt_name)
            self.optimizer = opt.SGD(model.parameters(), lr=self.lr)
            
        return
    
    def _check_time(self, start=False):
        """
        Used for timing, self.time = [starting time, last step time]
        @input:
        - start: is it for starting
        @output:
        - return: if it is a start, return current time, otherwise, return the time span from the last step
        """
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time
    
    def eva_terminaion(self):
        """
        whether stop the training, based on the validation evaluation
        @output:
        - return: stop the training or not, True or False 
        """
        valid = self.valid_results
        
        if len(valid) > 20 and utils.strictly_decreasing(valid[-5:]):
            return True
        elif len(valid) - valid.index(max(valid)) > 20:
            return True
        return False
    
    def predict(self, model, data):
        """
        Get prediction for evaluation, not training
        @input:
        - model: the model
        - data: validation data or test data
        @output:
        - p: a prediction dict, {user: [predictions]}
        - gt: a ground truth dict, {user: [ground truth]}
        """
        dataLoader = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=16)
        
        model.eval()
        p = defaultdict(list)
        gt = defaultdict(list)
        rec_item = defaultdict(list)
        pbar = tqdm(total=len(data), leave=False, ncols=100, mininterval=1, desc='Predict')
        for i, batchData in enumerate(dataLoader):
            pbar.update(batchData['uid'].shape[0])
            out_dict = model.predict(batchData)
#             pdb.set_trace()
            prediction = out_dict['prediction'].detach().cpu().numpy()
            label = out_dict['label'].detach().cpu().numpy()
            user = out_dict['uid'].detach().cpu().numpy()
            item = out_dict['iid'].detach().cpu().numpy()
            for key, rec_id, pred, lab in zip(user, item, prediction, label):
                p[key].append(pred)
                gt[key].append(lab)
                rec_item[key].append(rec_id)
        pbar.close()
#         pdb.set_trace()
        return p, gt, rec_item
    
    def fit(self, model, data):
        losses = []
        dataLoader = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=16)
        model.train()
        pbar = tqdm(total=len(data), leave=False, ncols=100, mininterval=1, desc='Predict')
        for i, batchData in enumerate(dataLoader):
            pbar.update(batchData['uid'].shape[0])
            self.optimizer.zero_grad()
            out_dict = model(batchData)
            loss = out_dict['loss'] + model.l2() * self.l2
            losses.append(loss.detach().cpu())
            loss.backward()
            self.optimizer.step()
        pbar.close()
#         pdb.set_trace()
        model.eval()
        return np.mean(losses)
        
    def train(self, model, trainset, validationset, testset):
        if self.optimizer is None:
            self._build_optimizer(model)
        self._check_time(start=True)
        
        init_valid = self.evaluate(model, validationset)
        init_test = self.evaluate(model, testset)
#         pdb.set_trace()
        
        logging.info('Init: \t validation= %s test= %s [%.1f s]' % (init_valid[1], init_test[1], self._check_time()) + ','.join(self.metrics))
        
        for epoch in range(self.epoch):
            self._check_time()
            trainset.get_neg(testset.pos_hist_dict)
            
            train_loss = self.fit(model, trainset)
            train_time = self._check_time()
            
            valid = self.evaluate(model, validationset)
            test = self.evaluate(model, testset)
            test_time = self._check_time()
            
            self.train_results.append(train_loss)
            self.valid_results.append(valid[0][0])
            self.test_results.append(test[0][0])
            self.valid_msg.append(valid[1])
            self.test_msg.append(test[1])
            
            logging.info("Epoch %5d [%.1f s] \t train= %s validation= %s test= %s [%.1f s]" 
                         % (epoch + 1, train_time, str(train_loss), valid[1], test[1], test_time) + ','.join(self.metrics))
            if self.valid_results[-1] == max(self.valid_results):
                model.save_model()
            if self.eva_terminaion() and self.early_stop == 1:
                logging.info("Early stop at %d based on validation result" % (epoch + 1))
                break
            logging.info("")
                
        best_valid_eval = max(self.valid_results)
        best_valid_epoch = self.valid_results.index(best_valid_eval)
        logging.info("Best Iteration (validation)= %5d \t validation= %s test= %s" 
                     % (best_valid_epoch + 1, self.valid_msg[best_valid_epoch], self.test_msg[best_valid_epoch]))
        
        best_test_eval = max(self.test_results)
        best_test_epoch = self.test_results.index(best_test_eval)
        logging.info("Best Iteration (test)= %5d \t validation= %s test= %s" 
                     % (best_test_epoch + 1, self.valid_msg[best_test_epoch], self.test_msg[best_test_epoch]))
        model.load_model()
        
    def evaluate(self, model, data):
        p, gt, rec_item = self.predict(model, data)
        return evaluator.evaluate(p, gt, self.metrics, model.item_num)
    
    