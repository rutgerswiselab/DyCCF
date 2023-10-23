import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import logging
import pickle
import pdb

class BaseLoader(Dataset):
    def parse_loader_args(parser):
        parser.add_argument('--dataset', default='ml100k', type=str,
                           help='the name of the dataset')
        parser.add_argument('--path', default='../data', type=str,
                           help='data path')
        parser.add_argument('--val_neg', default=100, type=int,
                           help='number of negative samples during validation evaluation, use all items if -1')
        parser.add_argument('--test_neg', default=100, type=int,
                           help='number of negative samples during testing evaluation, use all items if -1')
        return parser
    
    def __init__(self, path, dataset, val_neg, test_neg):
        super(BaseLoader, self).__init__()
        self.data_path = os.path.join(path, dataset)
        self.dataset = dataset
        self.data = None
        self.task = None
        self.data_info()
        self.val_neg = val_neg
        self.test_neg = test_neg
        self.L = None
        
        return
    
    def set_task(self, task):
        """
        train, validation or test
        """
        self.task = task
        cols = ['uid', 'iid', 'pid', 'rating', 'time']
        if self.task == 'train':
            data_df = pd.read_csv(os.path.join(self.data_path, self.dataset + '.train.csv'), names=cols)
        elif self.task == 'validation':
            data_df = pd.read_csv(os.path.join(self.data_path, self.dataset + '.validation.csv'), names=cols)
        elif self.task == 'test':
            data_df = pd.read_csv(os.path.join(self.data_path, self.dataset + '.test.csv'), names=cols)
        else:
            logging.info("Unknow task:" + task)
            raise Exception("Unknow task:" + task)
        if self.pt > 0:
            self.data = data_df[['uid', 'pid', 'rating', 'time']].copy()
            self.data.rename(columns={'pid':'iid'}, inplace=True)
        else:
            self.data = data_df[['uid', 'iid', 'rating', 'time']].copy()
        self.data['rating'] = [1] * len(self.data)
        logging.info('Getting ' + self.task + ' data...')
        return
    
    def data_info(self):
        """
        load data statistic info
        """
        with open(os.path.join(self.data_path, 'user2id.pickle'), 'rb') as f:
            user2id = pickle.load(f)
        self.user_num = len(user2id)
        with open(os.path.join(self.data_path, 'item2id.pickle'), 'rb') as f:
            item2id = pickle.load(f)
        self.item_num = len(item2id)
        return
    