from loaders.BaseLoader import BaseLoader
import os
import pandas as pd
import numpy as np
import pickle
import logging
import pdb

class KuaiRecLoader(BaseLoader):
    def parse_loader_args(parser):
        parser.add_argument('--max_hist', default=10, type=int,
                           help='the max number of items in history')
        return BaseLoader.parse_loader_args(parser)
    
    def __init__(self, max_hist, *args, **kwargs):
        BaseLoader.__init__(self, *args, **kwargs)
        self.item_num += 1 # use 0 as dummy history
        self.max_hist = max_hist
        
    def set_task(self, task):
        """
        train, validation or test
        """
        self.task = task
        cols = ['uid', 'iid', 'rating', 'time']
        if self.task == 'train':
            data_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'), names=cols)
        elif self.task == 'validation':
            data_df = pd.read_csv(os.path.join(self.data_path, 'validation.csv'), names=cols)
        elif self.task == 'test':
            data_df = pd.read_csv(os.path.join(self.data_path, 'test.csv'), names=cols)
        else:
            logging.info("Unknow task:" + task)
            raise Exception("Unknow task:" + task)
        # if self.task == 'test':
        #     self.data = data_df[['uid', 'iid', 'rating', 'time']].copy()
        self.data = data_df[['uid', 'iid', 'rating', 'time']].copy()
        logging.info('Getting ' + self.task + ' data...')
        return
        
    def get_hist(self, hist_dict=None, pos_hist_dict=None):
        """
        get history sequence for data with leave-one-out splitting.
        """
        
        if 'history' in self.data:
            return
        self.data['iid'] += 1
        history = []
        uids = self.data['uid'].tolist()
        iids = self.data['iid'].tolist()
        ratings = self.data['rating'].tolist()
        if hist_dict is None:
            if self.task != 'train':
                logging.info("No user history for validation/test set")
                raise Exception("No user history for validation/test set")
            hist_dict = dict()
        if pos_hist_dict is None:
            pos_hist_dict = dict()
        
        for i, uid in enumerate(uids):
            iid = iids[i]
            rating = ratings[i]
            if uid not in hist_dict:
                hist_dict[uid] = []
                pos_hist_dict[uid] = []
            hist = hist_dict[uid]
            if len(hist) < self.max_hist:
                hist += [0] * (self.max_hist - len(hist))
            history.append(str(hist[-self.max_hist:]).replace(' ', '')[1:-1])
            if rating > 0:
                hist_dict[uid].append(iid)
                pos_hist_dict[uid].append(iid)
            else:
                hist_dict[uid].append(-iid)
            
        self.data['history'] = history
        empty_his = ','.join(['0']*self.max_hist)
        self.data = self.data[self.data.history != empty_his].reset_index(drop=True)
        self.data = self.data[self.data.rating > 0].reset_index(drop=True)
        self.hist_dict = hist_dict
        self.pos_hist_dict = pos_hist_dict
        
        return
    
    def get_neg(self, pos_hist_dict):
        """
        generate negative samples
        """
        if self.task == 'train':
            self.data = self.data.sort_values(by=['time'], ignore_index=True)
            self._get_neg_train_df(pos_hist_dict)
        else:
            self._get_neg_eval(pos_hist_dict)
            
    def _get_neg_train_df(self, pos_hist_dict):
        uids = self.data['uid'].tolist()
        iids = self.data['iid'].tolist()
        negs = []
        for i, uid in enumerate(uids):
            while True:
                neg_id = np.random.randint(self.item_num)
                if neg_id not in pos_hist_dict[uid]:
                    negs.append(neg_id)
                    break
        self.data['negative'] = negs
        self.L = len(self.data)
        return
    
    def _get_neg_eval(self, pos_hist_dict):
        if self.task == 'validation':
            neg_num = self.val_neg
        else:
            neg_num = self.test_neg
        
        neg_sample = dict()
        neg_sample['uid'] = []
        neg_sample['iid'] = []
        neg_sample['rating'] = []
        neg_sample['time'] = []
        neg_sample['history'] = []
        
        uids = self.data['uid'].tolist()
        iids = self.data['iid'].tolist()
        times = self.data['time'].tolist()
        histories = self.data['history'].tolist()
        for i, uid in enumerate(uids):
            iid = iids[i]
            time = times[i]
            history = histories[i]
            if neg_num > 0:
                neg_list = []
                while len(neg_list) < neg_num:
                    neg_id = np.random.randint(1, self.item_num)
                    if neg_id not in neg_list and neg_id not in pos_hist_dict[uid]:
                            neg_list.append(neg_id)
            else:
                neg_list = [i for i in range(1, self.item_num) if i not in hist_dict[uid]]
            neg_sample['uid'].extend([uid] * len(neg_list))
            neg_sample['iid'].extend(neg_list)
            neg_sample['time'].extend([time] * len(neg_list))
            neg_sample['rating'].extend([0] * len(neg_list))
            neg_sample['history'].extend([history] * len(neg_list))
        self.data = self.data.append(pd.DataFrame(neg_sample))
        self.L = len(self.data)
        self.data = self.data.to_dict(orient='list')
        self.data['history'] = [[int(i) for i in his.split(',')] for his in self.data['history']]
#         pdb.set_trace()
        
        return
    
    def __len__(self):
        if self.task is not None:
            return self.L
        raise Exception("Set up task first")
        
    def __getitem__(self, idx):
        if self.task == 'train':
            return self._getitem_train(idx)
        elif self.task == 'validation':
            return self._getitem_eval(idx)
        return self._getitem_test(idx)
    
    def _getitem_train(self, idx):
        row = self.data.iloc[idx]
        return {'uid': np.array(row['uid']).astype(np.int64),
                'iid': np.array(row['iid']).astype(np.int64),
                'negative': np.array(row['negative']).astype(np.int64),
                'history': np.array([int(i) for i in row['history'].split(',')]).astype(np.int64)}
    
    def _getitem_eval(self, idx):
        uid = self.data['uid'][idx]
        iid = self.data['iid'][idx]
        his = self.data['history'][idx]
        rating = self.data['rating'][idx]
        return {'uid': np.array(uid).astype(np.int64),
                'iid': np.array(iid).astype(np.int64),
                'rating': np.array(rating).astype(np.int64),
                'history': np.array(his).astype(np.int64)}
            
            
    def _getitem_test(self, idx):
        uid = self.data_list['uid'][idx]
        iid = self.data_list['iid'][idx]
        his = self.test_hist[uid]
        rating = self.data_list['rating'][idx]
        return {'uid': np.array(uid).astype(np.int64),
                'iid': np.array(iid).astype(np.int64),
                'rating': np.array(rating).astype(np.int64),
                'history': np.array(his).astype(np.int64)}
    
    def test_hist(self, user_hist):
        self.test_hist = user_hist
        for user in self.test_hist:
            self.test_hist[user] = ([0] * self.max_hist + self.test_hist[user])[-self.max_hist:]
        
    def test_add_hist(self, interacted_item, item_reward):
        for user in interacted_item:
            if item_reward[user][0] >= 2.0:
                self.test_hist[user] = np.concatenate((self.test_hist[user][1:], interacted_item[user]))
            else:
                self.test_hist[user] = np.concatenate((self.test_hist[user][1:], -interacted_item[user]))
            
    def update_df(self, user_list=None):
        if user_list is not None:
            self.data = self.data[self.data['uid'].isin(user_list)]
        self.data_list = self.data.to_dict(orient='list')
        self.L = len(self.data_list['uid'])
    
    def get_item_feat(self):
        item_feat = pd.read_csv(os.path.join(self.data_path, 'item_feat.csv'), names=['iid','feat'])
        item_feat["feat"] = item_feat["feat"].map(eval)
        self.item_feat = dict(zip(item_feat.iid, item_feat.feat))
        
            
    
            