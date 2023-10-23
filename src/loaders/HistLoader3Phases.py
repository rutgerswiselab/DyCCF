from loaders.HistLoader import HistLoader
import os
import pandas as pd
import numpy as np
import pickle
import logging
import pdb

class HistLoader3Phases(HistLoader):
    def set_task(self, task, phase):
        """
        train, validation or test
        """
        self.task = task
        cols = ['uid', 'iid', 'rating', 'time']
        if phase == '1':
            if self.task == 'train':
                data_df = pd.read_csv(os.path.join(self.data_path, 'phase1_train.csv'), names=cols)
            elif self.task == 'validation':
                data_df = pd.read_csv(os.path.join(self.data_path, 'phase1_validation.csv'), names=cols)
            elif self.task == 'test':
                data_df = pd.read_csv(os.path.join(self.data_path, 'phase1_test.csv'), names=cols)
            else:
                logging.info("Unknow task:" + task)
                raise Exception("Unknow task:" + task)
        elif phase == '2':
            if self.task == 'train':
                data_df = pd.read_csv(os.path.join(self.data_path, 'phase2_train.csv'), names=cols)
            elif self.task == 'validation':
                data_df = pd.read_csv(os.path.join(self.data_path, 'phase2_validation.csv'), names=cols)
            elif self.task == 'test':
                data_df = pd.read_csv(os.path.join(self.data_path, 'phase2_test.csv'), names=cols)
            else:
                logging.info("Unknow task:" + task)
                raise Exception("Unknow task:" + task)
        else:
            if self.task == 'train':
                data_df = pd.read_csv(os.path.join(self.data_path, 'phase3_train.csv'), names=cols)
            elif self.task == 'validation':
                data_df = pd.read_csv(os.path.join(self.data_path, 'validation.csv'), names=cols)
            elif self.task == 'test':
                data_df = pd.read_csv(os.path.join(self.data_path, 'test.csv'), names=cols)
            else:
                logging.info("Unknow task:" + task)
                raise Exception("Unknow task:" + task)
        self.data = data_df[['uid', 'iid', 'rating', 'time']].copy()
        self.data['rating'] = [1] * len(self.data)
        logging.info('Getting ' + self.task + ' data...')
        return
        
            
            
        