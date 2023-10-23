from loaders.HistLoader import HistLoader
import os
import pandas as pd
import numpy as np
import pickle
import pdb

class DyccfHistLoader(HistLoader):
    
    def get_neg(self, hist_dict):
        """
        generate negative samples
        """
        if self.task == 'train':
            self.data = self.data.sort_values(by=['time'], ignore_index=True)
            self._get_neg_train_df(hist_dict)
        else:
            self._get_neg_eval(hist_dict)