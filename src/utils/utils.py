import os
import argparse
import logging
import inspect

import pdb

def parse_global_args():
    parser = argparse.ArgumentParser(description='Causal Discovery with Unknown Intervention')
    parser.add_argument('--seed', default=2021, type=int, 
                       help='Random seed of numpy and torch.')
    parser.add_argument('--verbose', default=logging.INFO, type=str, 
                       help='Logging Level')
    parser.add_argument('--gpu', default='0', type=str,
                       help='set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--log_file', default='../../log/log.txt', type=str,
                       help='Logging file path')
    parser.add_argument('--train', default=1, type=int,
                       help='whether to train the model')
    parser.add_argument('--phase1', default=1, type=int,
                       help='whether to train base model on phase1 data')
    
    
    return parser


def get_init_paras_dict(class_name, paras_dict):
    base_list = inspect.getmro(class_name)
    paras_list = []
    for base in base_list:
        paras = inspect.getfullargspec(base.__init__)
        paras_list.extend(paras.args)
    paras_list = sorted(list(set(paras_list)))
    out_dict = {}
    for para in paras_list:
        if para == 'self':
            continue
        out_dict[para] = paras_dict[para]
    return out_dict


def check_dir_and_mkdir(path):
    if os.path.basename(path).find('.') == -1 or path.endswith('/'):
        dirname = path
    else:
        dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        print('make dirs:', dirname)
        os.makedirs(dirname)
    return

def strictly_decreasing(l):
    return all(x > y for x, y in zip(l, l[1:]))