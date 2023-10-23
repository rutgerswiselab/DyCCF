from utils import utils
from utils import evaluator
from loaders.HistLoader import HistLoader
from loaders.DccfHistLoader import DccfHistLoader
from loaders.BaseLoader import BaseLoader
from loaders.KuaiRecLoader import KuaiRecLoader

from models.GRU4Rec import GRU4Rec
from models.DccfGRU4Rec import DccfGRU4Rec
from models.DccfNCR import DccfNCR
from models.IPSGRU4Rec import IPSGRU4Rec
from models.DICEGRU4Rec import DICEGRU4Rec
from models.MACRGRU4Rec import MACRGRU4Rec
from models.IPSSTAMP import IPSSTAMP
from models.DccfSTAMP import DccfSTAMP
from models.IPSNCR import IPSNCR
from models.STAMP import STAMP
from models.NCR import NCR
from models.MACRNCR import MACRNCR
from models.MACRSTAMP import MACRSTAMP
from runners.BaseRunner import BaseRunner
from runners.KuaiRecDICERunner import KuaiRecDICERunner
from runners.KuaiRecMACRRunner import KuaiRecMACRRunner
from runners.KuaiRecRunner import KuaiRecRunner
from runners.KuaiRecDccfRunner import KuaiRecDccfRunner

import argparse
import logging
import sys
import torch
import numpy as np
import os
import pdb

def main():
    init_parser = argparse.ArgumentParser(description='Indicate Model')
    init_parser.add_argument('--model', default='GRU4Rec', type=str,
                             help='the name of model')
    init_parser.add_argument('--runner', default='KuaiRecRunner', type=str,
                             help='the name of runner')
    init_parser.add_argument('--dataloader', default='KuaiRecLoader', type=str,
                             help='the name of dataloader')
    init_parser.add_argument('--dccf', default=0, type=int,
                             help='whether to use dccf framework')
    init_args, init_extra_args = init_parser.parse_known_args()
    
    
    model_name = eval(init_args.model)
    runner_name = eval(init_args.runner)
    loader_name = eval(init_args.dataloader)
    
    parser = utils.parse_global_args()
    if init_args.dccf == 1:
        model_name = eval('Dccf' + init_args.model)
        parser = model_name.parse_model_args(parser)
        runner_name = eval('KuaiRecDccfRunner')
        parser = runner_name.parse_runner_args(parser)
    elif init_args.dccf == 2:
        model_name = eval('MMR' + init_args.model)
        parser = model_name.parse_model_args(parser)
        runner_name = eval('KuaiRecMMRRunner')
        parser = runner_name.parse_runner_args(parser)
    elif init_args.dccf == 3:
        model_name = eval('IPS' + init_args.model)
        parser = model_name.parse_model_args(parser)
        parser = runner_name.parse_runner_args(parser)
# #         dccf_runner_name = eval('DccfRunner')
#         parser = dccf_runner_name.parse_runner_args(parser)
    elif init_args.dccf == 4:
        model_name = eval('MACR' + init_args.model)
        parser = model_name.parse_model_args(parser)
        runner_name = eval('KuaiRecMACRRunner')
        parser = runner_name.parse_runner_args(parser)
    elif init_args.dccf == 5:
        model_name = eval('DICE' + init_args.model)
        parser = model_name.parse_model_args(parser)
        runner_name = eval('KuaiRecDICERunner')
        parser = runner_name.parse_runner_args(parser)
    else:
        parser = model_name.parse_model_args(parser)
        parser = runner_name.parse_runner_args(parser)
    parser = loader_name.parse_loader_args(parser)
    
    
    args, extra_args = parser.parse_known_args()
    
    paras = sorted(vars(args).items(), key=lambda kv: kv[0])
    
    log_name_exclude = ['verbose', 'gpu', 'seed', 'dataset', 'path', 'phase1', 'dccf',
                        'model_path', 'log_file', 'metrics', 'load', 'train', 'eval_batch_size', 
                        'early_stop']
    
    log_file_name = [str(init_args.dccf), str(init_args.model), str(args.dataset), str(args.seed)] + \
                    [p[0].replace('_','')[:3] + str(p[1]) for p in paras if p[0] not in log_name_exclude]
    log_file_name = [l.replace(' ','-').replace('_', '-') for l in log_file_name]
    log_file_name = '_'.join(log_file_name)
    
    # phase1_model_exclude = ['verbose', 'gpu', 'seed', 'dataset', 'path', 'phase1', 'dccf', 'ctf_num', 'fact_prob',
    #                         'model_path', 'log_file', 'metrics', 'load', 'train', 'eval_batch_size', 
    #                         'early_stop']
    # phase1_model_file_name = [str(init_args.model), str(args.dataset), str(args.seed)] + \
    #                          [p[0].replace('_','')[:3] + str(p[1]) for p in paras if p[0] not in phase1_model_exclude]
    # phase1_model_file_name = [l.replace(' ','-').replace('_', '-') for l in phase1_model_file_name]
    # phase1_model_file_name = '_'.join(phase1_model_file_name)
    
    
    args.log_file = os.path.join('../log/', '%s/%s/%s.txt' % (init_args.model, args.dataset, log_file_name))
    utils.check_dir_and_mkdir(args.log_file)
    args.model_path = os.path.join('../model/', '%s/%s/%s_phase1.pt' % (init_args.model, args.dataset, log_file_name))
    utils.check_dir_and_mkdir(args.model_path)
    
    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info('=======================================')
    logging.info(vars(init_args))
    logging.info(vars(args))
    
    logging.info('DataLoader: ' + init_args.dataloader)
    logging.info('Model: ' + init_args.model)
    logging.info('Runner: ' + init_args.runner)

    
    
    # random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.info('# cuda devices: %d' % torch.cuda.device_count())
    
    dl_para_dict = utils.get_init_paras_dict(loader_name, vars(args))
    logging.info(loader_name.__name__ + ': ' + str(dl_para_dict))
    trainset = loader_name(**dl_para_dict)
    trainset.set_task('train')
    validationset = loader_name(**dl_para_dict)
    validationset.set_task('validation')
    testset = loader_name(**dl_para_dict)
    testset.set_task('test')

    if init_args.dataloader in ['KuaiRecLoader']:
        trainset.get_hist()
        validationset.get_hist(trainset.hist_dict, trainset.pos_hist_dict)
        # testset.get_hist(validationset.hist_dict, validationset.pos_hist_dict)
        trainset.get_neg(validationset.pos_hist_dict)
        validationset.get_neg(validationset.pos_hist_dict)
        testset.test_hist(validationset.hist_dict)
        testset.get_item_feat()
    logging.info('# users: ' + str(trainset.user_num))
    logging.info('# items: ' + str(trainset.item_num))

    # create model    
    loader_vars = vars(trainset)
    for key in loader_vars:
        if key not in args.__dict__:
            args.__dict__[key] = loader_vars[key]
    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    logging.info(model_name.__name__ + ': ' + str(model_para_dict))
    model = model_name(**model_para_dict)
    
    # create runner
    runner_para_dict = utils.get_init_paras_dict(runner_name, vars(args))
    logging.info(runner_name.__name__ + ': ' + str(runner_para_dict))
    runner = runner_name(**runner_para_dict)
    
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    if args.load > 0:
        model.load_model()
    if args.train > 0:
        runner.train(model, trainset, validationset, testset)
    # else:
    #     logging.info("Test Performance: %s" 
    #                      % (runner_phase1.evaluate(model, testset)[1]))
    model.load_model()
    
    cum_sat, length = runner.interaction(model, testset)
    avg_cs = sum(cum_sat.values()) / len(cum_sat.values())
    avg_len = sum(length.values()) / len(length.values())
    max_len = max(length.values())
    logging.info("average cumulative satisfaction is {}, average interaction length is {}, max interaction length is {}".format(str(avg_cs), str(avg_len),str(max_len)))
        
#     p, gt, rec_item = runner.predict(model_phase1, testset)
#     rec_list = evaluator.get_rec_list(p, rec_item, 10)
#     content_diversity_phase1 = evaluator.content_diversity(model_phase1, rec_list)
    
#     logging.info("content diversity at phase 1 is {}, at phase 2 is {}, the difference is {}".format(str(content_diversity_phase1), str(content_diversity_phase2), str(content_diversity_phase1 - content_diversity_phase2)))
    
    
    
if __name__ == '__main__':
    main()
    
    
