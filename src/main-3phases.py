from utils import utils
from utils import evaluator
from loaders.HistLoader import HistLoader
from loaders.DyccfHistLoader import DyccfHistLoader
from loaders.BaseLoader import BaseLoader
from models.GRU4Rec import GRU4Rec
from models.DyccfGRU4Rec import DyccfGRU4Rec
from models.STAMP import STAMP
from models.NCR import NCR
from models.DyccfNCR import DyccfNCR
from models.DyccfSTAMP import DyccfSTAMP
from runners.BaseRunner import BaseRunner
from runners.DyccfRunner import DyccfRunner
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
    init_parser.add_argument('--runner', default='BaseRunner', type=str,
                             help='the name of runner')
    init_parser.add_argument('--dataloader', default='BaseLoader', type=str,
                             help='the name of dataloader')
    init_parser.add_argument('--dccf', default=0, type=int,
                             help='whether to use dccf framework')
    init_args, init_extra_args = init_parser.parse_known_args()
    
    
    model_name = eval(init_args.model)
    init_args.runner = model_name.runner
    runner_name = eval(init_args.runner)
    init_args.dataloader = model_name.loader
    loader_name = eval(init_args.dataloader)
    
    parser = utils.parse_global_args()
    if init_args.dccf == 1:
        dccf_model_name = eval('Dccf' + init_args.model)
        parser = dccf_model_name.parse_model_args(parser)
        dccf_runner_name = eval('DccfRunner')
        parser = dccf_runner_name.parse_runner_args(parser)
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
    
    phase1_model_exclude = ['verbose', 'gpu', 'seed', 'dataset', 'path', 'phase1', 'dccf', 'ctf_num', 'fact_prob',
                            'model_path', 'log_file', 'metrics', 'load', 'train', 'eval_batch_size', 'batch_size',
                            'early_stop', 'sim_coef', 'chron', 'lambda_para']
    phase1_model_file_name = [str(init_args.model), str(args.dataset), str(args.seed)] + \
                             [p[0].replace('_','')[:3] + str(p[1]) for p in paras if p[0] not in phase1_model_exclude]
    phase1_model_file_name = [l.replace(' ','-').replace('_', '-') for l in phase1_model_file_name]
    phase1_model_file_name = '_'.join(phase1_model_file_name)
    
    
    args.log_file = os.path.join('../log/', '%s/%s/%s.txt' % (init_args.model, args.dataset, log_file_name))
    utils.check_dir_and_mkdir(args.log_file)
    args.model_path = os.path.join('../model/', '%s/%s/%s_phase1.pt' % (init_args.model, args.dataset, phase1_model_file_name))
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
    logging.info(init_args.dataloader + ': ' + str(dl_para_dict))
    trainset_phase1 = loader_name(**dl_para_dict)
    trainset_phase1.set_task('train', '1')
    validationset_phase1 = loader_name(**dl_para_dict)
    validationset_phase1.set_task('validation', '1')
    testset_phase1 = loader_name(**dl_para_dict)
    testset_phase1.set_task('test', '1')

    if init_args.dataloader in ['HistLoader']:
        trainset_phase1.get_hist()
        validationset_phase1.get_hist(trainset_phase1.hist_dict, trainset_phase1.pos_hist_dict)
        testset_phase1.get_hist(validationset_phase1.hist_dict, validationset_phase1.pos_hist_dict)
        trainset_phase1.get_neg(testset_phase1.pos_hist_dict)
        validationset_phase1.get_neg(testset_phase1.pos_hist_dict)
        testset_phase1.get_neg(testset_phase1.pos_hist_dict)
#         pdb.set_trace()
    logging.info('# users: ' + str(trainset_phase1.user_num))
    logging.info('# items: ' + str(trainset_phase1.item_num))

    # create model    
    loader_vars = vars(trainset_phase1)
    for key in loader_vars:
        if key not in args.__dict__:
            args.__dict__[key] = loader_vars[key]
    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    logging.info(init_args.model + ': ' + str(model_para_dict))
    model_phase1 = model_name(**model_para_dict)
    
    # create runner
    runner_para_dict = utils.get_init_paras_dict(runner_name, vars(args))
    logging.info(init_args.runner + ': ' + str(runner_para_dict))
    runner_phase1 = runner_name(**runner_para_dict)
    
    if args.phase1 == 0 and os.path.exists(args.model_path):
        model_phase1.load_model()

    else:
        if torch.cuda.device_count() > 0:
            model_phase1 = model_phase1.cuda()

        if args.load > 0:
            model_phase1.load_model()
        if args.train > 0:
            runner_phase1.train(model_phase1, trainset_phase1, validationset_phase1, testset_phase1)
        else:
            logging.info("Test Performance: %s" 
                         % (runner_phase1.evaluate(model_phase1, testset_phase1)[1]))
        model_phase1.load_model()
        
    p, gt, rec_item = runner_phase1.predict(model_phase1, testset_phase1)
    rec_list = evaluator.get_rec_list(p, rec_item, 10)
    content_diversity_phase1 = evaluator.content_diversity(model_phase1, rec_list)
    
    logging.info('start phase 2 training ...')
    
    if init_args.dccf == 1:
        init_args.model = 'Dccf' + init_args.model
        model_name = eval(init_args.model)
        init_args.runner = model_name.runner
        runner_name = eval(init_args.runner)
        init_args.dataloader = model_name.loader
        loader_name = eval(init_args.dataloader)
        loader_vars = vars(trainset_phase1)
        for key in loader_vars:
            if key not in args.__dict__:
                args.__dict__[key] = loader_vars[key]
        model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
        logging.info(init_args.model + ': ' + str(model_para_dict))
        runner_para_dict = utils.get_init_paras_dict(runner_name, vars(args))
        logging.info(init_args.runner + ': ' + str(runner_para_dict))

    dl_para_dict = utils.get_init_paras_dict(loader_name, vars(args))
    logging.info(init_args.dataloader + ': ' + str(dl_para_dict))
    trainset_phase2 = loader_name(**dl_para_dict)
    trainset_phase2.set_task('train', '2')
    validationset_phase2 = loader_name(**dl_para_dict)
    validationset_phase2.set_task('validation', '2')
    testset_phase2 = loader_name(**dl_para_dict)
    testset_phase2.set_task('test', '2')

    if init_args.dataloader in ['HistLoader', 'DyccfHistLoader']:
        trainset_phase2.get_hist(testset_phase1.hist_dict, testset_phase1.pos_hist_dict)
        validationset_phase2.get_hist(trainset_phase2.hist_dict, trainset_phase2.pos_hist_dict)
        testset_phase2.get_hist(validationset_phase2.hist_dict, validationset_phase2.pos_hist_dict)
        trainset_phase2.get_neg(testset_phase2.pos_hist_dict)
        validationset_phase2.get_neg(testset_phase2.pos_hist_dict)
        testset_phase2.get_neg(testset_phase2.pos_hist_dict)
    
    args.model_path = os.path.join('../model/', '%s/%s/%s_phase2.pt' % (init_args.model, args.dataset, log_file_name))
    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    model_phase2 = model_name(**model_para_dict)
    model_phase2.copy_params(model_phase1)
    
    runner_phase2 = runner_name(**runner_para_dict)
    
    if torch.cuda.device_count() > 0:
        model_phase2 = model_phase2.cuda()

    if args.load > 0:
        model_phase2.load_model()
    if args.train > 0:
        runner_phase2.train(model_phase2, trainset_phase2, validationset_phase2, testset_phase2)
    else:
        logging.info("Test Performance: %s" 
                     % (runner_phase2.evaluate(model_phase2, testset_phase2)[1]))
    model_phase2.load_model()
    
    p, gt, rec_item = runner_phase2.predict(model_phase2, testset_phase2)
    if init_args.dccf == 2:
        similarity = np.array(torch.matmul(model_phase2.iid_embeddings.weight.detach(), model_phase2.iid_embeddings.weight.detach().t()).cpu())
        _, rec_list = evaluator.mmr_sort(similarity, p, gt, rec_item, runner_phase2.lambda_para, 10)
    else:
        rec_list = evaluator.get_rec_list(p, rec_item, 10)
    content_diversity_phase2 = evaluator.content_diversity(model_phase2, rec_list)
    
    logging.info('start phase 3 training ...')

    dl_para_dict = utils.get_init_paras_dict(loader_name, vars(args))
    logging.info(init_args.dataloader + ': ' + str(dl_para_dict))
    trainset_phase3 = loader_name(**dl_para_dict)
    trainset_phase3.set_task('train', '3')
    validationset_phase3 = loader_name(**dl_para_dict)
    validationset_phase3.set_task('validation', '3')
    testset_phase3 = loader_name(**dl_para_dict)
    testset_phase3.set_task('test', '3')

    if init_args.dataloader in ['HistLoader', 'DyccfHistLoader']:
        trainset_phase3.get_hist(testset_phase2.hist_dict, testset_phase2.pos_hist_dict)
        validationset_phase3.get_hist(trainset_phase3.hist_dict, trainset_phase3.pos_hist_dict)
        testset_phase3.get_hist(validationset_phase3.hist_dict, validationset_phase3.pos_hist_dict)
        trainset_phase3.get_neg(testset_phase3.pos_hist_dict)
        validationset_phase3.get_neg(testset_phase3.pos_hist_dict)
        testset_phase3.get_neg(testset_phase3.pos_hist_dict)
    
    args.model_path = os.path.join('../model/', '%s/%s/%s_phase3.pt' % (init_args.model, args.dataset, log_file_name))
    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    model_phase3 = model_name(**model_para_dict)
    model_phase3.copy_params(model_phase2)
    
    runner_phase3 = runner_name(**runner_para_dict)
    
    if torch.cuda.device_count() > 0:
        model_phase3 = model_phase3.cuda()

    if args.load > 0:
        model_phase3.load_model()
    if args.train > 0:
        runner_phase3.train(model_phase3, trainset_phase3, validationset_phase3, testset_phase3)
    else:
        logging.info("Test Performance: %s" 
                     % (runner_phase3.evaluate(model_phase3, testset_phase3)[1]))
    model_phase3.load_model()
    
    p, gt, rec_item = runner_phase3.predict(model_phase3, testset_phase3)
    
    if init_args.dccf == 2:
        similarity = np.array(torch.matmul(model_phase3.iid_embeddings.weight.detach(), model_phase3.iid_embeddings.weight.detach().t()).cpu())
        _, rec_list = evaluator.mmr_sort(similarity, p, gt, rec_item, runner_phase3.lambda_para, 10)
    else:
        rec_list = evaluator.get_rec_list(p, rec_item, 10)
    content_diversity_phase3 = evaluator.content_diversity(model_phase3, rec_list)
    
    diff12 = content_diversity_phase1 - content_diversity_phase2
    diff13 = content_diversity_phase1 - content_diversity_phase3
    diff23 = content_diversity_phase2 - content_diversity_phase3
    
    logging.info("content diversity at phase 1 is {}, at phase 2 is {}, at phase 3 is {}, 1-2 diff is {}, 1-3 diff is {}, 2-3 diff is {}.".format(
        str(content_diversity_phase1), str(content_diversity_phase2), str(content_diversity_phase3), str(diff12), str(diff13), str(diff23)))
    
    
    
if __name__ == '__main__':
    main()
    
    
