import numpy as np
import pickle
import os
import pandas as pd
import argparse
import pdb

def init_parser():
    parser = argparse.ArgumentParser(description='movielens data split')
    parser.add_argument('--dataDir', default='../../data', type=str,
                       help='data directory')
    parser.add_argument('--dataset', default='ml100k-3', type=str,
                       help='the name of the dataset')
    parser.add_argument('--phase1_ratio', default=0.2, type=float,
                       help='the ratio of phase 1 data')
    parser.add_argument('--phase2_ratio', default=0.4, type=float,
                       help='the ratio of phase 1 data')
    return parser

def generate_index(args):
    rating_path = os.path.join(args.dataDir, args.dataset, 'ratings.csv')
    rating_df = pd.read_csv(data_file)
    
    users = data_df['userId'].unique()
    print('number of users: ' + str(len(users)))
    items = data_df['movieId'].unique()
    print('number of items: ' + str(len(items)))
    print('number of interactions: ' + str(len(data_df)))
    
    user2id = {users[i]: i for i in range(len(users))}
    id2user = {i: users[i] for i in range(len(users))}
    item2id = {items[i]: i for i in range(len(items))}
    id2item = {i: items[i] for i in range(len(items))} 
    
    with open(os.path.join(args.dataDir, args.dataset, 'item2id.pickle'), 'wb') as f:
        pickle.dump(item2id, f)
    with open(os.path.join(args.dataDir, args.dataset, 'id2item.pickle'), 'wb') as f:
        pickle.dump(id2item, f)
    with open(os.path.join(args.dataDir, args.dataset, 'user2id.pickle'), 'wb') as f:
        pickle.dump(user2id, f)
    with open(os.path.join(args.dataDir, args.dataset, 'id2user.pickle'), 'wb') as f:
        pickle.dump(id2user, f)
        
    
def reindex_data_df(args, data_df):
    if not os.path.exists(os.path.join(args.dataDir, args.dataset, 'item2id.pickle')):
        generate_index(args)
        
    with open(os.path.join(args.dataDir, args.dataset, 'item2id.pickle'), 'rb') as f:
        item2id = pickle.load(f)
    with open(os.path.join(args.dataDir, args.dataset, 'id2item.pickle'), 'rb') as f:
        id2item = pickle.load(f)
    with open(os.path.join(args.dataDir, args.dataset, 'user2id.pickle'), 'rb') as f:
        user2id = pickle.load(f)
    with open(os.path.join(args.dataDir, args.dataset, 'id2user.pickle'), 'rb') as f:
        id2user = pickle.load(f)
        
        
    users = data_df['userId']
    items = data_df['movieId']
    uid = [user2id[user] for user in users]
    iid = [item2id[item] for item in items]
    
    data_df['uid'] = uid
    data_df['iid'] = iid
    data_df = data_df[['uid', 'iid', 'rating', 'timestamp']]
    data_df = data_df.sort_values(by=['uid', 'timestamp'], ignore_index=True)
    
    return data_df

def leave_one_out_by_time(data_df):
    split_index = []
    for uid, group in data_df.groupby('uid'):
        split_index.append(group.index[-1])
    split_df = data_df.loc[split_index].reset_index(drop=True)
    remain_df = data_df.drop(split_index).reset_index(drop=True)
    
    return remain_df, split_df

def three_phase_train(args, train_df):
    phase1_train_index = []
    phase1_val_index = []
    phase1_test_index = []
    phase2_train_index = []
    phase2_val_index = []
    phase2_test_index = []
    phase3_train_index = []
    
    for uid, group in train_df.groupby('uid'):
        idx1 = int(len(group) * args.phase1_ratio)
        idx2 = int(len(group) * args.phase2_ratio) + idx1
        phase1_train_index.extend(group.index[:idx1])
        phase1_val_index.append(group.index[idx1])
        phase1_test_index.append(group.index[idx1+1])
        phase2_train_index.extend(group.index[idx1:idx2])
        phase2_val_index.append(group.index[idx2])
        phase2_test_index.append(group.index[idx2+1])
        phase3_train_index.extend(group.index[idx2:])
    phase1_train_df = train_df.loc[phase1_train_index].reset_index(drop=True)
    phase1_val_df = train_df.loc[phase1_val_index].reset_index(drop=True)
    phase1_test_df = train_df.loc[phase1_test_index].reset_index(drop=True)
    phase2_train_df = train_df.loc[phase2_train_index].reset_index(drop=True)
    phase2_val_df = train_df.loc[phase2_val_index].reset_index(drop=True)
    phase2_test_df = train_df.loc[phase2_test_index].reset_index(drop=True)
    phase3_train_df = train_df.loc[phase3_train_index].reset_index(drop=True)
    
    return phase1_train_df, phase1_val_df, phase1_test_df, phase2_train_df, phase2_val_df, phase2_test_df, phase3_train_df

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    data_file = os.path.join(args.dataDir, args.dataset, 'ratings.csv')
    data_df = pd.read_csv(data_file)
    data_df = reindex_data_df(args, data_df)
    remaining_df, test_df = leave_one_out_by_time(data_df)
    train_df, validation_df = leave_one_out_by_time(remaining_df)
    train_df.to_csv(os.path.join(args.dataDir, args.dataset, 'train.csv'), index=False, header=False)
    validation_df.to_csv(os.path.join(args.dataDir, args.dataset, 'validation.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(args.dataDir, args.dataset, 'test.csv'), index=False, header=False)
    phase1_train_df, phase1_val_df, phase1_test_df, phase2_train_df, phase2_val_df, phase2_test_df, phase3_train_df = three_phase_train(args, train_df)
    phase1_train_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase1_train.csv'), index=False, header=False)
    phase1_val_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase1_validation.csv'), index=False, header=False)
    phase1_test_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase1_test.csv'), index=False, header=False)
    phase2_train_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase2_train.csv'), index=False, header=False)
    phase2_val_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase2_validation.csv'), index=False, header=False)
    phase2_test_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase2_test.csv'), index=False, header=False)
    phase3_train_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase3_train.csv'), index=False, header=False)
    
    
    