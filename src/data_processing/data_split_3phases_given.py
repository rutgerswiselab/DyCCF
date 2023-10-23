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
    parser.add_argument('--dataset', default='ml-100k-3', type=str,
                       help='the name of the dataset')
    parser.add_argument('--thresh', default=4, type=int,
                       help='interaction threshold to split valid and test.')
    return parser

def generate_index(args):
    rating_path = os.path.join(args.dataDir, args.dataset, 'all.csv')
    rating_df = pd.read_csv(rating_path, header=None)
    rating_df.columns = ['uid', 'iid', 'rating', 'timestamp']
    
    users = rating_df['uid'].unique()
    print('number of users: ' + str(len(users)))
    items = rating_df['iid'].unique()
    print('number of items: ' + str(len(items)))
    print('number of interactions: ' + str(len(rating_df)))
    
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
    with open(os.path.join(args.dataDir, args.dataset, 'user2id.pickle'), 'rb') as f:
        user2id = pickle.load(f)
        
        
    users = data_df['uid']
    items = data_df['iid']
    uid = [user2id[user] for user in users]
    iid = []
    for item in items:
        try:
            iid.append(item2id[item])
        except:
            pdb.set_trace()
    iid = [item2id[item] for item in items]
    
    data_df['uid'] = uid
    data_df['iid'] = iid
    data_df['rating'] = data_df['rating'].apply(lambda x: 1 if x > 3 else 0)
    data_df = data_df.sort_values(by=['uid', 'timestamp'], ignore_index=True)
    
    return data_df
def load_three_phases_df(args):
    phase1_path = os.path.join(args.dataDir, args.dataset, 'phase1.csv')
    phase2_path = os.path.join(args.dataDir, args.dataset, 'phase2.csv')
    phase3_path = os.path.join(args.dataDir, args.dataset, 'phase3.csv')
    
    phase1 = pd.read_csv(phase1_path, header=None)
    phase1.columns = ['uid', 'iid', 'rating', 'timestamp']
    phase1 = reindex_data_df(args, phase1)
    phase2 = pd.read_csv(phase2_path, header=None)
    phase2.columns = ['uid', 'iid', 'rating', 'timestamp']
    phase2 = reindex_data_df(args, phase2)
    phase3 = pd.read_csv(phase3_path, header=None)
    phase3.columns = ['uid', 'iid', 'rating', 'timestamp']
    phase3 = reindex_data_df(args, phase3)
    
    return phase1, phase2, phase3

def leave_one_out_by_time(data_df, thresh):
    split_index = []
    for uid, group in data_df.groupby('uid'):
        found, found_id = 0, -1
        if len(group.index) < thresh:
            continue
        for idx in reversed(range(len(group.index))):
            df_index = group.index[idx]
            if group.loc[df_index, 'rating'] > 0:
                found_id = idx
                found = 1
                break
        if found == 1:
            split_index.extend(group.index[found_id:])
    split_df = data_df.loc[split_index].reset_index(drop=True)
    remain_df = data_df.drop(split_index).reset_index(drop=True)
    
    return remain_df, split_df

def three_phases_split(phase1, phase2, phase3, thresh):
    phase1_remain, phase1_test_df = leave_one_out_by_time(phase1, thresh)
    phase1_train_df, phase1_val_df = leave_one_out_by_time(phase1_remain, thresh-1)
    phase2_remain, phase2_test_df = leave_one_out_by_time(phase2, thresh)
    phase2_train_df, phase2_val_df = leave_one_out_by_time(phase2_remain, thresh-1)
    phase3_remain, phase3_test_df = leave_one_out_by_time(phase3, thresh)
    phase3_train_df, phase3_val_df = leave_one_out_by_time(phase3_remain, thresh-1)
    
    return phase1_train_df, phase1_val_df, phase1_test_df, phase2_train_df, phase2_val_df, phase2_test_df, phase3_train_df, phase3_val_df, phase3_test_df

def print_info(phase1, phase2, phase3, phase1_train_df, phase1_val_df, phase1_test_df, phase2_train_df, phase2_val_df, phase2_test_df, phase3_train_df, phase3_val_df, phase3_test_df):
    trans_1 = len(phase1['uid'])
    trans_2 = len(phase2['uid'])
    trans_3 = len(phase3['uid'])
    u1 = len(phase1['uid'].unique())
    u2 = len(phase2['uid'].unique())
    u3 = len(phase3['uid'].unique())
    i1 = len(phase1['iid'].unique())
    i2 = len(phase2['iid'].unique())
    i3 = len(phase3['iid'].unique())
    
    train_trans_1 = len(phase1_train_df['uid'])
    train_trans_2 = len(phase2_train_df['uid'])
    train_trans_3 = len(phase3_train_df['uid'])
    train_u1 = len(phase1_train_df['uid'].unique())
    train_u2 = len(phase2_train_df['uid'].unique())
    train_u3 = len(phase3_train_df['uid'].unique())
    train_i1 = len(phase1_train_df['iid'].unique())
    train_i2 = len(phase2_train_df['iid'].unique())
    train_i3 = len(phase3_train_df['iid'].unique())
    
    val_trans_1 = len(phase1_val_df['uid'])
    val_trans_2 = len(phase2_val_df['uid'])
    val_trans_3 = len(phase3_val_df['uid'])
    val_u1 = len(phase1_val_df['uid'].unique())
    val_u2 = len(phase2_val_df['uid'].unique())
    val_u3 = len(phase3_val_df['uid'].unique())
    val_i1 = len(phase1_val_df['iid'].unique())
    val_i2 = len(phase2_val_df['iid'].unique())
    val_i3 = len(phase3_val_df['iid'].unique())
    
    test_trans_1 = len(phase1_test_df['uid'])
    test_trans_2 = len(phase2_test_df['uid'])
    test_trans_3 = len(phase3_test_df['uid'])
    test_u1 = len(phase1_test_df['uid'].unique())
    test_u2 = len(phase2_test_df['uid'].unique())
    test_u3 = len(phase3_test_df['uid'].unique())
    test_i1 = len(phase1_test_df['iid'].unique())
    test_i2 = len(phase2_test_df['iid'].unique())
    test_i3 = len(phase3_test_df['iid'].unique())
    
    print("phase 1:")
    print('phase 1 total: u {}, i {}, trans {}; train: u {}, i {}, trans {}; valid: u {}, i {}, trans {}; test:u {}, i {}, trans {}'.format(u1, i1, trans_1, train_u1, train_i1, train_trans_1, val_u1, val_i1, val_trans_1, test_u1, test_i1, test_trans_1))
    print("phase 2:")
    print('phase 2 total: u {}, i {}, trans {}; train: u {}, i {}, trans {}; valid: u {}, i {}, trans {}; test:u {}, i {}, trans {}'.format(u2, i2, trans_2, train_u2, train_i2, train_trans_2, val_u2, val_i2, val_trans_2, test_u2, test_i2, test_trans_2))
    print("phase 3:")
    print('phase 3 total: u {}, i {}, trans {}; train: u {}, i {}, trans {}; valid: u {}, i {}, trans {}; test:u {}, i {}, trans {}'.format(u3, i3, trans_3, train_u3, train_i3, train_trans_3, val_u3, val_i3, val_trans_3, test_u3, test_i3, test_trans_3))

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    phase1, phase2, phase3 = load_three_phases_df(args)
    phase1_train_df, phase1_val_df, phase1_test_df, phase2_train_df, phase2_val_df, phase2_test_df, phase3_train_df, phase3_val_df, phase3_test_df = three_phases_split(phase1, phase2, phase3, args.thresh)
    print_info(phase1, phase2, phase3, phase1_train_df, phase1_val_df, phase1_test_df, phase2_train_df, phase2_val_df, phase2_test_df, phase3_train_df, phase3_val_df, phase3_test_df)
    phase1_train_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase1_train.csv'), index=False, header=False)
    phase1_val_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase1_validation.csv'), index=False, header=False)
    phase1_test_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase1_test.csv'), index=False, header=False)
    phase2_train_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase2_train.csv'), index=False, header=False)
    phase2_val_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase2_validation.csv'), index=False, header=False)
    phase2_test_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase2_test.csv'), index=False, header=False)
    phase3_train_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase3_train.csv'), index=False, header=False)
    phase3_val_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase3_validation.csv'), index=False, header=False)
    phase3_test_df.to_csv(os.path.join(args.dataDir, args.dataset, 'phase3_test.csv'), index=False, header=False)
    
    
    