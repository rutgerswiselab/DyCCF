import pandas as pd
import numpy as np
import csv
import os
import pickle
import argparse
import pdb

def multicore(in_file, out_file, user_threshold, item_threshold):
    csvFile = open(in_file,'r')
    reader = csv.reader(csvFile)
    data = []
    for i in reader:
        if i[2] == '':
            continue
        data.append(i)
    csvFile.close()
    

    user2count = {}
    item2count = {}
    for record in data:
        user = record[0]
        item = record[1]
        if user not in user2count:
            user2count[user] = 1
        else:
            user2count[user] += 1
        if item not in item2count:
            item2count[item] = 1
        else:
            item2count[item] += 1
    user_counter = 0
    item_counter = 0
    for user in user2count:
        if user2count[user] >= user_threshold:
            user_counter += 1
    for item in item2count:
        if item2count[item] >= item_threshold:
            item_counter += 1
    print('------------ report ------------')
    print('#total records: {}'.format(len(data)))
    print('#user exceed threshold: {}'.format(user_counter))
    print('#item exceed threshold: {}\n'.format(item_counter))
    old_data = data.copy()

    print('--------------------------------')
    change_num = 1  
    while change_num != 0:
        print(str(len(data)),end='\r')
        change_num = 0
        datanew = data.copy()
        data = []
        for record in datanew:
            sign_u = False
            sign_i = False
            user = record[0]
            item = record[1]
            if user2count[user] >= user_threshold:
                sign_u = True
            if item2count[item] >= item_threshold:
                sign_i = True
            if sign_u is False or sign_i is False:
    #         if user2count[user] < user_threshold or item2count[item] < item_threshold:
                user2count[user] -= 1
                item2count[item] -= 1
                change_num += 1
            else:
                data.append(record)
    #     if sign_u is True and sign_i is False:
    #         user2count[user] -= 1
    #     elif sign_u is False and sign_i is True:
    #         item2count[item] -= 1

    print('--------------------------------')
    user_counter = 0
    item_counter = 0
    uset=set()
    iset=set()
    for u in user2count:
        if user2count[u] >= user_threshold:
            user_counter += 1
            uset.add(u)
    for v in item2count:
        if item2count[v] >= item_threshold:
            item_counter += 1
            iset.add(v)
    print('Intersection: \n')
    print('#user in intersection: {}'.format(user_counter))
    print('#item in intersection: {}'.format(item_counter))
    print('uset: {}'.format(len(uset)))
    print('iset: {}'.format(len(iset)))

    result = []
    for record in old_data:
        sign_u = False
        sign_i = False
        user = record[0]
        item = record[1]
        if user2count[user] >= user_threshold:
            sign_u = True
        if item2count[item] >= item_threshold:
            sign_i = True
        if sign_u is True and sign_i is True:
            result.append(record)

    user2countF = {}
    item2countF = {}
    for record in result:
        user = record[0]
        item = record[1]
        if user not in user2countF:
            user2countF[user] = 1
        else:
            user2countF[user] += 1
        if item not in item2countF:
            item2countF[item] = 1
        else:
            item2countF[item] += 1
    user_counter = 0
    item_counter = 0
    uset2=set()
    iset2=set()
    for user in user2countF:
        if user2countF[user] >= user_threshold:
            user_counter += 1
            uset2.add(user)
    for item in item2countF:
        if item2countF[item] >= item_threshold:
            item_counter += 1
            iset2.add(item)
    print('------------ report ------------')
    print('#total result: {}'.format(len(result)))
    print('#users: {}'.format(len(user2countF)))
    print('#items: {}'.format(len(item2countF)))
    print('#user exceed threshold: {}'.format(user_counter))
    print('#item exceed threshold: {}\n'.format(item_counter))
    print('uset: {}'.format(len(uset2)))
    print('iset: {}'.format(len(iset2)))
    
    write = csv.writer(open(out_file, 'w'))
    write.writerows(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model', add_help=False)
    parser.add_argument('--dataset', type=str, default='CDs-3', help='the name of dataset')
    parser.add_argument('--file_name', type=str, default='ratings_CDs_and_Vinyl.csv', help='the name of the file')
    parser.add_argument('--u-thresh', type=int, default=5, help='threshold number of users')
    parser.add_argument('--i-thresh', type=int, default=5, help='threshold number of items')
    args = parser.parse_args()
    
    root = '../../data/'
    in_file = os.path.join(root, args.dataset, args.file_name)
    out_file = os.path.join(root, args.dataset, '{}_{}_{}.csv'.format(args.dataset, str(args.u_thresh), str(args.i_thresh)))
    multicore(in_file, out_file, args.u_thresh, args.i_thresh)