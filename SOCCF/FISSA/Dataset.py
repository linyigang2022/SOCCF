import pandas as pd
import numpy as np
import random
import math
import multiprocessing
import time

class Dataset(object):
    def __init__(self, file_prefix, is_valid=1):
        self.train = pd.read_csv(file_prefix + "_train.csv", header=None,
                                 names=['user_id', 'item_id', 'timestamp'],
                                 dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32})
        self.valid = pd.read_csv(file_prefix + "_valid.csv", header=None,
                                 names=['user_id', 'item_id', 'timestamp'],
                                 dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32})

        if is_valid == 0:
            self.train = pd.concat([self.train, self.valid], axis='index')
            self.valid = pd.read_csv(file_prefix + "_test.csv", header=None,
                                     names=['user_id', 'item_id', 'timestamp'],
                                     dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32})

        self.candidate = pd.read_csv(file_prefix + "_negative.csv", header=None)

    def fix_length(self, maxlen=50):
        self.maxlen = maxlen

        self.train.sort_values(by=['user_id','timestamp'], axis='index', ascending=True, inplace=True, kind='mergesort')
        self.train = self.train.groupby(['user_id']).tail(maxlen + 1)

        self.user_set = set(self.train.user_id.unique())
        self.item_set = set(self.train.item_id.unique())
        self.user_maxid = np.max(self.train.user_id.unique())
        self.item_maxid = np.max(self.train.item_id.unique())

        self.train_seq = {}
        self.valid_seq = {}
        self.valid_neg_cand = {}
        self.my_valid = {}

        for u in self.user_set:
            items = self.train[self.train['user_id'] == u].item_id.values
            seq = np.pad(items, (maxlen+1-len(items), 0), 'constant')
            self.train_seq[u] = list(seq)

        valid_user_list = list(self.valid.user_id.unique())
        for u in valid_user_list:
            seq = self.train_seq[u][1:]
            seq.append(self.valid[self.valid['user_id'] == u].item_id.values[0])
            self.valid_seq[u] = seq

            self.valid_neg_cand[u] = list(self.candidate[self.candidate[0] == u].values[0][1:])

            # self.valid_cand[u] = [self.valid[self.valid['user_id'] == u].item_id.values[0]]
            # self.valid_cand[u].extend(list(self.candidate[self.candidate[0] == u].values[0][1:]))


    def create_valid(self):
        self.my_valid = []
        user_items=[]
        candidate_items=[]
        valid_user_list = list(self.valid.user_id.unique())
        for u in valid_user_list:
            user_items.append(self.train_seq[u][1:])
            cand = [self.valid[self.valid['user_id'] == u].item_id.values[0]]
            cand.extend(list(self.candidate[self.candidate[0] == u].values[0][1:]))
            candidate_items.append(cand)
        self.my_valid.append(user_items)
        self.my_valid.append(candidate_items)


    def sample_batch(self, batch_size=128):
        batch_x = []
        batch_yp = []
        batch_yn = []

        batch_uid = random.sample(self.user_set, batch_size)

        for u in batch_uid:
            x = self.train_seq[u][:-1]
            yp = self.train_seq[u][1:]
            yn = random.sample(self.item_set.difference(set(self.train_seq[u])), self.maxlen)

            batch_x.append(x)
            batch_yp.append(yp)
            batch_yn.append(yn)

        return batch_uid, batch_x, batch_yp, batch_yn
