from __future__ import division
from __future__ import print_function
import numpy as np
import math
import random
import pandas as pd


class Fossil():
    ''' Implementation of the algorithm presented in "Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation", by He R. and McAuley J., 2016.
    '''

    def __init__(self, d=100, L=1, k=10, n=22748, m=11146, epochs=200, lr=0.01, alpha_v=0.1, aplha_w=0.1, beta_v=0.1,
                 beta_eta=0.1, **kwargs):

        self.n = n
        self.m = m
        self.epochs = epochs
        self.lr = lr
        self.d = d
        self.L = L  # length of markov chain
        # recall@k
        self.k = k
        self.alpha_v = alpha_v
        self.alpha_w = aplha_w
        self.beta_v = beta_v
        self.beta_eta = beta_eta
        self.nabla_b_true = 0
        self.nabla_b_false = 0
        self.nabla_V_true = 0
        self.nabla_V_false = 0
        self.nabla_eta = 0
        self.nabla_eta_u = 0
        self.nabla_W_early = 0
        self.nabla_W_true = 0
        self.nabla_W_short = 0
        # Initialize the model parameters
        # self.W = np.random.randn(self.m+1, self.d).astype(np.float32)
        # self.V = np.random.randn(self.m+1, self.d).astype(np.float32)
        # self.eta = np.random.randn(1, self.L).astype(np.float32)
        # self.eta_u = np.random.randn(self.n+1, self.L).astype(np.float32)
        # self.bias = np.random.randn(self.m+1, 1).astype(np.float32)
        self.W = np.random.normal(0, 0.01, (self.m + 1, self.d)).astype(np.float32)
        self.V = np.random.normal(0, 0.01, (self.m + 1, self.d)).astype(np.float32)
        self.eta = np.random.normal(0, 0.01, (1, self.L)).astype(np.float32)
        self.eta_u = np.random.normal(0, 0.01, (self.n + 1, self.L)).astype(np.float32)
        self.bias = np.random.normal(0, 0.01, (self.m + 1, 1)).astype(np.float32)

    def load_data(self):
        # load train data
        self.S = [[] for row in range(22748 + 1)]
        self.interactions_count = [0] * (22748 + 1)
        self.sampler = pd.read_csv('../dataset/Foursquare/Foursquare_train.csv', header=None).astype(int)
        for index, row in self.sampler.iterrows():
            user, item = row[0], row[1]
            self.S[user].append(row[1])
            row[2] = self.interactions_count[user] + 1
            self.interactions_count[user] = self.interactions_count[user] + 1
        df = pd.DataFrame(self.S)
        df.to_csv('S.csv', index=False)
        self.sampler.to_csv('sampler.csv', index=False)
        self.test_items = pd.read_csv('../dataset/Foursquare/Foursquare_valid.csv', header=None).astype(int)

        # load negative data
        self.negative = [[] for row in range(22748 + 1)]
        negative_sampler = pd.read_csv('../dataset/Foursquare/Foursquare_negative.csv', header=None).astype(int)
        for index, row in negative_sampler.iterrows():
            user_id = row[0]
            self.negative[user_id] = row[1:]
        df = pd.DataFrame(self.negative)
        df.to_csv('negative.csv', index=False)

    def item_score(self, user_id, long_term, short_term, item=None):
        ''' Compute the prediction score of the Fossil model for the item "item", based on the list of items "user_items".
        '''

        return self.bias[item].squeeze() + np.dot(long_term + short_term, self.V[item, :].T)

    def sgd_step(self, user_id, user_items, t, true_item, false_item):
        ''' Make one SGD update, given that the interaction between user and true_item exists,
        but the one between user and false_item does not.

        return error
        '''
        # 求 Wi'的和
        Wi_sum_true = self.W[user_items[0], :].copy()
        # Wi_sum_true = np.expand_dims(Wi_sum_true, axis=0)
        Wi_sum_false = self.W[user_items[0], :].copy()
        # Wi_sum_false = np.expand_dims(Wi_sum_false, axis=0)
        for i in range(0, self.interactions_count[user_id]):
            if user_items[i] == true_item:
                Wi_sum_false += self.W[user_items[i]]
                continue
            Wi_sum_true += self.W[user_items[i]]
            Wi_sum_false += self.W[user_items[i]]
        long_term_true = Wi_sum_true / math.sqrt((self.interactions_count[user_id] - 1))
        long_term_false = Wi_sum_false / math.sqrt((self.interactions_count[user_id]))
        short_term = np.dot(
            self.eta + self.eta_u[user_id],
            (self.W[user_items[t - self.L - 1:t - 1]])
        ).squeeze()

        # Compute error
        x_true = self.item_score(user_id, long_term_true, short_term, true_item)
        x_false = self.item_score(user_id, long_term_false, short_term, false_item)
        sigmoid = 1 / (1 + math.exp(x_true - x_false))  # sigmoid of the error
        # sigmoid = 1 / (1 + math.exp(min(-15, max(15, x_false - x_true))))

        # compute gradient
        nabla_b_true = self.beta_v * self.bias[true_item] - sigmoid
        nabla_b_false = self.beta_v * self.bias[false_item] + sigmoid
        nabla_V_true = self.alpha_v * self.V[true_item] - sigmoid * (long_term_true + short_term)
        nabla_V_false = self.alpha_v * self.V[false_item] - sigmoid * (long_term_false + short_term)
        nabla_eta = self.beta_eta * self.eta - sigmoid * np.dot(
            self.W[user_items[t - self.L - 1:t - 1]],
            (self.V[true_item] - self.V[false_item]).T
        )
        nabla_eta_u = self.beta_eta * self.eta_u[user_id] - sigmoid * np.dot(
            self.W[user_items[t - self.L - 1:t - 1]],
            (self.V[true_item] - self.V[false_item]).T
        )
        nabla_W_early = self.alpha_w * self.W[user_items[0:t - self.L - 1]] - sigmoid * (np.dot(
            1 / math.sqrt(self.interactions_count[user_id] - 1),
            self.V[true_item]
        ) - np.dot(1 / math.sqrt(self.interactions_count[user_id]),
                   self.V[false_item]))
        nabla_W_true = self.alpha_w * self.W[true_item] - sigmoid * (-np.dot(
            1 / math.sqrt(self.interactions_count[user_id]),
            self.V[false_item]
        ))
        # nabla_W_true = np.expand_dims(nabla_W_true, axis=0)
        L2 = sigmoid * (np.repeat(np.expand_dims(self.V[true_item] - self.V[false_item], axis=0), self.L, axis=0) * (
                    self.eta + self.eta_u[user_id]).T)
        L3 = sigmoid * (np.dot(
            1 / math.sqrt(self.interactions_count[user_id] - 1),
            self.V[true_item]
        ) - np.dot(
            1 / math.sqrt(self.interactions_count[user_id]),
            self.V[false_item]
        ))
        nabla_W_short = self.alpha_w * self.W[user_items[t - self.L - 1:t - 1]] - L2 - L3


        # Update
        self.bias[true_item] -= self.lr * nabla_b_true
        self.bias[false_item] -= self.lr * nabla_b_false
        self.V[true_item] -= self.lr * nabla_V_true
        self.V[false_item] -= self.lr * nabla_V_false
        self.eta -= self.lr * nabla_eta
        self.eta_u[user_id] -= self.lr * nabla_eta_u
        self.W[user_items[0:t - self.L - 1]] -= self.lr * nabla_W_early
        self.W[true_item] -= self.lr * nabla_W_true
        self.W[user_items[t - self.L - 1: t - 1]] -= self.lr * nabla_W_short

        return sigmoid

    def recall_at_k(self, user_id=None, test_item=None):
        ''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
        '''
        user_items = self.S[user_id]
        query_items = [test_item]
        query_items.extend(self.negative[user_id].values.tolist())
        Wi_sum = self.W[user_items[0], :].copy()
        for i in range(0, self.interactions_count[user_id]):
            Wi_sum += self.W[user_items[i]]
        long_term = Wi_sum / math.sqrt((self.interactions_count[user_id]))
        t = self.interactions_count[user_id]+1
        short_term = np.dot(
            self.eta + self.eta_u[user_id],
            (self.W[user_items[t - self.L - 1:t - 1]])
        ).squeeze()

        # Compute error
        items_score = self.item_score(user_id, long_term, short_term, np.array(query_items))
        rank_101 = np.argsort(np.argsort(-items_score))
        # 如果test_item的排序在101个样本中小于k，那么召回成果，返回1，否则返回0
        # print(rank_101)
        # return rank_101[0]
        if(rank_101[0]<self.k):
            return 1
        else:
            return 0

    def validate(self):
        k = 10
        recall_sum = 0.
        count = 0
        for index, u_i_pair in self.test_items.iterrows():
            count = count + 1
            user_id = u_i_pair[0]
            test_item = u_i_pair[1]
            recall_sum += self.recall_at_k(user_id, test_item)
            # if(count%500==0):
            #     print("validate-- mean recall@k:{:.2f}".format(recall_sum/count))

        print("validate-- mean recall@k:{:.4f}".format(recall_sum / count))
    def train_uipair(self):
        self.validate()
        for i in range(self.epochs):
            count = 0
            loss = 0
            u_i_pairs = self.sampler.sample(frac=1)
            for index, u_i_pair in u_i_pairs.iterrows():
                t = u_i_pair[2]
                # 抽到的样本的t 不大于 markov的长度L，则没法计算，跳过
                if t <= self.L:
                    continue
                user_id = u_i_pair[0]
                user_items = self.S[u_i_pair[0]]
                true_item = u_i_pair[1]
                # print(user_id)
                random.seed(100)
                false_item = random.choice(self.negative[user_id])
                t = u_i_pair[2]
                sigmoid = self.sgd_step(user_id, user_items, t, true_item, false_item)
                loss = loss + sigmoid
                if (count % 2000 == 0):
                    print('epochs: {},batch:{},loss:{:.3f}'.format(i, count, loss))
                    loss = 0
                count = count + 1
            self.validate()

    def train_u(self):
        for i in range(self.epochs):
            count = 0
            loss = 0
            self.sampler = [row for row in range(1, self.n+1)]
            for _ in range(0, self.n):
                user_id = random.sample(self.sampler, 1)[0]
                user_items = self.S[user_id]
                for t in range(self.L+1, len(user_items)+1):
                    true_item = user_items[t-1]
                    false_item = random.sample(self.negative[user_id].values.tolist(), 1)[0]
                    sigmoid = self.sgd_step(user_id, user_items, t, true_item, false_item)
                    loss = loss + sigmoid
                if (count % 2000 == 0):
                    print('epochs: {},batch:{},loss:{:.3f}'.format(i, count, loss))
                    loss = 0
                count = count + 1
            self.validate()
fossil = Fossil(d=20, L=2, k=10, n=22748, m=11146, epochs=200, lr=0.01, alpha_v=0.01, aplha_w=0.01, beta_v=0.01, beta_eta=0.01)
fossil.load_data()
fossil.train_u()

# def _get_model_filename(self, epochs):
#     '''Return the name of the file to save the current model
#     '''
#     filename = "fossil_ne" + str(epochs) + "_lr" + str(self.init_learning_rate) + "_an" + str(
#         self.annealing_rate) + "_k" + str(self.k) + "_o" + str(self.order) + "_reg" + str(self.reg) + "_ini" + str(
#         self.init_sigma)
#
#     return filename + ".npz"
# def save(self, filename):
#     '''Save the parameters of a network into a file
#     '''
#     print('Save model in ' + filename)
#     if not os.path.exists(os.path.dirname(filename)):
#         os.makedirs(os.path.dirname(filename))
#     np.savez(filename, V=self.V, H=self.H, bias=self.bias, eta=self.eta, eta_bias=self.eta_bias)
#
# def load(self, filename):
#     '''Load parameters values form a file
#     '''
#     f = np.load(filename)
#     self.V = f['V']
#     self.H = f['H']
#     self.bias = f['bias']
#     self.eta = f['eta']
#     self.eta_bias = f['eta_bias']
