import time
from random import sample

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from Dataset import Dataset
import pandas as pd

torch.autograd.set_detect_anomaly(True)
class FISSA(nn.Module):
    def __init__(self, n=22748, m=11146, epochs=200, batch_size=1, lr=0.01, d=20, L=3, k=10, num_heads=1, num_SAB=1, dropout_rate=0.5, device='gpu'):
        super().__init__()

        self.n = n
        self.m = m
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.d = d
        self.L = L  # length of markov chain
        # recall@k
        self.k = k
        self.num_heads = num_heads
        self.num_SAB = num_SAB
        self.dropout_rate = dropout_rate
        self.device = device
        self.dataset = Dataset('../dataset/Foursquare/Foursquare', is_valid=1)
        self.dataset.fix_length(self.L)
        self.dataset.create_valid()

        # init embedding
        # ==== item embedding matrix ====
        self.item_embedding = nn.Parameter(
            self.get_variable([self.m+1, self.d]),
            requires_grad=True,
        )
        # ==== position embedding matrix ====
        self.pos_embedding = nn.Parameter(
            self.get_variable([self.L, self.d]),
            requires_grad=True,
        )
        # ==== input embedding matrix ====
        # nothing to do,just add item_embedding to position_embedding,then dropout,plus padding_mask and layer_norm

        # init local_representation
        # include SABS
        self.Q_sabs = []
        self.K_sabs = []
        self.V_sabs = []
        self.conv1_sabs = []
        self.conv2_sabs = []
        for i in range(self.num_SAB):
            Q_temp = nn.Parameter(
                self.get_variable([self.d, self.d]),
                requires_grad=True,
            )
            K_temp = nn.Parameter(
                self.get_variable([self.d, self.d]),
                requires_grad=True,
            )
            V_temp = nn.Parameter(
                self.get_variable([self.d, self.d]),
                requires_grad=True,
            )
            conv1_temp = nn.Conv1d(self.L, self.L, 1)
            conv2_temp = nn.Conv1d(self.L, self.L, 1)
            self.Q_sabs.append(Q_temp)
            self.K_sabs.append(K_temp)
            self.V_sabs.append(V_temp)
            self.conv1_sabs.append(conv1_temp)
            self.conv2_sabs.append(conv2_temp)

        # init global_representation
        self.query_item = nn.Parameter(
            self.get_variable([1, 1, self.d]),
            requires_grad=True,
        )
        self.K_lba = nn.Parameter(
            self.get_variable([self.d, self.d]),
            requires_grad=True,
        )
        self.V_lba = nn.Parameter(
            self.get_variable([self.d, self.d]),
            requires_grad=True,
        )
        self.conv1_lba = nn.Conv1d(1, 1, 1)
        self.conv2_lba = nn.Conv1d(1, 1, 1)

        # init item_similarity_gating
        self.gated_weight = nn.Parameter(
            self.get_variable(
                [self.d * 3, 1],
                0.0, (2 / (3 * self.d + 1)) ** 0.5,
                initializer="trunc_norm",
            ),
            requires_grad=True,
        )
        self.gated_bias = nn.Parameter(
            self.get_variable(
                [1, 1],
                0.0, (2 / (2 * self.d + 1)) ** 0.5,
                initializer="trunc_norm",
            ),
            requires_grad=True,
        )

        # init hybrid_representation
        # nothing to do

    def forward_train(self, user_items, pos_items, neg_items, padding_mask):
        # ==== input batch data ====

        # ==== embedding ====
        item_embeddings = self.item_embedding[user_items, :]
        pos_embeds = self.pos_embedding[torch.arange(self.L), :]
        inputs = item_embeddings + pos_embeds
        inputs = F.dropout(inputs, self.dropout_rate)
        inputs *= padding_mask
        inputs = F.layer_norm(inputs, (self.d,), eps=1e-8)

        # ==== local respresentations ====
        # 不这么写，会导致一个tensor被直接替代掉，报错
        loc_embeddings_list = [inputs]
        for i in range(self.num_SAB):
            loc_embeddings_temp = self.self_attention_block(loc_embeddings_list[i], padding_mask, self.Q_sabs[i], self.K_sabs[i], self.V_sabs[i])
            loc_embeddings_temp = F.layer_norm(loc_embeddings_temp, (self.d,), eps=1e-8)
            loc_embeddings_temp = self.feed_forward(loc_embeddings_temp, self.conv1_sabs[i], self.conv2_sabs[i])
            loc_embeddings_temp *= padding_mask
            loc_embeddings_temp = F.layer_norm(loc_embeddings_temp, (self.d,), eps=1e-8)
            loc_embeddings_list.append(loc_embeddings_temp)
        loc_embeddings = loc_embeddings_list[-1]
        # ==== global respresentations ====
        glo_embedding = self.location_base_attention(inputs, padding_mask)
        glo_embedding = F.layer_norm(glo_embedding, (self.d,), eps=1e-8)
        glo_embedding = self.feed_forward(glo_embedding, self.conv1_lba, self.conv2_lba)
        glo_embedding = F.layer_norm(glo_embedding, (self.d,), eps=1e-8)

        # ==== item similarity gating ====
        positive_embeds = self.item_embedding[pos_items, :]
        negative_embeds = self.item_embedding[neg_items, :]
        gating = self.item_similarity_gating(
            torch.tile(item_embeddings, [2, 1, 1]),
            torch.tile(glo_embedding, [2, self.L, 1]),
            torch.concat([positive_embeds, negative_embeds], 0)
        )
        gating = torch.sigmoid(gating)
        # ==== hybrid/final respresentations ====
        tiled_loc_embeds = torch.tile(loc_embeddings, [2, 1, 1])
        tiled_glo_embeds = torch.tile(glo_embedding, [2, 1, 1])

        outputs = tiled_loc_embeds * gating + \
                  tiled_glo_embeds * (1 - gating)
        outputs *= torch.tile(padding_mask, [2, 1, 1])
        outputs = F.layer_norm(outputs, (self.d,), eps=1e-8)

        outputs_positive = outputs[:user_items.shape[0]]
        outputs_negative = outputs[user_items.shape[0]:]

        outputs = torch.stack(
            [
                torch.sum(outputs_positive * positive_embeds, -1),
                torch.sum(outputs_negative * negative_embeds, -1),
            ],
            dim=0,
        )

        return outputs


    def forward_eval(self, user_items, padding_mask, candidate_items):
        # ==== input batch data ====

        # ==== embedding ====
        item_embeddings = self.item_embedding[user_items, :]
        candidate_embeddings = self.item_embedding[candidate_items, :]
        pos_embeds = self.pos_embedding[torch.arange(self.L), :]
        inputs = item_embeddings + pos_embeds
        inputs = F.dropout(inputs, self.dropout_rate)
        inputs *= padding_mask
        inputs = F.layer_norm(inputs, (self.d,), eps=1e-8)

        # ==== local respresentations ====
        loc_embeddings_list = [inputs]
        for i in range(self.num_SAB):
            loc_embeddings_temp = self.self_attention_block(loc_embeddings_list[i], padding_mask, self.Q_sabs[i],
                                                            self.K_sabs[i], self.V_sabs[i])
            loc_embeddings_temp = F.layer_norm(loc_embeddings_temp, (self.d,), eps=1e-8)
            loc_embeddings_temp = self.feed_forward(loc_embeddings_temp, self.conv1_sabs[i], self.conv2_sabs[i])
            loc_embeddings_temp *= padding_mask
            loc_embeddings_temp = F.layer_norm(loc_embeddings_temp, (self.d,), eps=1e-8)
            loc_embeddings_list.append(loc_embeddings_temp)
        loc_embeddings = loc_embeddings_list[-1]

        # ==== global respresentations ====
        glo_embedding = self.location_base_attention(inputs, padding_mask)
        glo_embedding = F.layer_norm(glo_embedding, (self.d,), eps=1e-8)
        glo_embedding = self.feed_forward(glo_embedding, self.conv1_lba, self.conv2_lba)
        glo_embedding = F.layer_norm(glo_embedding, (self.d,), eps=1e-8)

        # ==== item similarity gating ====
        gating = self.item_similarity_gating(
            torch.tile(item_embeddings[:, -1, :].unsqueeze(1), [1, 101, 1]),
            torch.tile(glo_embedding, [1, 101, 1]),
            torch.tile(candidate_embeddings, [1, 1, 1]),
        )
        gating = torch.sigmoid(gating)
        # ==== hybrid/final respresentations ====
        tiled_loc_embeds = torch.tile(loc_embeddings[:, -1, :].unsqueeze(1), [1, 101, 1])
        tiled_glo_embeds = torch.tile(glo_embedding, [1, 101, 1])

        outputs = tiled_loc_embeds * gating + tiled_glo_embeds * (1 - gating)
        outputs = F.layer_norm(outputs, (self.d,), eps=1e-8)
        outputs = torch.sum(outputs * gating, -1)

        return outputs



    # =========================================================================================================
    def item_similarity_gating(self, ms, y, mi):
        inputs = torch.concat([ms, y, mi], -1).reshape([-1, self.d * 3])
        inputs = F.dropout(inputs, p=self.dropout_rate)

        logits = torch.matmul(inputs, self.gated_weight) + self.gated_bias
        logits = F.dropout(logits, p=self.dropout_rate)
        logits = logits.reshape([-1, y.shape[1], 1])

        outputs = torch.sigmoid(logits)

        return outputs

    def location_base_attention(self, x, padding_mask):
        Kx = x @ self.K_lba
        Vx = x @ self.V_lba
        Kh = torch.concat(torch.chunk(Kx, self.num_heads, dim=2), dim=0)
        Vh = torch.concat(torch.chunk(Vx, self.num_heads, dim=2), dim=0)
        outputs = torch.matmul(self.query_item, Kh.transpose(1, 2))

        key_mask = torch.tile(padding_mask, [self.num_heads, 1, outputs.shape[1]]).transpose(1, 2)
        outputs = torch.where(torch.eq(key_mask, 0),
                              torch.ones_like(outputs) * (-2 ** 32 + 1), outputs)

        outputs = F.softmax(outputs, dim=-1)
        outputs = F.dropout(outputs, self.dropout_rate)
        outputs = torch.matmul(outputs, Vh)
        outputs = torch.concat(torch.chunk(outputs, self.num_heads, dim=0), dim=-1)
        outputs = F.dropout(outputs, self.dropout_rate)

        return outputs

    def self_attention_block(self, x, padding_mask, Q, K, V):
        Qx = x @ Q
        Kx = x @ K
        Vx = x @ V
        Qh = torch.concat(torch.chunk(Qx, self.num_heads, dim=2), dim=0)
        Kh = torch.concat(torch.chunk(Kx, self.num_heads, dim=2), dim=0)
        Vh = torch.concat(torch.chunk(Vx, self.num_heads, dim=2), dim=0)
        outputs = torch.matmul(Qh, Kh.transpose(1, 2)) / (Kh.shape[-1] ** 0.5)
        tril = torch.tril(torch.ones_like(outputs[0, :, :]))
        # (batch_size * num_heads) * L * L
        causality_mask = torch.tile(torch.unsqueeze(tril, 0), [outputs.shape[0], 1, 1])
        outputs = torch.where(torch.eq(causality_mask, 0),
                              torch.ones_like(outputs) * (-2 ** 32 + 1), outputs)
        # (batch_size * num_heads) * L * L
        key_mask = torch.tile(padding_mask, [self.num_heads, 1, outputs.shape[1]]).transpose(1, 2)
        outputs = torch.where(torch.eq(key_mask, 0), torch.ones_like(outputs) * (-2 ** 32 + 1), outputs)

        # outputs = F.softmax(outputs, dim=-1)

        query_mask = torch.tile(padding_mask, [self.num_heads, 1, x.shape[1]])
        outputs *= query_mask
        outputs = F.dropout(outputs, p=self.dropout_rate)

        outputs = torch.matmul(outputs, Vh)
        outputs = torch.concat(torch.chunk(outputs, self.num_heads, dim=0), dim=-1)
        outputs = F.dropout(outputs, p=self.dropout_rate)

        outputs += x
        return outputs


    def feed_forward(self, x, conv_1, conv_2):
        outputs = F.relu(conv_1(x))
        outputs = F.dropout(outputs, self.dropout_rate)
        outputs = conv_2(outputs)
        outputs = F.dropout(outputs, self.dropout_rate)
        outputs += x
        return outputs
    # =========================================================================================================
    # params：mean and variance
    def get_variable(self, shape: list, *params, initializer="xavier",):

        out = torch.empty(shape)
        if initializer == "xavier":
            if params:
                out = nn.init.xavier_uniform_(out, params[0])
            else:
                out = nn.init.xavier_uniform_(out)
        elif initializer == "trunc_norm":
            out = nn.init.trunc_normal_(out, params[0], params[1])

        return out



    # =========================================================================================================
    def train_model(self):
        # for p in self.parameters():
        #     print(p)
        #
        recall_at_k_list = []
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        self.eval()
        recall_at_k = self.eval_topk()
        recall_at_k_list.append(recall_at_k)
        print('recall@k:{:.4f}'.format(recall_at_k * 100))
        for epoch in range(self.epochs):

            start = time.time()
            self.train()
            batch = int(self.n / self.batch_size)
            loss_sum = 0.
            for i in range(batch):
                users, user_items, pos_items, neg_items = self.dataset.sample_batch(self.batch_size)

                user_items = torch.as_tensor(user_items).long()
                padding_mask = torch.not_equal(user_items, 0).unsqueeze(-1)

                pos_items = torch.as_tensor(pos_items).long()
                neg_items = torch.as_tensor(neg_items).long()

                outputs = self.forward_train(
                    user_items,
                    pos_items,
                    neg_items,
                    padding_mask,
                )

                flag_exist = torch.ne(pos_items, 0)
                loss = torch.mean(
                    -torch.log(torch.sigmoid(outputs[0]) + 1e-24) * flag_exist \
                    -torch.log(1 - torch.sigmoid(outputs[1]) + 1e-24) * flag_exist
                )
                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.eval()
            recall_at_k = self.eval_topk()
            end = time.time()
            running_time = end - start
            # recall_at_k = 1
            recall_at_k_list.append(recall_at_k)
            print('epoch:{:03d}\t\trecall@k:{:.4f}\t\tloss:{:.4f}\t\trunning time:{:.3f}'.format(epoch, recall_at_k * 100,
                                                                                             loss_sum, running_time))
            if(epoch%10==0):

                df = pd.DataFrame(recall_at_k_list, columns=['Recall@10'])
                df.to_csv('recall@10.csv')
                filename = 'checkpoint{:2d}.pth.tar'.format(epoch)
                state={
                    'epoch': epoch + 1,
                    'state_dict': self.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)




    def eval_topk(self):
        Recall_at_K = 0.
        valid_user_list = list(self.dataset.valid.user_id.unique())
        # for u in sample(valid_user_list, 500):
        # for u in valid_user_list:
        #     user_items = self.dataset.train_seq[u][1:]
        #     user_items = torch.as_tensor(user_items).unsqueeze(0).long()
        #     candidate_items = [self.dataset.valid_seq[u][-1]]
        #     candidate_items.extend(self.dataset.valid_neg_cand[u])
        #     candidate_items = torch.as_tensor(candidate_items).unsqueeze(0).long()
        #     padding_mask = torch.not_equal(user_items, 0).unsqueeze(-1)
        #     padding_mask = torch.as_tensor(padding_mask)
        #
        #     outputs = self.forward_eval(user_items, padding_mask, candidate_items)
        #     items_score = outputs.detach_().cpu().numpy()
        #
        #     topK = np.argsort(np.argsort(-items_score))[:, 0:self.k]
        #
        #     Recall_at_K += np.sum(topK == 0)
            # print(topK)

        # 一次全丢进去
        user_items = self.dataset.my_valid[0]
        user_items = torch.as_tensor(user_items).long()
        candidate_items = self.dataset.my_valid[1]
        candidate_items = torch.as_tensor(candidate_items).long()
        padding_mask = torch.not_equal(user_items, 0).unsqueeze(-1)
        padding_mask = torch.as_tensor(padding_mask)

        outputs = self.forward_eval(user_items, padding_mask, candidate_items)
        items_score = outputs.detach_().cpu().numpy()
        # print(np.argsort(np.argsort(-items_score)).shape)
        topK = np.argsort(np.argsort(-items_score))[:, 0]
        # topK = np.argsort(-items_score)[:, 0:self.k]
        # print(topK.shape)
        Recall_at_K += np.sum(topK < self.k)
        # print(topK)

        # temp
        # for u in valid_user_list:
        #     user_items = [self.dataset.train_seq[u][1:],self.dataset.train_seq[u][1:]]
        #     user_items = torch.as_tensor(user_items).long()
        #     candidate_items = [self.dataset.valid_seq[u][-1]]
        #     candidate_items.extend(self.dataset.valid_neg_cand[u])
        #     candidate_items = [candidate_items,candidate_items]
        #     candidate_items = torch.as_tensor(candidate_items).long()
        #     padding_mask = torch.not_equal(user_items, 0).unsqueeze(-1)
        #     padding_mask = torch.as_tensor(padding_mask)
        #
        #     outputs = self.forward_eval(user_items, padding_mask, candidate_items)
        #     items_score = outputs.detach_().cpu().numpy()
        #
        #     topK = np.argsort(np.argsort(-items_score))[:, 0:self.k]
        #
        #     Recall_at_K += np.sum(topK == 0)
            # print(topK)

        # return Recall_at_K / 500
        return Recall_at_K / len(valid_user_list)

    def forward(self, x):
        user_items = x['user_items']
        pos_items = x['pos_items']
        neg_items = x['neg_items']
        padding_mask = x['padding_mask']
        self.forward_train(user_items,pos_items,neg_items,padding_mask)
# =========================================================================================================
if __name__ == '__main__':
    Fissa = FISSA(n=22748, m=11146, epochs=500, batch_size=200, lr=0.01, d=50, L=30, k=10, num_SAB=3, dropout_rate=0.5, device='cuda:0')
    Fissa.train_model()
