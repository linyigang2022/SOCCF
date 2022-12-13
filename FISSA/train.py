from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import tensorflow as tf
import time

from Dataset import Dataset
from Model import Model
from evaluate import *


def parse_args():
    parser = argparse.ArgumentParser(description="Configurations.")
    parser.add_argument('--path', type=str, default='./DataSets/processed_data/', help='Path of data files.')
    parser.add_argument('--dataset', type=str, default='Steam', help='Name of the dataset (e.g. Steam).')
    parser.add_argument('--valid', type=int, default=1, help='Whether to evaluate on the validation set.')

    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

    parser.add_argument('--maxlen', type=int, default=50, help='Maximum length of sequences.')
    parser.add_argument('--hidden_units', type=int, default=50, help='i.e. latent vector dimensionality.')
    parser.add_argument('--num_blocks', type=int, default=1, help='Number of self-attention blocks.')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads for attention.')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--l2_reg', type=float, default=0.0)

    parser.add_argument('--ext_modules', type=str, default='lgt',   ## -l: local/causal representation; # -g: global representations; 
                        help='Extension modules based on SASRec. ') ## -c: consistency-aware gating; # -t: target-aware gating, i.e., the item similarity gating.
    parser.add_argument('--gating_mode', type=str, default='individual') # individual, feature
    parser.add_argument('--gating_input', type=str, default='concat') # concat, prod

    parser.add_argument('--eva_interval', type=int, default=50, help='Number of epoch interval for evaluation.')
    parser.add_argument('--eva_grouped', type=int, default=0, help='Whether to show grouped evaluation results.')

    parser.add_argument('--save_model', type=int, default=0, help='Whether to save the tensorflow model.')
    parser.add_argument('--restore_file', type=str, default='', help='Tensorflow model to restore.')
    parser.add_argument('--restore_epoch', type=int, default=0, help='Model epoch to restore.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('\n'.join([str(k) + ',' + str(v) for k, v in vars(args).items()]))

    # Loading data
    dataset = Dataset(args.path + args.dataset, args.valid)
    dataset.fix_length(args.maxlen)

    # Build model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    model = Model(dataset.user_maxid, dataset.item_maxid, args)

    # writer = tf.summary.FileWriter("./", sess.graph)
    sess.run(tf.global_variables_initializer())

    if args.restore_file != '':
        model_saver = tf.train.Saver()
        model_saver.restore(sess, args.restore_file)
        print("Model restored from file: %s" % args.restore_file)

    # ---- Train model ----
    num_full_batch = len(dataset.user_set) // args.batch_size
    size_last_batch = len(dataset.user_set) % args.batch_size

    print("----")
    t_start = time.time()

    best_rec = 0.0
    best_epoch = 0
    for ep in range(args.restore_epoch + 1, args.num_epochs + 1):
        for b in range(num_full_batch):
            users, inps, poss, negs = dataset.sample_batch(args.batch_size)
            _, loss = sess.run([model.train_op, model.loss], 
                               {model.user_id: users, model.input_id: inps, 
                                model.pos_id: poss, model.neg_id: negs, 
                                model.is_training: True})
        users, inps, poss, negs = dataset.sample_batch(size_last_batch)
        _, loss = sess.run([model.train_op, model.loss], 
                           {model.user_id: users, model.input_id: inps, 
                            model.pos_id: poss, model.neg_id: negs, 
                            model.is_training: True})
        # print(loss)

        # Evaluate model
        if ep % args.eva_interval == 0:
            u_list = list(dataset.valid_seq.keys())
            input_list = np.array(list(dataset.valid_seq.values()))[:, :-1]

            tar = np.array(list(dataset.valid_seq.values()))[:, -1][:, np.newaxis]
            neg_cand = np.array(list(dataset.valid_neg_cand.values()))
            cand_list = np.hstack((tar, neg_cand)).tolist()

            pred_ratings = []
            while len(u_list) > 0:
                pred_ratings.extend(list(sess.run(model.cand_rating,
                                                  {model.user_id: u_list[-args.batch_size:], model.input_id: input_list[-args.batch_size:], 
                                                   model.candidate_id: cand_list[-args.batch_size:], 
                                                   model.is_training: False})))
                u_list = u_list[:-args.batch_size]
                input_list = input_list[:-args.batch_size]
                cand_list = cand_list[:-args.batch_size]

            rec, ndcg, mrr = evaluate_rec_ndcg_mrr(pred_ratings, top_k=10, row_target_position=0)
            print("epoch: %4d, HR@10: %.6f, NDCG@10: %.6f, MRR: %.6f [loss: %.6f, time: %ds]" % (ep, rec, ndcg, mrr, loss, (time.time()-t_start)), end=' ')

            # Save model
            if rec > best_rec:
                print("*", end=' ')
                best_rec = rec
                best_epoch = ep

                if args.save_model == 1:
                    model_saver = tf.train.Saver()
                    save_fname = ("FISSA_" + args.ext_modules + '_' 
                                  + args.dataset + ("_valid" if args.valid > 0 else "_test") 
                                  + (('_' + args.gating_mode + '_' + args.gating_input) if 't' in args.ext_modules else '') 
                                  + "_nb" + str(args.num_blocks) + "_nh" + str(args.num_heads))
                    save_path = model_saver.save(sess, "Model/model_" + save_fname + ".ckpt")
                    print("Model saved: %s" % save_path, end=' ')
            print("")

            if ep >= best_epoch + 100:
                break
    # --------

    # final test
    if args.eva_grouped == 1:
        if args.save_model == 1:
            model_saver.restore(sess, "Model/model_" + save_fname + ".ckpt")

        u_list = list(dataset.valid_seq.keys())
        input_list = np.array(list(dataset.valid_seq.values()))[:, :-1]
        seq_length = np.count_nonzero(np.array(input_list), axis=-1) + 1

        tar = np.array(list(dataset.valid_seq.values()))[:, -1][:, np.newaxis]
        neg_cand = np.array(list(dataset.valid_neg_cand.values()))
        cand_list = np.hstack((tar, neg_cand)).tolist()

        pred_ratings = []
        while len(u_list) > 0:
            pred_ratings.extend(list(sess.run(model.cand_rating,
                                              {model.user_id: u_list[-args.batch_size:], model.input_id: input_list[-args.batch_size:], 
                                               model.candidate_id: cand_list[-args.batch_size:], 
                                               model.is_training: False})))
            u_list = u_list[:-args.batch_size]
            input_list = input_list[:-args.batch_size]
            cand_list = cand_list[:-args.batch_size]

        rec, ndcg, mrr = evaluate_rec_ndcg_mrr_grouped(pred_ratings, seq_length, 
                                                       top_k=10, row_target_position=0, group_segment=[5,10,20,50])
