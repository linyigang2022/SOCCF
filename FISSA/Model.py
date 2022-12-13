import tensorflow as tf
import tensorflow as tf

class Model(object):
    def __init__(self, num_user, num_item, args):
        # ==== some configurations ====
        self.num_user = num_user
        self.num_item = num_item
        self.maxlen = args.maxlen
        self.num_units = args.hidden_units
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.l2_reg = args.l2_reg

        # ==== input batch data ====
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.user_id = tf.placeholder(tf.int32, shape=(None))
        self.input_id = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.pos_id = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.neg_id = tf.placeholder(tf.int32, shape=(None, self.maxlen))

        padding_mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_id, 0)), -1)

        # ==== item embedding matrix ====
        with tf.variable_scope("item_embeddings"):
            self.item_emb_mat = tf.get_variable('lookup_table', 
                                                dtype = tf.float32, 
                                                shape = [self.num_item, self.num_units], 
                                                regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg))
            self.item_emb_mat = tf.concat((tf.zeros(shape = [1, self.num_units]), self.item_emb_mat), 0) * (self.num_units ** 0.5)

            item_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.input_id)

        # ==== position embedding matrix ====
        with tf.variable_scope("position_embeddings"):
            pos_emb_mat = tf.get_variable('lookup_table', 
                                          dtype = tf.float32, 
                                          shape = [self.maxlen, self.num_units], 
                                          regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg))

            pos_embs = tf.nn.embedding_lookup(pos_emb_mat, tf.range(self.maxlen))

        # ==== input embedding matrix ====
        with tf.variable_scope("input_matrix"):
            inputs = item_embs + pos_embs
            inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            inputs = inputs * padding_mask
            inputs = self.layer_normalization(inputs)

        # ==== output initialization ====
        loc_embs = tf.zeros([tf.shape(inputs)[0], self.maxlen, self.num_units])
        glo_embs = tf.zeros([tf.shape(inputs)[0], 1, self.num_units])

        # ==== local representation (hierarchical self-attention blocks) ====
        if 'l' in args.ext_modules:
            loc_embs = inputs

            for b in range(args.num_blocks):
                with tf.variable_scope("self-attention_blocks_%d" % b):
                    # multi-head attention
                    loc_embs, laststep_attention = self.multihead_attention(loc_embs, 
                                                                           padding_mask=padding_mask, 
                                                                           num_heads=args.num_heads, 
                                                                           dropout_rate=args.dropout_rate, 
                                                                           score='scaled_dot', 
                                                                           causality=True, 
                                                                           get_att='lastq_ave',
                                                                           residual=True)
                    loc_embs = self.layer_normalization(loc_embs)

                    # feed forward
                    loc_embs = self.feed_forward(loc_embs)
                    loc_embs *= padding_mask
                    loc_embs = self.layer_normalization(loc_embs)

                    # output for attention visualization
                    if b == 0:
                        self.bottom_laststep_attention = laststep_attention
                    if b == args.num_blocks-1:
                        self.top_laststep_attention = laststep_attention

        # ==== global respresentations ====
        self.global_attention = tf.zeros([args.num_heads, 1, self.maxlen])

        if 'g' in args.ext_modules:
            with tf.variable_scope("global_representations"):
                glo_embs, self.global_attention = self.multihead_attention(item_embs, 
                                                                           padding_mask=padding_mask, 
                                                                           num_heads=args.num_heads, 
                                                                           dropout_rate=args.dropout_rate, 
                                                                           score='location', 
                                                                           causality=False, #> for g_causality: True
                                                                           get_att='lastq_multi',
                                                                           residual=False)
                glo_embs = self.layer_normalization(glo_embs) # shape: [batch_size, 1, num_units]

                glo_embs = self.feed_forward(glo_embs)
                glo_embs = self.layer_normalization(glo_embs)

        # ==== hybrid/final respresentations ====
        positive_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.pos_id)
        negative_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.neg_id)

        self.candidate_id = tf.placeholder(tf.int32, shape=(None, 101))
        candidate_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.candidate_id)

        if 't' in args.ext_modules:
            with tf.variable_scope("gating_parameters"):
                if args.gating_mode == 'individual':
                    if args.gating_input == 'concat':
                        W_g = tf.Variable(tf.truncated_normal(shape=[3 * self.num_units, 1], 
                                                              mean=0.0, stddev=tf.sqrt(tf.div(2.0, 3 * self.num_units + 1))), 
                                          name='weights_for_gating', 
                                          dtype=tf.float32)
                    elif args.gating_input == 'prod':
                        W_g = tf.Variable(tf.truncated_normal(shape=[2 * self.num_units, 1], 
                                                              mean=0.0, stddev=tf.sqrt(tf.div(2.0, 2 * self.num_units + 1))), 
                                          name='weights_for_gating', 
                                          dtype=tf.float32)
                    b_g = tf.Variable(tf.truncated_normal(shape=[1, 1], 
                                                          mean=0.0, stddev=tf.sqrt(tf.div(2.0, 2 * self.num_units + 1))), 
                                      name='bias_for_gating', 
                                      dtype=tf.float32)
                elif args.gating_mode == 'feature':
                    if args.gating_input == 'concat':
                        W_g = tf.Variable(tf.truncated_normal(shape=[3 * self.num_units, self.num_units], 
                                                              mean=0.0, stddev=tf.sqrt(tf.div(2.0, 4 * self.num_units))), 
                                          name='weights_for_gating', 
                                          dtype=tf.float32)
                    elif args.gating_input == 'prod':
                        W_g = tf.Variable(tf.truncated_normal(shape=[2 * self.num_units, self.num_units], 
                                                              mean=0.0, stddev=tf.sqrt(tf.div(2.0, 3 * self.num_units))), 
                                          name='weights_for_gating', 
                                          dtype=tf.float32)
                    b_g = tf.Variable(tf.truncated_normal(shape=[1, self.num_units], 
                                                          mean=0.0, stddev=tf.sqrt(tf.div(2.0, 3 * self.num_units))), 
                                      name='bias_for_gating', 
                                      dtype=tf.float32)

            # for training ...
            gat_val_tr = self.item_similarity_gating(tf.tile(glo_embs, [2, self.maxlen, 1]), 
                                                     tf.tile(item_embs, [2, 1, 1]), 
                                                     tf.concat([positive_embs, negative_embs], 0), 
                                                     W_g, b_g, 
                                                     mode=args.gating_mode, 
                                                     inp_func=args.gating_input)
            gat_val_tr = tf.nn.sigmoid(gat_val_tr)

            tile_loc_embs = tf.tile(tf.layers.dropout(loc_embs, 
                                                      rate=args.dropout_rate, training=tf.convert_to_tensor(self.is_training)), 
                                    [2, 1, 1])
            tile_glo_embs = tf.tile(tf.layers.dropout(tf.tile(glo_embs, [1, self.maxlen, 1]), 
                                                      rate=args.dropout_rate, training=tf.convert_to_tensor(self.is_training)), 
                                    [2, 1, 1])

            outputs_tr = tile_loc_embs * gat_val_tr + tile_glo_embs * (1 - gat_val_tr)
            outputs_tr *= tf.tile(padding_mask, [2, 1, 1])
            outputs_tr = self.layer_normalization(outputs_tr)

            outputs_pos = outputs_tr[:tf.shape(self.input_id)[0]]
            outputs_neg = outputs_tr[tf.shape(self.input_id)[0]:]

            # for test ...
            gat_val_te = self.item_similarity_gating(tf.tile(glo_embs, [1, 101, 1]), 
                                                     tf.tile(tf.expand_dims(item_embs[:, -1, :], 1), [1, 101, 1]), 
                                                     candidate_embs, 
                                                     W_g, b_g, 
                                                     mode=args.gating_mode, 
                                                     inp_func=args.gating_input)
            gat_val_te = tf.nn.sigmoid(gat_val_te)

            outputs_te = tf.tile(tf.expand_dims(loc_embs[:, -1, :], 1), [1, 101, 1]) * gat_val_te + tf.tile(glo_embs, [1, 101, 1]) * (1 - gat_val_te)
            outputs_te = self.layer_normalization(outputs_te)

        else:
            if 'c' in args.ext_modules:
                with tf.variable_scope("consistency_aware_gating"):
                    glo_gat_in, _ = self.multihead_attention(item_embs, 
                                                             padding_mask=padding_mask, 
                                                             num_heads=args.num_heads, 
                                                             dropout_rate=args.dropout_rate, 
                                                             score='location', 
                                                             causality=False, #> for g_causality: True
                                                             get_att='lastq_multi',
                                                             residual=False)

                    loc_gat_in = self.centroid_coherence_func(item_embs)

                    gatings = tf.concat([loc_gat_in, tf.tile(glo_gat_in, [1, self.maxlen, 1])], -1)
                    # gatings = tf.concat([loc_gat_in, glo_gat_in], -1) #< for g_causality
                    gatings = tf.layers.dropout(gatings, rate=args.dropout_rate, training=tf.convert_to_tensor(self.is_training))

                    gatings = tf.nn.softmax(tf.layers.dense(gatings, 2, activation=None))

                    loc_gat_val = tf.expand_dims(gatings[:, :, 0], -1)
                    glo_gat_val = tf.expand_dims(gatings[:, :, 1], -1)
                    outputs = loc_embs * loc_gat_val + tf.tile(glo_embs, [1, self.maxlen, 1]) * glo_gat_val
                    # outputs = loc_embs * loc_gat_val + glo_embs * glo_gat_val #< for g_causality

            else:
                outputs = loc_embs + tf.tile(glo_embs, [1, self.maxlen, 1])
                # outputs = loc_embs + glo_embs #< for g_causality

            if 'g' in args.ext_modules: # dropout for the global representation
                outputs = tf.layers.dropout(outputs, rate=args.dropout_rate, training=tf.convert_to_tensor(self.is_training))
                outputs *= padding_mask
                outputs = self.layer_normalization(outputs)

            outputs_pos = outputs
            outputs_neg = outputs
            outputs_te = tf.expand_dims(outputs[:, -1, :], 1)

        # ==== training loss ====
        positive_rating = tf.reduce_sum(outputs_pos * positive_embs, -1)
        negative_rating = tf.reduce_sum(outputs_neg * negative_embs, -1)

        flag_exist = tf.to_float(tf.not_equal(self.pos_id, 0))
        self.loss = tf.reduce_sum(- tf.log(tf.sigmoid(positive_rating) + 1e-24) * flag_exist 
                                  - tf.log(1 - tf.sigmoid(negative_rating) + 1e-24) * flag_exist) / tf.reduce_sum(flag_exist)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_loss)

        # ==== test prediction ====
        self.cand_rating = tf.reduce_sum(outputs_te * candidate_embs, -1)

        # ==== optimizer ====
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


    # ================================================================
    # ======== multihead attention ========
    def multihead_attention(self, inputs, padding_mask=None, 
                            num_heads=2, dropout_rate=None, score='scaled_dot', 
                            causality=True, get_att='last_ave', residual=True):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        with tf.variable_scope("multihead_attention"):
            # linear projections, shape: [batch_size, seq_length, num_units]
            if score == 'location':
                weights = tf.get_variable('weights', 
                                     dtype = tf.float32, 
                                     shape = [1, self.num_units], 
                                     regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg))
                if causality == False: # obtain output shape: [batch_size, 1, num_units]
                    Q = tf.tile(tf.expand_dims(weights, 0), [tf.shape(inputs)[0], 1, 1])
                else: # obtain output shape: [batch_size, seq_length, num_units]
                    Q = tf.tile(tf.expand_dims(weights, 0), [tf.shape(inputs)[0], tf.shape(inputs)[1], 1])
            else:
                Q = tf.layers.dense(inputs, self.num_units, activation=None)

            K = tf.layers.dense(inputs, self.num_units, activation=None)
            V = tf.layers.dense(inputs, self.num_units, activation=None)

            # split and place in parallel, shape: [batch_size * num_heads, seq_length, num_units / num_heads]
            Qh = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            Kh = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            Vh = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            # score function, shape: [batch_size * num_heads, seq_length_Q, seq_length_K]
            if score == 'scaled_dot':
                outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1])) / (tf.to_float(tf.shape(Kh))[-1] ** 0.5)
            elif score == 'location' or score == 'dot':
                outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1]))

            # causality masking
            if causality == True:
                tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(outputs[0, :, :])).to_dense()
                # #v for tf.__version__='1.2.1'
                # tril = tf.contrib.linalg.LinearOperatorTriL(tf.ones_like(outputs[0, :, :])).to_dense()
                causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
                outputs = tf.where(tf.equal(causality_mask, 0), tf.ones_like(outputs)*(-2**32+1), outputs)

            # key masking
            if padding_mask is not None:
                key_mask = tf.transpose(tf.tile(padding_mask, [num_heads, 1, tf.shape(Q)[1]]), [0, 2, 1])
                outputs = tf.where(tf.equal(key_mask, 0), tf.ones_like(outputs)*(-2**32+1), outputs)

            # softmax normalization
            outputs = tf.nn.softmax(outputs)

            # query masking
            if (score != 'location') and (padding_mask is not None):
                query_mask = tf.tile(padding_mask, [num_heads, 1, tf.shape(inputs)[1]])
                outputs = outputs * query_mask

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

            # return attention for visualization
            if get_att == 'lastq_ave':
                attention = tf.reduce_mean(tf.split(outputs[:, -1], num_heads, axis=0), axis=0)
            if get_att == 'lastq_multi':
                attention = tf.split(outputs[:, -1], num_heads, axis=0)
            elif get_att == 'batch_multi':
                attention = outputs

            # weighted sum, shape: [batch_size * num_heads, seq_length_Q, num_units / num_heads]
            outputs = tf.matmul(outputs, Vh)

            # concatenate different heads, shape: [batch_size, seq_length_Q, num_units]
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        # residual connection
        if residual == True:
            outputs += inputs

        return outputs, attention

    # ======== feed forward ========
    def feed_forward(self, inputs, 
                     inner_units=None, dropout_rate=None):
        if inner_units is None:
            inner_units = self.num_units
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        with tf.variable_scope("feed_forward"):
            # inner layer
            params = {"inputs": inputs, "filters": inner_units, "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

            # readout layer
            params = {"inputs": outputs, "filters": self.num_units, "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        # residual connection
        outputs += inputs

        return outputs

    # ======== layer normalization ========
    def layer_normalization(self, inputs, epsilon=1e-8):
        with tf.variable_scope("layer_normalization"):
            alpha = tf.Variable(tf.ones(self.num_units))
            beta = tf.Variable(tf.zeros(self.num_units))

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = alpha * normalized + beta

        return outputs

    # ================================================================
    # ======== consistency of a list ========
    def centroid_coherence_func(self, inputs):
        #inputs = tf.layers.dense(inputs, num_units, activation=None)
        key_masks = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1)))
        key_masks = tf.tile(tf.expand_dims(key_masks, -1), [1, 1, tf.shape(inputs)[1]])

        b = tf.ones([tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[1]])
        tril = tf.linalg.LinearOperatorLowerTriangular(b).to_dense()
        # #v for tf.__version__='1.2.1'
        # tril = tf.contrib.linalg.LinearOperatorTriL(b).to_dense()
        tril_masked = tril*tf.transpose(key_masks, [0, 2, 1])

        outputs = tf.matmul(tril_masked, inputs)
        row_sum = tf.reduce_sum(tril_masked, 2, keep_dims=True) + 1e-10
        outputs_mean = outputs / row_sum
        res = tf.abs(inputs - outputs_mean)#inputs*outputs_mean#
        return res

    # ======== item similarity gating ========
    def item_similarity_gating(self, a_embs, b_embs, c_embs, W, b, 
                    mode='individual', inp_func='concat', dropout_rate=None):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        if inp_func == 'concat':
            inputs = tf.reshape(tf.concat([a_embs, b_embs, c_embs], -1), [-1, 3*self.num_units])
        elif inp_func == 'prod':
            inputs = tf.reshape(tf.concat([a_embs * c_embs, b_embs * c_embs], -1), [-1, 2*self.num_units])
        inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        logits = tf.matmul(inputs, W) + b
        logits = tf.layers.dropout(logits, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        if mode == 'individual':
            logits = tf.reshape(logits, [-1, tf.shape(b_embs)[1], 1])
        elif mode == 'feature':
            logits = tf.reshape(logits, [-1, tf.shape(b_embs)[1], self.num_units])

        weights = tf.nn.sigmoid(logits)

        return weights
