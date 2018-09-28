# -*- coding: utf-8 -*-

import tensorflow as tf

class Discriminator:
    def __init__(self, vocab_num, seq_len, emb_dim, window_size=4, filter_num=32):
        self.vocab_num = vocab_num
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.filter_num = filter_num

        self.x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len])
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, 2])
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32)

        with tf.variable_scope("discriminator"):

            self.embedding = tf.Variable(self.init_matrix([self.vocab_num, self.emb_dim]))

            emb_x = tf.nn.embedding_lookup(self.embedding, self.x)
            emb_x = tf.expand_dims(emb_x, -1)

            filter_shape = [window_size, self.emb_dim, 1, self.filter_num]
            filters = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.filter_num]))
            conv = tf.nn.conv2d(
                emb_x,
                filters,
                strides=[1, 1, 1, 1],
                padding="VALID",
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pool = tf.nn.max_pool(
                h,
                ksize=[1, self.seq_len - self.window_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            h_pool = tf.reshape(pool, shape=[-1, self.filter_num])
            h_drop = tf.nn.dropout(h_pool, keep_prob=self.dropout_keep_prob)

            W = tf.Variable(tf.truncated_normal([self.filter_num, 2], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[2]))

            self.output = tf.nn.softmax(tf.matmul(h_drop, W) + b)
            # pred = tf.argmax(output, 1)

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y)
            )
        params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(loss, params, aggregation_method=2)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)
