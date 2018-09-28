# -*- coding: utf-8 -*-

import tensorflow as tf

class Generator:
    def __init__(self, vocab_num, batch_size, emb_dim, latent_dim, seq_len, start_token):
        self.vocab_num = vocab_num
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)

        self.params = []
        self.embedding = tf.Variable(self.init_matrix([self.vocab_num, self.emb_dim]))
        self.params.append(self.embedding)
        self.recurrent_unit = self.create_recurrent_unit(self.params)
        self.output_unit = self.create_output_unit(self.params)

        self.h0 = tf.zeros([batch_size, latent_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        self.gen_x = tf.TensorArray(dtype=tf.int32, size=self.seq_len, dynamic_size=False, infer_shape=True)

        def _recurrence(i, x, h_prev, gen_x):
            h = self.recurrent_unit(x, h_prev)
            o = self.output_unit(h)
            log_prob = tf.log(tf.nn.softmax(o))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            next_x = tf.nn.embedding_lookup(self.embedding, next_token)
            gen_x = gen_x.write(i, next_token)
            return i + 1, next_x, h, gen_x

        _, _, _, self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.seq_len,
            body=_recurrence,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.embedding, self.start_token), self.h0, self.gen_x
            )
        )
        self.gen_x = self.gen_x.stack()
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size * seq_len

        self.x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_len])
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_len])
        self.emb_x = tf.transpose(tf.nn.embedding_lookup(self.embedding, self.x), perm=[1, 0, 2]) # seq_len * batch_size * emb_dim

        ta_emb_x = tf.TensorArray(dtype=tf.float32, size=self.seq_len)
        ta_emb_x = ta_emb_x.unstack(self.emb_x)

        self.output = tf.TensorArray(dtype=tf.float32, size=self.seq_len, dynamic_size=False, infer_shape=True)

        def _pre_recurrence(i, x, h_prev, output):
            h = self.recurrent_unit(x, h_prev)
            o = self.output_unit(h)
            output = output.write(i, tf.nn.softmax(o))
            x_t = ta_emb_x.read(i)
            return i + 1, x_t, h, output

        _, _, _, self.output = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.seq_len,
            body=_pre_recurrence,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.embedding, self.start_token), self.h0, self.output
            )
        )

        self.output = tf.transpose(self.output.stack(), perm=[1, 0, 2])  # batch_size * seq_len * vocab_num

        self.train_loss = -tf.reduce_sum(
            tf.one_hot(tf.reshape(self.x, [-1]), self.vocab_num, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.output, [-1, self.vocab_num]), 1e-20, 1.0)
            )
        ) / (self.batch_size * self.seq_len)

        optimizer = tf.train.AdamOptimizer()

        grad, _ = tf.clip_by_global_norm(tf.gradients(self.train_loss, self.params), clip_norm=5.0)
        self.updates = optimizer.apply_gradients(zip(grad, self.params))

        self.loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.reshape(self.x, [-1]), self.vocab_num, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.output, [-1, self.vocab_num]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
        )

        gan_optimizer = tf.train.AdamOptimizer()
        grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.params), clip_norm=5.0)
        self.gan_updates = gan_optimizer.apply_gradients(zip(grad, self.params))

    def train(self, sess, x):
        outputs = sess.run([self.updates, self.train_loss], feed_dict={self.x: x})
        return outputs

    def generate(self, sess):
        outputs = sess.run(self.gen_x)
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.latent_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.latent_dim, self.latent_dim]))
        self.bi = tf.Variable(self.init_matrix([self.latent_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.latent_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.latent_dim, self.latent_dim]))
        self.bf = tf.Variable(self.init_matrix([self.latent_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.latent_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.latent_dim, self.latent_dim]))
        self.bog = tf.Variable(self.init_matrix([self.latent_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.latent_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.latent_dim, self.latent_dim]))
        self.bc = tf.Variable(self.init_matrix([self.latent_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.latent_dim, self.vocab_num]))
        self.bo = tf.Variable(self.init_matrix([self.vocab_num]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit
