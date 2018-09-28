# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class RollOut:
    def __init__(self, generator, update_rate):
        self.generator = generator
        self.update_rate = update_rate

        self.vocab_num = generator.vocab_num
        self.batch_size = generator.batch_size
        self.seq_len = generator.seq_len
        self.emb_dim = generator.emb_dim
        self.latent_dim = generator.latent_dim
        self.start_token = generator.start_token
        self.embedding = tf.identity(generator.embedding)

        self.recurrent_unit = self.create_recurrent_unit()
        self.output_unit = self.create_output_unit()

        self.h0 = tf.zeros([self.batch_size, self.latent_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len])
        self.emb_x = tf.transpose(tf.nn.embedding_lookup(self.embedding, self.x), perm=[1, 0, 2]) # seq_len * batch_size * emb_dim

        ta_emb_x = tf.TensorArray(dtype=tf.float32, size=self.seq_len)
        ta_emb_x = ta_emb_x.unstack(self.emb_x)

        ta_x = tf.TensorArray(dtype=tf.int32, size=self.seq_len)
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))

        self.gen_x = tf.TensorArray(dtype=tf.int32, size=self.seq_len, dynamic_size=False, infer_shape=True)
        self.given_num = tf.placeholder(tf.int32)

        def _recurrence_given(i, x, h_prev, gen_x):
            h = self.recurrent_unit(x, h_prev)
            next_x = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, next_x, h, gen_x

        def _recurrence_rollout(i, x, h_prev, gen_x):
            h = self.recurrent_unit(x, h_prev)
            o = self.output_unit(h)
            log_prob = tf.log(tf.nn.softmax(o))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            next_x = tf.nn.embedding_lookup(self.embedding, next_token)
            gen_x = gen_x.write(i, next_token)
            return i + 1, next_x, h, gen_x

        i, x, h, self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.given_num,
            body=_recurrence_given,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.embedding, self.start_token), self.h0, self.gen_x
            )
        )
        _, _, _, self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.seq_len,
            body=_recurrence_rollout,
            loop_vars=(
                i, x, h, self.gen_x
            )
        )
        self.gen_x = self.gen_x.stack()
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])

    def get_reward(self, sess, input_x, rollout_num, discriminator):
        rewards = []
        for i in range(rollout_num):
            for pos in range(1, self.seq_len):
                feed_dict = {self.x: input_x, self.given_num: pos}
                sampled_x = sess.run(self.gen_x, feed_dict=feed_dict)
                feed_dict = {discriminator.x: sampled_x, discriminator.dropout_keep_prob: 1.0}
                output = sess.run(discriminator.output, feed_dict=feed_dict)
                pred_real = np.array([pred[0] for pred in output])
                if i == 0:
                    rewards.append(pred_real)
                else:
                    rewards[pos - 1] += pred_real

            feed_dict = {discriminator.x: input_x, discriminator.dropout_keep_prob: 1.0}
            output = sess.run(discriminator.output, feed_dict=feed_dict)
            pred_real = np.array([pred[0] for pred in output])
            if i == 0:
                rewards.append(pred_real)
            else:
                rewards[self.seq_len - 1] += pred_real
        rewards = np.transpose(rewards) / float(rollout_num)  # batch_size * seq_len

        return rewards

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.identity(self.generator.Wi)
        self.Ui = tf.identity(self.generator.Ui)
        self.bi = tf.identity(self.generator.bi)

        self.Wf = tf.identity(self.generator.Wf)
        self.Uf = tf.identity(self.generator.Uf)
        self.bf = tf.identity(self.generator.bf)

        self.Wog = tf.identity(self.generator.Wog)
        self.Uog = tf.identity(self.generator.Uog)
        self.bog = tf.identity(self.generator.bog)

        self.Wc = tf.identity(self.generator.Wc)
        self.Uc = tf.identity(self.generator.Uc)
        self.bc = tf.identity(self.generator.bc)

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

    def update_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = self.update_rate * self.Wi + (1 - self.update_rate) * tf.identity(self.generator.Wi)
        self.Ui = self.update_rate * self.Ui + (1 - self.update_rate) * tf.identity(self.generator.Ui)
        self.bi = self.update_rate * self.bi + (1 - self.update_rate) * tf.identity(self.generator.bi)

        self.Wf = self.update_rate * self.Wf + (1 - self.update_rate) * tf.identity(self.generator.Wf)
        self.Uf = self.update_rate * self.Uf + (1 - self.update_rate) * tf.identity(self.generator.Uf)
        self.bf = self.update_rate * self.bf + (1 - self.update_rate) * tf.identity(self.generator.bf)

        self.Wog = self.update_rate * self.Wog + (1 - self.update_rate) * tf.identity(self.generator.Wog)
        self.Uog = self.update_rate * self.Uog + (1 - self.update_rate) * tf.identity(self.generator.Uog)
        self.bog = self.update_rate * self.bog + (1 - self.update_rate) * tf.identity(self.generator.bog)

        self.Wc = self.update_rate * self.Wc + (1 - self.update_rate) * tf.identity(self.generator.Wc)
        self.Uc = self.update_rate * self.Uc + (1 - self.update_rate) * tf.identity(self.generator.Uc)
        self.bc = self.update_rate * self.bc + (1 - self.update_rate) * tf.identity(self.generator.bc)

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

    def create_output_unit(self):
        self.Wo = tf.identity(self.generator.Wo)
        self.bo = tf.identity(self.generator.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x latent_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_output_unit(self):
        self.Wo = self.update_rate * self.Wo + (1 - self.update_rate) * tf.identity(self.generator.Wo)
        self.bo = self.update_rate * self.bo + (1 - self.update_rate) * tf.identity(self.generator.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x latent_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_params(self):
        self.embedding = tf.identity(self.generator.embedding)
        self.recurrent_unit = self.update_recurrent_unit()
        self.output_unit = self.update_output_unit()
