# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from generator import Generator
from discriminator import Discriminator
from rollout import RollOut
import pickle

total_epoch_num = 10
emb_dim = 32
latent_dim = 32
batch_size = 64

corpus, seq_len = pickle.load(open("models/corpus.pickle", "rb"))
w2i, i2w, vocab_num = pickle.load(open("models/vocabulary.pickle", "rb"))
print(seq_len, vocab_num)

generator = Generator(vocab_num=vocab_num, batch_size=batch_size, emb_dim=emb_dim, latent_dim=latent_dim,
                      seq_len=seq_len, start_token=w2i["__BOS"])
discriminator = Discriminator(vocab_num=vocab_num, seq_len=seq_len, emb_dim=emb_dim)
rollout = RollOut(generator=generator, update_rate=0.8)

corpus_len = len(corpus)
num_batch = len(corpus) // batch_size

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def create_dis_batch():
    rand_idx = np.random.randint(num_batch)
    pos_smaples = corpus[rand_idx*batch_size: (rand_idx+1)*batch_size]
    neg_samples = generator.generate(sess)
    pos_labels = [[0, 1] for _ in range(len(pos_smaples))]
    neg_labels = [[1, 0] for _ in range(len(neg_samples))]
    samples = np.concatenate([pos_smaples, neg_samples], 0)
    labels = np.concatenate([pos_labels, neg_labels], 0)
    shuffle_idx = np.random.permutation(range(len(samples)))[:batch_size]

    return samples[shuffle_idx], labels[shuffle_idx]


print("START PRE-TRAINING GENERATOR")
for epoch in range(5):
    losses = []
    for i in range(num_batch):
        train_corpus = corpus[i*batch_size: (i+1)*batch_size]
        _, loss = generator.train(sess, train_corpus)
        losses.append(loss)
    loss_mean = np.mean(losses)
    print("EPOCH: %s, LOSS: %s" % (epoch, loss_mean))


print("START PRE-TRAINING DISCRIMINATOR")
for epoch in range(50):
    x_batch, y_batch = create_dis_batch()
    feed_dict = {
        discriminator.x: x_batch,
        discriminator.y: y_batch,
        discriminator.dropout_keep_prob: 0.75
    }
    _ = sess.run(discriminator.train_op, feed_dict=feed_dict)


print("START ADVERSARIAL TRAINING")
for epoch in range(total_epoch_num):
    print(epoch)
    for e in range(3):
        samples = generator.generate(sess)
        rewards = rollout.get_reward(sess, samples, 16, discriminator)
        feed_dict = {
            generator.x: samples,
            generator.rewards: rewards
        }
        _ = sess.run(generator.gan_updates, feed_dict=feed_dict)

    rollout.update_params()

    for e in range(5):
        x_batch, y_batch = create_dis_batch()
        feed_dict = {
            discriminator.x: x_batch,
            discriminator.y: y_batch,
            discriminator.dropout_keep_prob: 0.75
        }
        _ = sess.run(discriminator.train_op, feed_dict=feed_dict)

