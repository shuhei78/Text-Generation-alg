# -*- coding: utf-8 -*-

import MeCab
import collections
import tensorflow as tf
import pickle

def tokenize(sentence,
             part_of_speech=["名詞", "動詞", "形容詞", "副詞", "接続詞", "助動詞", "助詞", "感動詞", "記号", "フィラー", "その他"]):
    mecab = MeCab.Tagger("-d data/mecab-ipadic-neologd")
    mecab.parse('')  # Avoiding UnicodeDecodeError
    node = mecab.parseToNode(sentence)
    token_list = []

    while node:
        feats = node.feature.split(',')
        if feats[0] in part_of_speech:
            token_list.append(node.surface)
        node = node.next
    return token_list


file_name = "./data/copy.txt"
counter = collections.Counter()
w2i = {"__BOS": 0, "__EOS": 2, "__UNK": 1}
texts = []
with open(file_name) as f_in:
    for line in f_in:
        text = line.strip()
        words = tokenize(text)
        for word in words:
            counter[word] += 1
        texts.append(words)

for word in counter.most_common():
    if word[1] > 10:
        w_id = len(w2i)
        w2i[word[0]] = w_id
i2w = {v: k for k, v in w2i.items()}

max_seq_len = 0
corpus = []
for words in texts:
    word_ids = [w2i[word] if word in w2i else w2i["__UNK"] for word in words]
    max_seq_len = max(len(word_ids), max_seq_len)
    corpus.append(word_ids)

corpus = tf.keras.preprocessing.sequence.pad_sequences(corpus, maxlen=max_seq_len, value=w2i["__EOS"], padding="post")

with open("models/corpus.pickle", "wb") as f_out:
    pickle.dump((corpus, max_seq_len), f_out)

with open("models/vocabulary.pickle", "wb") as f_out:
    pickle.dump((w2i, i2w, len(w2i)), f_out)
