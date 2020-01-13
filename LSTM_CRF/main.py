from lstm_crf import Model
from data import Data
from w2v import Word2vec
import numpy as np
import gensim
import os

w2v_dir = './data/w2v'
if not os.path.exists(w2v_dir):
    os.makedirs(w2v_dir)

d = Data()
word2index, inputs, max_len = d.tokenize()

w2v = Word2vec(inputs, w2v_dir)
if not os.path.exists(os.path.join(w2v_dir, 'word2vec_model')):
    w2v.train_w2v()

embedding_matrix = w2v.get_embedding_matrix()

label_id = d.label_process()


model = Model(word2index, max_len, embedding_matrix=embedding_matrix)
index2word = {word2index[x]: x for x in word2index}

model.train(inputs, label_id, 20)
model.evaluate(inputs, label_id)
model.print_dev(inputs, label_id)

# Other
# x_train, x_dev, y_train, y_dev = train_test_split(inputs, label_id, random_state=42, test_size=0.2)
# f3 = open('./result/dev_ch.txt', 'w', encoding='utf-8')
# for x in x_dev:
#     temp = []
#     for y in x:
#         assert y in index2word, 'error'
#         temp.append(index2word[y])
#     f3.write('\t'.join(temp)+'\n')
