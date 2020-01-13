import tensorflow as tf
import os
from data import Data
import math
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class Model:
    def __init__(self, word2index, max_len, embedding_matrix=None):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.label = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.seqlen = tf.placeholder(dtype=tf.int32, shape=[None])
        self.drop = tf.placeholder(dtype=tf.float32)
        self.maxlen = max_len
        self.embedding_matrix = embedding_matrix
        self.word2index = word2index
        self.index2word = {word2index[x]: x for x in word2index}
        self.model_dic = './model/lstm_crf'
        self.pic_dic = './pic/lstm_crf'

        # embedding
        if self.embedding_matrix is not None:
            embedd_matrix = tf.get_variable('embedd_matrix', [len(word2index), 128],
                                            initializer=tf.constant_initializer(self.embedding_matrix))
        else:
            embedd_matrix = tf.get_variable('embedding', [len(word2index), 128],
                                            initializer=tf.random_normal_initializer)

        self.embedd = tf.nn.embedding_lookup(embedd_matrix, self.inputs)


        # Position Embedding
        self.position_encoding = np.array([
            [pos / math.pow(10000, 2.0 * (j // 2) / 128) for j in range(128)]
            for pos in range(self.maxlen)])
        # 偶数列使用sin，奇数列使用cos
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])

        # Embedding Concat
        self.Embedd_concat = tf.add(self.embedd, self.position_encoding)

        # BI-LSTM
        self.cell_fw = tf.contrib.rnn.BasicLSTMCell(256)
        self.cell_bw = tf.contrib.rnn.BasicLSTMCell(256)
        (self.lstm_out1, self.lstm_out2), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw,
                                                                              cell_bw=self.cell_bw,
                                                                              inputs=self.Embedd_concat,
                                                                              dtype=tf.float32,
                                                                              sequence_length=self.seqlen)

        # dropout
        self.lstm_out = tf.concat([self.lstm_out1, self.lstm_out2], axis=-1)
        self.lstm_out = tf.nn.dropout(self.lstm_out, keep_prob=self.drop)


        # Dense
        self.dense1 = tf.layers.dense(self.lstm_out, 256, activation=tf.nn.relu)

        # dropout
        self.dense1 = tf.nn.dropout(self.dense1, keep_prob=self.drop)

        # Dense2
        self.logits = tf.layers.dense(self.dense1, 5)

        # CRF

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.label,
                                                                                        self.seqlen)
        self.lossNER = tf.reduce_mean(-self.log_likelihood)
        self.predNers, self.viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.transition_params, self.seqlen)
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.lossNER)

    def train(self, x, y, epoch):
        saver = tf.train.Saver()
        x_train, x_dev, y_train, y_dev = train_test_split(x, y, random_state=42, test_size=0.2)
        x_dev, y_dev, seqlen = self.get_dev(x_dev, y_dev)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_dev = []
            loss_train = []
            loss_min = float('inf')
            n = 0
            es_step = 0
            for step in range(epoch):
                # each step
                loss_t = []
                begin = 0
                for x_batch, y_batch, seqlen_batch in self.get_batch(32, x_train, y_train):
                    _, loss, pre = sess.run([self.train_op, self.lossNER, self.predNers],
                                            feed_dict={self.inputs: x_batch,
                                                       self.label: y_batch,
                                                       self.seqlen: seqlen_batch,
                                                       self.drop: 0.7,
                                                       })
                    print('[{}/{}] loss===>{:.5f}'.format(begin, len(x_train), loss))
                    begin += len(x_batch)
                    loss_t.append(loss)
                loss_t = np.mean(loss_t)
                loss_train.append(loss_t)
                # dev
                loss_d = []
                for x_batch, y_batch, seqlen_batch in self.get_batch(32, x_dev, y_dev):
                    loss, pre = sess.run([self.lossNER, self.predNers],
                                         feed_dict={self.inputs: x_batch, self.label: y_batch,
                                                    self.seqlen: seqlen_batch, self.drop: 1.0})
                    loss_d.append(loss)
                loss_d = np.mean(loss_d)
                print('Epoch {}: loss===>{:.5f}'.format(step, loss_d))
                loss_dev.append(loss_d)
                if loss_d < loss_min:
                    # save model
                    if not os.path.exists(self.model_dic):
                        os.makedirs(self.model_dic)
                    saver.save(sess, os.path.join(self.model_dic, 'model'))
                    loss_min = loss_d
                else:
                    # count
                    if n > 3:
                        es_step = step
                        break
                    else:
                        n += 1
            print('Early Stop At Step {}'.format(es_step))
            plt.plot(loss_train)
            plt.plot(loss_dev)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['train, test'], loc='upper left')
            if not os.path.exists(self.pic_dic):
                os.makedirs(self.pic_dic)
            plt.savefig(os.path.join(self.pic_dic, 'loss.png'))
            plt.close()

    def get_batch(self, batch_size, x, y):
        """
        :param batch_size:
        :param x:
        :param y:
        :return:
        """
        if len(x) % batch_size == 0:
            step = len(x)//batch_size
        else:
            step = len(x)//batch_size + 1

        start = 0
        for i in range(step):
            end = start+batch_size
            x_batch = x[start: end]
            y_batch = y[start: end]
            seqlen_batch = [len(seq) for seq in x_batch]
            x_batch = pad_sequences(x_batch, maxlen=self.maxlen, padding='post')
            y_batch = pad_sequences(y_batch, maxlen=self.maxlen, padding='post')
            yield x_batch, y_batch, seqlen_batch
            start = end

    def get_dev(self, x, y):
        seqlen = [len(seq) for seq in x]
        x_dev = pad_sequences(x, maxlen=self.maxlen, padding='post')
        y_dev = pad_sequences(y, maxlen=self.maxlen, padding='post')
        return x_dev, y_dev, seqlen

    def evaluate(self, x, y):
        """
        输出验证集上各个标签的分类报告
        :param x:
        :param y:
        :return:
        """
        x_train, x_dev, y_train, y_dev = train_test_split(x, y, random_state=42, test_size=0.2)
        x_dev, y_dev, seqlen = self.get_dev(x_dev, y_dev)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.model_dic)
            saver.restore(sess, ckpt)
            result = []
            for x_batch, y_batch, seqlen_batch in self.get_batch(32, x_dev, y_dev):
                _, pre = sess.run([self.lossNER, self.predNers],
                                  feed_dict={self.inputs: x_batch, self.label: y_batch,
                                             self.seqlen: seqlen_batch, self.drop: 1.0})
                result.append(pre)
                print(pre.shape)
            result = np.concatenate(result, axis=0)
            print('result shape', result.shape)
            print('y_dev', y_dev.shape)
            pre = result
            pre_list = []
            for x in pre:
                for y in x:
                    pre_list.append(y)
            true = y_dev
            true_list = []
            for x in true:
                for y in x:
                    true_list.append(y)
            target_names = ['O', 'B_subject', 'I_subject', 'B_object', 'I_object']
            print(classification_report(true_list, pre_list, target_names=target_names))

    def print_dev(self, x, y):
        x_train, x_dev, y_train, y_dev = train_test_split(x, y, random_state=42, test_size=0.2)
        x_dev, y_dev, seqlen = self.get_dev(x_dev, y_dev)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.model_dic)
            saver.restore(sess, ckpt)
            result = []
            for x_batch, y_batch, seqlen_batch in self.get_batch(32, x_dev, y_dev):
                _, pre = sess.run([self.lossNER, self.predNers],
                                  feed_dict={self.inputs: x_batch, self.label: y_batch,
                                             self.seqlen: seqlen_batch, self.drop: 1.0,
                                             })
                print(pre.shape)
                result.append(pre)
            result = np.concatenate(result, axis=0)
            print('result shape', result.shape)
            print('y_dev', y_dev.shape)
            f1 = open('./result/dev.txt', 'w', encoding='utf-8')
            f2 = open('./result/pre.txt', 'w', encoding='utf-8')
            pre = result

            # print predict
            self.f_print(pre, x_dev, f2)
            self.f_print(y_dev, x_dev, f1)
            f1.close()
            f2.close()

    def f_print(self, data, x_dev, file):
        """
        将具体的预测结果和真实结果输出
        :param data:
        :param x_dev:
        :param file:
        :return:
        """
        for index_r in range(len(data)):
            index_c = 0
            obj = []
            sub = []
            while index_c < len(data[index_r]):
                if data[index_r][index_c] == 1:
                    temp_sub = [self.index2word[x_dev[index_r][index_c]]]
                    index_c += 1
                    while data[index_r][index_c] == 2:
                        temp_sub.append(self.index2word[x_dev[index_r][index_c]])
                        index_c += 1
                    sub.append(''.join(temp_sub))
                elif data[index_r][index_c] == 3:
                    temp_ob = [self.index2word[x_dev[index_r][index_c]]]
                    index_c += 1
                    while data[index_r][index_c] == 4:
                        temp_ob.append(self.index2word[x_dev[index_r][index_c]])
                        index_c += 1
                    obj.append(''.join(temp_ob))
                else:
                    index_c += 1
            file.write('=====' + str(index_r) + '=====' + '\n')
            file.write('subject:' + '\t'.join(sub) + '\n')
            file.write('object:' + '\t'.join(obj) + '\n')

