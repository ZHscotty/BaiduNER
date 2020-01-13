import os
import json
from collections import Counter


class Data:
    def __init__(self):
        self.train_dir = './原始数据'
        self.stopword = self.load_stopword()

    def load_train(self):
        """
        从提供的json格式的数据中，提取训练数据
        通过先分词-BIO标注-分字的方式处理数据（分词后的BIO较直接对字进行BIO标注会有些许误差）
        :return:
        """
        train_path = os.path.join(self.train_dir, 'train_data.json')
        f = open(train_path, 'r', encoding='utf-8')
        if not os.path.exists('./data'):
            os.makedirs('./data')

        f_text = open('./data/text.txt', 'w', encoding='utf-8')
        f_text_token = open('./data/text_token.txt', 'w', encoding='utf-8')
        f_text_bio = open('./data/text_bio.txt', 'w', encoding='utf-8')

        for i in f.readlines():
            setence = []
            tags = []
            tokens = []

            j = json.loads(i)
            Object = [o['object'] for o in j['spo_list']]
            Subject = [s['subject'] for s in j['spo_list']]

            # 过滤停用词
            for x in j['postag']:
                if x['word'] not in self.stopword:
                    setence.append(x['word'])

            for s in setence:
                tokens.extend(s)

            # 通过分词结果找主客体---有点问题
            for s in setence:
                if s in Object:
                    temp = ['I_object' for _ in range(len(s))]
                    temp[0] = 'B_object'
                elif s in Subject:
                    temp = ['I_subject' for _ in range(len(s))]
                    temp[0] = 'B_subject'
                else:
                    temp = ['O' for _ in range(len(s))]
                tags.extend(temp)

            # 过滤有问题的数据
            if 'B_object' not in tags or 'B_subject' not in tags:
                pass
            else:
                f_text.write(j['text'] + '\n')
                f_text_token.write('\t'.join(tokens) + '\n')
                f_text_bio.write('\t'.join(tags)+'\n')

        f_text.close()
        f_text_token.close()
        f_text_bio.close()

    def load_schemas(self):
        data_path = os.path.join(self.train_dir, 'all_50_schemas')
        f = open(data_path, 'r', encoding='utf-8')
        lines = f.readlines()
        object_list = []
        subject_list = []
        for l in lines:
            t = json.loads(l)
            object_list.append(t['object_type'])
            subject_list.append(t['subject_type'])
        return list(set(object_list+subject_list))

    def load_stopword(self):
        f = open('哈工大停用词表.txt', 'r', encoding='utf-8')
        stopword = []
        for line in f.readlines():
            stopword.append(line.strip())
        return stopword

    def check(self):
        f = open('./data/text_bio.txt', 'r', encoding='utf-8')
        index = 1
        error = []
        for x in f.readlines():
            bio = x.strip().split('\t')
            if 'B_object' not in bio or 'B_subject' not in bio:
                error.append(index)
            index += 1
        print(index)
        return error

    def tokenize(self):
        """
        按字进行划分
        :return:
        """
        f = open('./data/text_token.txt', 'r', encoding='utf-8')
        texts = [l.strip().split('\t') for l in f.readlines()]
        all = []
        for x in texts:
            for y in x:
                all.append(y)
        ch_pre = Counter(all)
        ch = [x for x in ch_pre if ch_pre[x] > 3]
        word2index = {}
        for index, x in enumerate(ch):
            word2index[x] = index
        word2index = {x: word2index[x]+2 for x in word2index}
        word2index['PAD'] = 0
        word2index['UNK'] = 1
        texts_id = []
        max_len = 0
        for x in texts:
            temp = []
            max_len = max(max_len, len(x))
            for ch in x:
                if ch in word2index:
                    temp.append(word2index[ch])
                else:
                    temp.append(word2index['UNK'])
            texts_id.append(temp)
        f.close()
        return word2index, texts_id, max_len

    def label_process(self):
        f_bio = open('./data/text_bio.txt', 'r', encoding='utf-8')
        label_bio = [x.strip().split('\t') for x in f_bio.readlines()]
        label2id = {'O': 0, 'B_subject': 1, 'I_subject': 2, 'B_object': 3, 'I_object': 4}

        label_id = []
        for x in label_bio:
            temp = []
            for y in x:
                temp.append(label2id[y])
            label_id.append(temp)

        return label_id
