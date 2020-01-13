import gensim
import logging
import os
import numpy as np

class Word2Vec:
    def __init__(self, inputs, w2v_dir):
        self.inputs = inputs
        self.w2v_dir = w2v_dir

    def train_w2v(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        inputs_str = [[str(y) for y in x]for x in self.inputs]
        w2v = gensim.models.word2vec.Word2Vec(sentences=inputs_str, size=128, min_count=-1)
        w2v.save(os.path.join(self.w2v_dir, 'word2vec_model'))
        w2v.wv.save_word2vec_format(os.path.join(self.w2v_dir, 'w2v'), binary=False)

    def get_embedding_matrix(self):
        model_embedd = gensim.models.word2vec.Word2Vec.load(os.path.join(self.w2v_dir, 'word2vec_model'))
        matrix = model_embedd.wv.load_word2vec_format(os.path.join(self.w2v_dir, 'w2v')).vectors
        pad = np.random.uniform(-0.05, 0.05, size=(1, 128))
        embedding_matrix = np.concatenate([pad, matrix], axis=0)
        return embedding_matrix