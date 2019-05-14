import os
import time
import logging
import multiprocessing
from gensim.models.word2vec import Word2Vec, LineSentence, PathLineSentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class word2vec_config(object):
    def __init__(self,
                 corpus_path=None,
                 model_save_path=None,
                 embedding_size=200,  # 词向量的维度
                 win_size=5,   # 在一个句子中，当前词和预测词的最大距离(词向量上下文最大距离)
                 min_count=5,  # 词频少于min_count次数的单词会被丢弃掉
                 sg=0,         # 训练算法：sg=0 使用cbow训练, sg=1 使用skip-gram 对低频词较为敏感
                 n_iter=10,    # 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
                 cpu_count=multiprocessing.cpu_count()  # 设置多线程训练模型，机器的核数越多，训练越快
                 ):

        self.__corpus_path = corpus_path
        self.__model_save_path = model_save_path
        self.__embedding_size = embedding_size
        self.__win_size = win_size
        self.__min_count = min_count
        self.__sg = sg
        self.__n_iter = n_iter
        self.__cpu_count = cpu_count

    @property
    def corpus_path(self):
        return self.__corpus_path

    @property
    def model_save_path(self):
        return self.__model_save_path

    @property
    def embedding_size(self):
        return self.__embedding_size

    @property
    def win_size(self):
        return self.__win_size

    @property
    def min_count(self):
        return self.__min_count

    @property
    def sg(self):
        return self.__sg

    @property
    def n_iter(self):
        return self.__n_iter

    @property
    def cpu_count(self):
        return self.__cpu_count


# PathLineSentences(input_dir): 会去指定目录依次读取语料数据文件，采用iterator方式加载训练数据到内存
def train_wd2vec(config):  # 训练word2vec词向量
    logging.info('开始训练词向量....')
    t1 = time.time()
    if os.path.isfile(config.corpus_path):
        sentences = LineSentence(config.corpus_path)
    else:
        sentences = PathLineSentences(config.corpus_path)

    word2vec_model = Word2Vec(sentences=sentences,
                              size=config.embedding_size,
                              window=config.win_size,
                              min_count=config.min_count,
                              sg=config.sg,
                              workers=config.cpu_count,
                              iter=config.n_iter)
    t2 = time.time()
    logging.info('词向量训练结束！总用时：{}min'.format((t2 - t1) / 60.0))

    word2vec_model.save(config.model_save_path)  # 保存词向量模型
    logging.info('词向量模型已保存......')


# 重新训练词向量模型(gensim的强大之处)
def retrain_wd2vec(sentences, config):
    logging.info('重新训练词向量....')
    t1 = time.time()
    word2vec_model = Word2Vec.load(config.model_save_path)
    word2vec_model.train(sentences, total_examples=100, epochs=5)
    # 例：model.train([["hello", "world"], ["are", "you", "ok", "?"]], total_examples=2, epochs=1)
    t2 = time.time()
    logging.info('词向量训练结束！总用时：{}min'.format((t2 - t1) / 60.0))

    word2vec_model.save(config.model_save_path)  # 保存词向量模型
    logging.info('词向量模型已保存......')


if __name__ == '__main__':
    config = word2vec_config(corpus_path='wd2vec_corpus/',
                             model_save_path='model/word2vec.model')

    train_wd2vec(config)
