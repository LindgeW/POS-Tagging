import pickle
import numpy as np


# 词性标注 - 字符表
class CharVocab(object):
	def __init__(self, char_set):
		super(CharVocab, self).__init__()
		self._UNK = 0
		self._char2idx = None
		self._idx2char = None
		# self._char2idx = {char: idx+1 for idx, char in enumerate(char_set)}
		# self._char2idx['un'] = self._UNK
		# self._idx2char = {idx: char for char, idx in self._char2idx.items()}

	def get_embedding_weights(self, embed_path):
		# 保存每个词的词向量
		ch2vec_tab = {}
		vector_size = 0
		with open(embed_path, 'r', encoding='utf-8', errors='ignore') as fin:
			for line in fin:
				tokens = line.split()
				vector_size = len(tokens) - 1
				ch2vec_tab[tokens[0]] = list(map(lambda x: float(x), tokens[1:]))

		self._char2idx = {ch: idx + 1 for idx, ch in enumerate(ch2vec_tab.keys())}  # 词索引字典 {词: 索引}，索引从1开始计数
		self._char2idx['<unk>'] = self._UNK
		self._idx2char = {idx: ch for ch, idx in self._char2idx.items()}

		vocab_size = len(self._char2idx)  # 词典大小(索引数字的个数)
		embedding_weights = np.zeros((vocab_size, vector_size), dtype='float32')  # vocab_size * EMBEDDING_SIZE的0矩阵
		for idx, wd in self._idx2char.items():  # 从索引为1的词语开始，用词向量填充矩阵
			if idx != self._UNK:
				embedding_weights[idx] = ch2vec_tab[wd]
				# embedding_weights[self._UNK] += ch2vec_tab[wd]

		# 对于OOV的词赋值为0 或 随机初始化 或 赋其他词向量的均值
		# embedding_weights[self._UNK, :] = np.random.uniform(-0.25, 0.25, vector_size)
		embedding_weights[self._UNK] = np.random.uniform(-0.25, 0.25, vector_size)
		# embedding_weights[self._UNK] = embedding_weights[self._UNK] / vocab_size
		embedding_weights = embedding_weights / np.std(embedding_weights)  # 归一化
		return embedding_weights

	def save(self, save_path):
		with open(save_path, 'wb') as fw:
			pickle.dump(self, fw)

	def char2idx(self, chars):  # 可 爱
		if len(chars) > 1:
			return [self._char2idx.get(c, self._UNK) for c in chars]
		else:
			return self._char2idx.get(chars, self._UNK)

	def idx2char(self, idxs):
		if isinstance(idxs, list):
			return [self._idx2char[i] for i in idxs]
		else:
			return self._idx2char[idxs]

	@property
	def vocab_size(self):
		return len(self._char2idx)


# 词性标注 - 词表
class POSVocab(object):
	def __init__(self, words_set, pos_set):
		super(POSVocab, self).__init__()

		self._UNK = 0
		self._wd2idx = None
		self._idx2wd = None

		# self._wd2idx = {wd: idx+1 for idx, wd in enumerate(words_set)}
		# self._wd2idx['<unk>'] = self.UNK
		# self._idx2wd = {idx: wd for wd, idx in self._wd2idx.items()}

		self._pos2idx = {pos: idx for idx, pos in enumerate(pos_set)}
		# self._pos2idx['un'] = self.UNK
		self._idx2pos = {idx: pos for pos, idx in self._pos2idx.items()}
		print('词性数量：', len(self._pos2idx))

	def get_embedding_weights(self, embed_path):
		# 保存每个词的词向量
		wd2vec_tab = {}
		vector_size = 0
		with open(embed_path, 'r', encoding='utf-8', errors='ignore') as fin:
			for line in fin:
				tokens = line.split()
				vector_size = len(tokens) - 1
				wd2vec_tab[tokens[0]] = list(map(lambda x: float(x), tokens[1:]))

		self._wd2idx = {wd: idx + 1 for idx, wd in enumerate(wd2vec_tab.keys())}  # 词索引字典 {词: 索引}，索引从1开始计数
		self._wd2idx['<unk>'] = self._UNK
		self._idx2wd = {idx: wd for wd, idx in self._wd2idx.items()}

		vocab_size = len(self._wd2idx)  # 词典大小(索引数字的个数)
		embedding_weights = np.zeros((vocab_size, vector_size), dtype='float32')  # vocab_size * EMBEDDING_SIZE的0矩阵
		for idx, wd in self._idx2wd.items():  # 从索引为1的词语开始，用词向量填充矩阵
			if idx != self._UNK:
				embedding_weights[idx] = wd2vec_tab[wd]
				# embedding_weights[self._UNK] += wd2vec_tab[wd]

		# 对于OOV的词赋值为0 或 随机初始化 或 赋其他词向量的均值
		# embedding_weights[self._UNK, :] = np.random.uniform(-0.25, 0.25, vector_size)
		embedding_weights[self._UNK] = np.random.uniform(-0.25, 0.25, vector_size)
		# embedding_weights[self._UNK] = embedding_weights[self._UNK] / vocab_size
		embedding_weights = embedding_weights / np.std(embedding_weights)  # 归一化
		return embedding_weights

	def word2index(self, ws):
		if isinstance(ws, list):
			return [self._wd2idx.get(w, self._UNK) for w in ws]
		else:
			return self._wd2idx.get(ws, self._UNK)

	def index2word(self, idxs):
		if isinstance(idxs, list):
			return [self._idx2wd[i] for i in idxs]
		else:
			return self._idx2wd[idxs]

	def pos2index(self, pos):
		if isinstance(pos, list):
			return [self._pos2idx[p] for p in pos]
		else:
			return self._pos2idx[pos]

	def index2pos(self, idxs):
		if isinstance(idxs, list):
			return [self._idx2pos[i] for i in idxs]
		else:
			return self._idx2pos[idxs]

	def save(self, save_vocab_path):
		with open(save_vocab_path, 'wb') as fw:
			pickle.dump(self, fw)

	@property
	def vocab_size(self):
		return len(self._wd2idx)

	@property
	def pos_size(self):
		return len(self._pos2idx)
