# from collections import Counter, defaultdict
import sys
sys.path.extend(['./', '../', '../../'])
import torch
import numpy as np
from dataloader.Vocab import POSVocab, CharVocab


# 一个Instance对应一行记录
class Instance(object):
	def __init__(self, words, pos):
		self.words = words  # 保存词序列
		self.pos = pos      # 保存词序列对应的词性序列

	def __str__(self):
		return ' '.join([wd+'_'+p for wd, p in zip(self.words, self.pos)])


# 加载数据集，数据封装成Instance实体
def load_data(corpus_path):
	insts = []
	with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as fin:
		for line in fin:
			tokens = line.strip().split()
			words, pos = [], []
			for token in tokens:
				words.append(token.split('_')[0])
				pos.append(token.split('_')[1])
			insts.append(Instance(words, pos))
	return insts


# 获取batch数据(思考：如何在后台异步加载数据prefetch?)
def get_batch(data, batch_size, shuffle=True):
	if shuffle:
		np.random.shuffle(data)

	num_batch = int(np.ceil(len(data) / float(batch_size)))
	for i in range(num_batch):
		batch_data = data[i*batch_size: (i+1)*batch_size]
		if shuffle:
			np.random.shuffle(batch_data)

		yield batch_data


# 创建词表
def create_vocab(corpus_path):
	# words_counter = Counter()
	# pos_counter = Counter()
	words_set = set()
	pos_set = set()
	with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as fin:
		for line in fin:
			tokens = line.strip().split()
			for token in tokens:
				wd, pos = token.split('_')
				# words_counter[wd] += 1
				# pos_counter[pos] += 1
				words_set.add(wd)
				pos_set.add(pos)

	return POSVocab(words_set, pos_set)
	# return POSVocab(words_counter, pos_counter)


def create_vocabs(corpus_path):
	words_set = set()
	char_set = set()
	pos_set = set()
	with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as fin:
		for line in fin:
			tokens = line.strip().split()
			for token in tokens:
				wd, pos = token.split('_')
				char_set.update([ch.strip() for ch in wd])
				words_set.add(wd)
				pos_set.add(pos)

	return CharVocab(char_set), POSVocab(words_set, pos_set)


def batch_variable(batch_data, vocab):
	batch_size = len(batch_data)
	max_len = max([len(inst.words) for inst in batch_data])

	wds_idxs = torch.zeros(batch_size, max_len, dtype=torch.long)
	pos_idxs = torch.zeros(batch_size, max_len, dtype=torch.long).fill_(-1)
	seq_lens = []
	for i, inst in enumerate(batch_data):
		seq_len = len(inst.words)
		seq_lens.append(seq_len)
		wds_idxs[i, :seq_len] = torch.LongTensor(vocab.word2index(inst.words))
		pos_idxs[i, :seq_len] = torch.LongTensor(vocab.pos2index(inst.pos))

	sorted_seq_lens, indices = torch.sort(torch.tensor(seq_lens), descending=True)
	_, unsorted_indices = torch.sort(indices)  # 排序前的顺序
	wds_idxs = torch.index_select(wds_idxs, dim=0, index=indices)
	pos_idxs = torch.index_select(pos_idxs, dim=0, index=indices)
	pos_idxs = pos_idxs.flatten()  # 展平成一维

	return wds_idxs, pos_idxs, sorted_seq_lens, unsorted_indices


# def batch_variable_mask(batch_data, vocab):
# 	batch_size = len(batch_data)
# 	max_len = max([len(inst.words) for inst in batch_data])
#
# 	wds_idxs = torch.zeros(batch_size, max_len, dtype=torch.long)
# 	pos_idxs = torch.zeros(batch_size, max_len, dtype=torch.long).fill_(-1)
# 	mask = torch.zeros(batch_size, max_len)
# 	seq_lens = []
# 	for i, inst in enumerate(batch_data):
# 		seq_len = len(inst.words)
# 		seq_lens.append(seq_len)
# 		wds_idxs[i, :seq_len] = torch.LongTensor(vocab.word2index(inst.words))
# 		pos_idxs[i, :seq_len] = torch.LongTensor(vocab.pos2index(inst.pos))
# 		mask[i, :seq_len] = torch.ones(seq_len)
# 	pos_idxs = pos_idxs.flatten()  # 展平成一维
#
# 	return wds_idxs, pos_idxs, mask, seq_lens


# def batch_variable_mask_easy(batch_data, vocab):
# 	batch_size = len(batch_data)
# 	max_len = max([len(inst.words) for inst in batch_data])
#
# 	wds_idxs = torch.zeros(batch_size, max_len, dtype=torch.long)
# 	pos_idxs = torch.zeros(batch_size, max_len, dtype=torch.long).fill_(-1)
# 	seq_lens = torch.zeros(batch_size, )
# 	for i, inst in enumerate(batch_data):
# 		seq_len = len(inst.words)
# 		seq_lens[i] = seq_len
# 		wds_idxs[i, :seq_len] = torch.LongTensor(vocab.word2index(inst.words))
# 		pos_idxs[i, :seq_len] = torch.LongTensor(vocab.pos2index(inst.pos))
# 	pos_idxs = pos_idxs.flatten()  # 展平成一维
#
# 	return wds_idxs, pos_idxs, seq_lens


def batch_variable_mask_easy(batch_data, vocab, char_vocab):
	batch_size = len(batch_data)

	max_seq_len, max_wd_len = 0, 0
	for inst in batch_data:
		if len(inst.words) > max_seq_len:
			max_seq_len = len(inst.words)
		for wd in inst.words:
			if len(wd) > max_wd_len:
				max_wd_len = len(wd)

	# max_wd_len = min(max_wd_len, 6)

	wds_idxs = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
	char_idxs = torch.zeros((batch_size, max_seq_len, max_wd_len), dtype=torch.long)
	pos_idxs = torch.zeros(batch_size, max_seq_len, dtype=torch.long).fill_(-1)
	seq_lens = torch.zeros(batch_size, )

	for i, inst in enumerate(batch_data):
		seq_len = len(inst.words)
		seq_lens[i] = seq_len
		for j, wd in enumerate(inst.words):
			char_idxs[i, j, :len(wd)] = torch.tensor(char_vocab.char2idx(wd), dtype=torch.long)
		wds_idxs[i, :seq_len] = torch.tensor(vocab.word2index(inst.words), dtype=torch.long)
		pos_idxs[i, :seq_len] = torch.tensor(vocab.pos2index(inst.pos), dtype=torch.long)

	pos_idxs = pos_idxs.flatten()  # 展平成一维

	return wds_idxs, char_idxs, pos_idxs, seq_lens
