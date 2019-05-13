import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from config.HyperConfig import Config
from TaggerModel_easy import POSTagger
from dataloader.DataLoader import create_vocab, batch_variable_mask_easy, load_data, get_batch


def save_model(model, save_path):
	# 保存整个模型
	torch.save(model, save_path)
	# 只保存模型参数
	# torch.save(model.state_dict(), save_path)


def draw(acc_lst, loss_lst):
	assert len(acc_lst) == len(loss_lst)
	epochs = len(acc_lst)
	plt.subplot(211)
	plt.plot(list(range(epochs)), loss_lst, c='r', label='loss')
	plt.legend()
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.subplot(212)
	plt.plot(list(range(epochs)), acc_lst, c='b', label='accuracy')
	plt.legend()
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.tight_layout()
	plt.show()


def calc_acc(pred, target):
	return torch.eq(torch.argmax(pred, dim=1), target).cpu().sum().item()


def calc_loss(pred, target):
	loss_func = nn.NLLLoss(ignore_index=-1)  # 忽略target中-1的值，且不会影响梯度
	loss = loss_func(pred, target)
	return loss


def evaluate(test_data, tagger, vocab, config):
	total_acc, total_loss = 0, 0
	nb_total = 0
	tagger.eval()
	for batch_data in get_batch(test_data, config.batch_size):
		xb, yb, seq_lens = batch_variable_mask_easy(batch_data, vocab)
		if config.use_cuda:
			xb = xb.cuda()
			yb = yb.cuda()
			seq_lens = seq_lens.cuda()

		pred = tagger(xb, seq_lens)
		loss = calc_loss(pred, yb)
		total_loss += loss.data.cpu().item()
		total_acc += calc_acc(pred, yb)
		nb_total += seq_lens.cpu().sum().item()

	total_acc = float(total_acc) / nb_total
	print('test: |loss: %f  acc: %f|' % (total_loss, total_acc))


def train(train_data, test_data, dev_data, vocab, config, embed_weights):
	tagger = POSTagger(vocab, config, embed_weights)
	optimizer = optim.Adam(tagger.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
	# optimizer = optim.Adam(filter(lambda p: p.requires_grad, tagger.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)

	if config.use_cuda:
		tagger = tagger.cuda()
	# print('model:', next(tagger.parameters()).is_cuda)

	tagger.train()
	total_acc, total_loss = [], []
	for i in range(config.epochs):
		print(' -- Epoch %d' % (i+1))
		epoch_acc, epoch_loss = 0, 0
		nb_total = 0
		t1 = time.time()
		for batch_data in get_batch(train_data, config.batch_size):
			# 1、准备数据
			xb, yb, seq_lens = batch_variable_mask_easy(batch_data, vocab)  # 处理batch_data

			if config.use_cuda:
				xb = xb.cuda()
				yb = yb.cuda()
				seq_lens = seq_lens.cuda()

			# 2、重置模型梯度为0
			tagger.zero_grad()

			# 3、将数据喂给模型
			pred = tagger(xb, seq_lens)

			# 4、计算损失值
			loss = calc_loss(pred, yb)
			epoch_acc += calc_acc(pred, yb)
			epoch_loss += loss.data.cpu().item()
			nb_total += seq_lens.cpu().sum().item()

			# 5、误差反向传播
			loss.backward()

			# 6、更新模型参数
			optimizer.step()

		t2 = time.time()
		print('time: %.3f min' % ((t2-t1)/60))
		epoch_acc = float(epoch_acc) / nb_total
		print('train: |loss: %f  acc: %f|' % (epoch_loss, epoch_acc))
		total_loss.append(epoch_loss)
		total_acc.append(epoch_acc)

		with torch.no_grad():
			dev_total_acc, dev_total_loss = 0, 0
			nb_total = 0
			tagger.eval()
			for batch_data in get_batch(dev_data, config.batch_size):
				xb, yb, seq_lens = batch_variable_mask_easy(batch_data, vocab)
				if config.use_cuda:
					xb = xb.cuda()
					yb = yb.cuda()
					seq_lens = seq_lens.cuda()

				pred = tagger(xb, seq_lens)
				loss = calc_loss(pred, yb)
				dev_total_loss += loss.data.cpu().item()
				dev_total_acc += calc_acc(pred, yb)
				nb_total += seq_lens.cpu().sum().item()

			dev_total_acc = float(dev_total_acc) / nb_total
			print('dev: |loss: %f  acc: %f|' % (dev_total_loss, dev_total_acc))

	# draw(total_acc, total_loss)

	save_model(tagger, config.save_model_path)

	evaluate(test_data, tagger, vocab, config)


if __name__ == '__main__':
	np.random.seed(1314)
	torch.manual_seed(3347)
	torch.cuda.manual_seed(3347)
	# torch.backends.cudnn.deterministic = True  # 解决reproducible问题，但是可能会影响性能
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.enabled = False  # cuDNN采用的是不确定性算法，会影响到reproducible

	print('GPU可用：', torch.cuda.is_available())
	print('CuDNN：',  torch.backends.cudnn.enabled)
	print('GPUs：', torch.cuda.device_count())

	print('training......')
	config = Config('config/hyper_param.cfg')
	config.use_cuda = torch.cuda.is_available()
	if config.use_cuda:
		torch.cuda.set_device(0)
	# print(torch.cuda.current_device())  # 当前的cuda设备序号

	train_data = load_data(config.train_data_path)
	test_data = load_data(config.test_data_path)
	dev_data = load_data(config.dev_data_path)
	print('train data size:', len(train_data))
	print('test data size:', len(test_data))
	print('dev data size:', len(dev_data))
	vocab = create_vocab(config.train_data_path)
	embed_weights = vocab.get_embedding_weights(config.embedding_path)
	vocab.save(config.save_vocab_path)

	train(train_data, test_data, dev_data, vocab, config, embed_weights)
