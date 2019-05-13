import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn import RNNEncoder


# # 通用RNN
# class RNNEncoder(nn.Module):
# 	def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=True, rnn_type='lstm'):
# 		super(RNNEncoder, self).__init__()
#
# 		self._RNNS = ['RNN', 'GRU', 'LSTM']
# 		self.rnn_type = rnn_type.upper()
# 		assert self.rnn_type in self._RNNS
#
# 		if num_layers == 1:
# 			dropout = 0.0
#
# 		self.hidden_size = hidden_size
# 		self.num_layers = num_layers
# 		self.num_directions = 2 if bidirectional else 1
#
# 		rnn = getattr(nn, self.rnn_type)  # 从torch.nn中获取对应的构造函数
#
# 		self.rnn = rnn(input_size=input_size,  # 输入的特征维度
# 						hidden_size=hidden_size,    # 隐层状态的特征维度
# 						num_layers=num_layers,      # LSTM 堆叠的层数，默认值是1层，如果设置为2，第二个LSTM接收第一个LSTM的计算结果
# 						dropout=dropout,            # 除了最后一层外，其它层的输出都会套上一个dropout层
# 						bidirectional=bidirectional,  # 是否为双向LSTM
# 						batch_first=True)  # [batch_size, seq, feature]
#
# 	def init_hidden(self, batch_size, retain=True):
# 		if retain:  # 是否保持每次隐层初始化值都相同
# 			torch.manual_seed(3347)
# 		h0 = torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_size)
# 		c0 = torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_size)
# 		return h0, c0
#
# 	def forward(self, inputs, batch_size, hidden=None):
# 		if hidden is None:
# 			hidden = self.init_hidden(batch_size)
# 			if self.rnn_type != 'LSTM':
# 				hidden = hidden[0]
#
# 		return self.rnn(inputs, hidden)


class POSTagger(nn.Module):
	def __init__(self, vocab, config, embedding_weights=None):
		super(POSTagger, self).__init__()

		self.config = config
		self.bidirectional = True
		self.embedding_size = embedding_weights.shape[1]
		self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights))

		self.rnn_encoder = RNNEncoder(
			input_size=self.embedding_size,
			hidden_size=self.config.hidden_size,
			num_layers=self.config.nb_layers,
			dropout=self.config.drop_rate,
			bidirectional=self.bidirectional,
			batch_first=True
		)

		self.hidden2pos = nn.Linear(config.hidden_size, vocab.pos_size)
		# self.hidden2pos = nn.Linear(num_directions * config.hidden_size, vocab.pos_size)
		self.embed_dropout = nn.Dropout(config.drop_embed_rate)
		self.linear_dropout = nn.Dropout(config.drop_rate)

	def forward(self, inputs, mask):
		# batch_size = inputs.size(0)
		# print(inputs.shape)  # (batch_size, seq_len)

		# print('data is in cuda: ', inputs.device, mask.device)

		embed = self.embedding(inputs)  # (batch_size, seq_len, embed_size)
		# print(embed.shape)

		if self.training:  # 预测时要置为False
			embed = self.embed_dropout(embed)

		rnn_out, hidden = self.rnn_encoder(embed, mask)  # (batch_size, seq_len, hidden_size)
		# print(rnn_out.shape)

		if self.bidirectional:
			rnn_out = rnn_out[:, :, :self.config.hidden_size] + rnn_out[:, :, self.config.hidden_size:]

		if self.training:
			rnn_out = self.linear_dropout(rnn_out)

		pos_space = self.hidden2pos(rnn_out)  # (batch_size, seq_len, pos_size)
		# print(pos_space.shape)
		pos_space = pos_space.reshape(-1, pos_space.size(-1))  # (batch_size * seq_len, pos_size)

		pos_score = F.log_softmax(pos_space, dim=1)  # (batch_size * seq_len, pos_size)
		# print(pos_score.shape)

		return pos_score
