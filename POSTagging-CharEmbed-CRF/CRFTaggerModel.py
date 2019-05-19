import torch
import torch.nn as nn
from rnn_easy import RNNEncoder
from char_encoder import CharEncoder
from torchCRF import CRF


class POSTagger(nn.Module):
	def __init__(self, vocab, config, wd_embedding_weights=None, char_embedding_weights=None):
		super(POSTagger, self).__init__()

		self.config = config
		self.bidirectional = True
		self.wd_embedding_size = wd_embedding_weights.shape[1]
		self.wd_embedding = nn.Embedding.from_pretrained(torch.from_numpy(wd_embedding_weights))  # 默认需要梯度
		self.wd_embedding.weight.requires_grad = False

		self.rnn_encoder = RNNEncoder(
			input_size=self.wd_embedding_size + self.config.char_hidden_size,
			hidden_size=self.config.hidden_size,
			num_layers=self.config.nb_layers,
			dropout=self.config.drop_rate,
			bidirectional=self.bidirectional,
			batch_first=True
		)

		self.char_encoder = CharEncoder(
			config=config,
			char_embedding_weights=char_embedding_weights
		)

		self.crf_layer = CRF(vocab.pos_size, batch_first=True)

		num_directions = 2 if self.bidirectional else 1
		# self.hidden2pos = nn.Linear(config.hidden_size, vocab.pos_size)
		self.hidden2pos = nn.Linear(num_directions * config.hidden_size, vocab.pos_size)
		self.embed_dropout = nn.Dropout(config.drop_embed_rate)
		self.linear_dropout = nn.Dropout(config.drop_rate)

	def _get_rnn_features(self, inputs, chars, mask):
		# inputs: (batch_size, seq_len)
		# chars: (batch_size*seq_len, wd_len)
		# mask: (batch_size, seq_len)

		char_representation = self.char_encoder(chars)
		wd_embed = self.wd_embedding(inputs)
		embed = torch.cat((char_representation, wd_embed), dim=2)

		if self.training:  # 预测时要置为False
			embed = self.embed_dropout(embed)

		rnn_out, hidden = self.rnn_encoder(embed, mask)  # (batch_size, seq_len, hidden_size)

		# if self.bidirectional:
		# 	rnn_out = rnn_out[:, :, :self.config.hidden_size] + rnn_out[:, :, self.config.hidden_size:]

		if self.training:
			rnn_out = self.linear_dropout(rnn_out)

		pos_space = self.hidden2pos(rnn_out)  # (batch_size, seq_len, pos_size)
		return pos_space

	def neg_log_likelihood(self, inputs, chars, mask, tags):
		# inputs: (batch_size, seq_len)
		# chars: (batch_size*seq_len, wd_len)
		# mask: (batch_size, seq_len)
		# tags: (batch_size, seq_len)

		emissions = self._get_rnn_features(inputs, chars, mask)
		# BiLSTM隐层的输出作为CRF层的输入，返回the negative log likelihood
		lld = self.crf_layer(emissions, tags, mask=mask)
		# the returned value is the log likelihood
		# so you’ll need to make this value negative as your loss.
		# By default, the log likelihood is summed over batches.
		return lld

	def forward(self, inputs, chars, mask):
		# inputs: (batch_size, seq_len)
		# chars: (batch_size*seq_len, wd_len)
		# mask: (batch_size, seq_len)

		emissions = self._get_rnn_features(inputs, chars, mask)
		best_tag_seq = self.crf_layer.decode(emissions, mask=None)
		# List of list containing the best tag sequence for each batch
		return torch.tensor(best_tag_seq, device=inputs.device)
