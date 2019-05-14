import torch
import torch.nn as nn
import torch.nn.functional as F


class CharEncoder(nn.Module):
    def __init__(self, config, char_embedding_weights):
        super(CharEncoder, self).__init__()

        self.char_embedding_size = char_embedding_weights.shape[1]
        self.char_embedding = nn.Embedding.from_pretrained(torch.from_numpy(char_embedding_weights))
        self.char_embedding.weight.requires_grad = True

        # self.char_embedding = nn.Embedding(num_embeddings=vocab_size,
        #                                    embedding_dim=embedding_size)

        self.win_size = 3
        self.padding = 1

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.char_embedding_size,
                      out_channels=config.char_hidden_size,
                      padding=self.padding,  # 输入的每一条边补充0的层数
                      kernel_size=self.win_size),
            nn.ReLU(),
            # stride=1时，一维卷积输出大小(宽度) = 序列大小 + 2*pad - 窗口大小 + 1
            # nn.MaxPool1d(kernel_size=max_word_len + 2*self.padding - self.win_size + 1)
        )

        # self.win_sizes = [2, 3, 4]
        # self.convs = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv1d(
        #             in_channels=self.char_embedding_size,
        #             out_channels=config.char_hidden_size,
        #             padding=self.padding,
        #             kernel_size=w
        #         ),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool1d(kernel_size=max_len - w + 1)
        #     ) for w in self.win_sizes
        # ])

        self.dropout_embed = nn.Dropout(config.drop_embed_rate)

    def forward(self, chars):  # (batch_size, max_seq_len, max_wd_len)
        batch_size, max_seq_len, max_wd_len = chars.size()

        chars = chars.reshape((-1, max_wd_len))  # (batch_size * max_seq_len, max_wd_len)

        embed_x = self.char_embedding(chars)

        # batch_size * max_len * embedding_size -> batch_size * embedding_size * max_len
        embed_x = embed_x.permute(0, 2, 1)

        if self.training:
            embed_x = self.dropout_embed(embed_x)

        out = self.conv(embed_x)
        # out = [conv(embed_x) for conv in self.convs]
        # torch.cat(tuple(out), dim=1)  # 对应第二个维度（行）拼接起来，如 5*2*1,5*3*1的拼接变成5*5*1

        kernel_size = max_wd_len + 2*self.padding - self.win_size + 1
        out = F.max_pool1d(out, kernel_size)

        out = out.reshape(batch_size, max_seq_len, -1)

        return out
