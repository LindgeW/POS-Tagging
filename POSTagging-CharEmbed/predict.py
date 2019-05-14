import os
import torch
import pickle
from dataloader.DataLoader import pred_data_variable
from config.HyperConfig import Config
from dataloader.DataLoader import Instance


def load_model(model_path):
    assert os.path.exists(model_path) and os.path.isfile(model_path)
    # GPU上训练的模型在CPU上运行
    model = torch.load(model_path, map_location='cpu')  # Load all tensors onto the CPU
    # model = torch.load(model_path, map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU, using a function
    # model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(1))  # Load all tensors onto GPU 1
    # model = torch.load(model_path, map_location={'cuda:1': 'cuda:0'})  # Map tensors from GPU 1 to GPU 0

    return model


def load_vocab(vocab_path):
    assert os.path.exists(vocab_path) and os.path.isfile(vocab_path)
    with open(vocab_path, 'rb') as fin:
        vocab = pickle.load(fin)
    return vocab


# 预测数据不含标签，标签值置成 -1
def predict(pred_data, tagger, vocab, char_vocab):
    tagger.eval()
    xb, xch, seq_lens = pred_data_variable(pred_data, vocab, char_vocab)
    pred = tagger(xb, xch, seq_lens)  # batch_size * tag_size
    pred_ids = torch.argmax(pred, dim=1)  # (batch_size, )
    return vocab.index2pos(pred_ids.tolist())


if __name__ == '__main__':
    config = Config('config/hyper_param.cfg')
    vocab = load_vocab(config.load_vocab_path)
    char_vocab = load_vocab(config.load_char_vocab_path)
    tagger = load_model(config.load_model_path)

    words = ['我', '的', '名字', '叫', '张三', '！']
    data = [Instance(words, None)]
    pos = predict(data, tagger, vocab, char_vocab)
    print(pos)
