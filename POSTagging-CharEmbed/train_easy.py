import numpy as np
import time
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from config.HyperConfig import Config
from TaggerModel_easy import POSTagger
from dataloader.DataLoader import create_vocabs, batch_variable_mask_easy, load_data, get_batch


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
    # loss = F.nll_loss(pred, target, ignore_index=-1)
    return loss


def evaluate(test_data, tagger, vocab, char_vocab, config):
    total_acc, total_loss = 0, 0
    nb_total = 0
    tagger.eval()
    for batch_data in get_batch(test_data, config.batch_size):
        xb, xch, yb, seq_lens = batch_variable_mask_easy(batch_data, vocab, char_vocab)
        if config.use_cuda:
            xb = xb.cuda()
            xch = xch.cuda()
            yb = yb.cuda()
            seq_lens = seq_lens.cuda()

        pred = tagger(xb, xch, seq_lens)
        loss = calc_loss(pred, yb)
        total_loss += loss.data.cpu().item()
        total_acc += calc_acc(pred, yb)
        nb_total += seq_lens.cpu().sum().item()

    total_acc = float(total_acc) / nb_total
    print('test: |loss: %f  acc: %f|' % (total_loss, total_acc))


def train(train_data, test_data, dev_data, vocab, config, wd_embed_weights, char_embed_weights):
    tagger = POSTagger(vocab, config, wd_embed_weights, char_embed_weights)
    # optimizer = optim.Adam(tagger.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, tagger.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)

    if config.use_cuda:
        tagger = tagger.cuda()
    # print('model:', next(tagger.parameters()).is_cuda)

    total_acc, total_loss = [], []
    val_total_acc, val_total_loss = [], []
    for ep in range(config.epochs):
        tagger.train()
        print(' -- Epoch %d' % (ep+1))
        epoch_acc, epoch_loss = 0, 0
        nb_total = 0
        t1 = time.time()
        for batch_data in get_batch(train_data, config.batch_size):
            # 1、准备数据
            xb, xch, yb, seq_lens = batch_variable_mask_easy(batch_data, vocab, char_vocab)  # 处理batch_data

            if config.use_cuda:
                xb = xb.cuda()
                xch = xch.cuda()
                yb = yb.cuda()
                seq_lens = seq_lens.cuda()

            # 2、重置模型梯度为0
            tagger.zero_grad()

            # 3、将数据喂给模型
            pred = tagger(xb, xch, seq_lens)

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
        print('acc:', total_acc)
        print('loss:', total_loss)

        with torch.no_grad():
            dev_total_acc, dev_total_loss = 0, 0
            nb_total = 0
            tagger.eval()
            for batch_data in get_batch(dev_data, config.batch_size):
                xb, xch, yb, seq_lens = batch_variable_mask_easy(batch_data, vocab, char_vocab)
                if config.use_cuda:
                    xb = xb.cuda()
                    xch = xch.cuda()
                    yb = yb.cuda()
                    seq_lens = seq_lens.cuda()

                pred = tagger(xb, xch, seq_lens)
                loss = calc_loss(pred, yb)
                dev_total_loss += loss.data.cpu().item()
                dev_total_acc += calc_acc(pred, yb)
                nb_total += seq_lens.cpu().sum().item()

            dev_total_acc = float(dev_total_acc) / nb_total
            print('dev: |loss: %f  acc: %f|' % (dev_total_loss, dev_total_acc))
            val_total_acc.append(dev_total_acc)
            val_total_loss.append(dev_total_loss)

        # EarlyStopping
        # if (ep+1) % 2 == 0 and dev_total_loss > min(val_total_loss):
        #     print('误差开始，执行早停！！！')
        #     save_model(tagger, config.save_model_path)
            # break

        # ModelCheckPoint: save_best_only
        # if (ep+1) % 2 == 0 and dev_total_acc < max(val_total_acc):
        #     save_model(tagger, 'model-%02d-%.3f.pkl' % (ep, dev_total_acc))

    print('acc:', total_acc, 'loss:', total_loss)
    print('val_acc:', val_total_acc, 'val_loss:', val_total_loss)

    # draw(total_acc, total_loss)

    save_model(tagger, config.save_model_path)

    evaluate(test_data, tagger, vocab, char_vocab, config)


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
    # wd_vocab = create_vocab(config.train_data_path)
    char_vocab, wd_vocab = create_vocabs(config.train_data_path)

    char_embed_weights = char_vocab.get_embedding_weights(config.char_embedding_path)
    wd_embed_weights = wd_vocab.get_embedding_weights(config.word_embedding_path)

    char_vocab.save(config.save_char_vocab_path)
    wd_vocab.save(config.save_vocab_path)

    print(wd_vocab.vocab_size, char_vocab.vocab_size)
    train(train_data, test_data, dev_data, wd_vocab, config, wd_embed_weights, char_embed_weights)
