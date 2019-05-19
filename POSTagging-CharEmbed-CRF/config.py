import argparse


def arg_parse():
    arg_parser = argparse.ArgumentParser(description='Parameters Configuration:')
    # 命令行接受参数
    arg_parser.add_argument('--cuda', type=int, default=-1, help='cuda id')
    arg_parser.add_argument('--nb_layers', type=int, default=1, help='the number of hidden layer')
    arg_parser.add_argument('--epochs', type=int, default=20, help='the number of iter')
    arg_parser.add_argument('--hidden_size', type=int, default=128, help='the dimension of hidden layer')
    arg_parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    arg_parser.add_argument('--drop_rate', type=float, default=0.3, help='the dropout rate of dense layers')
    arg_parser.add_argument('--drop_embed_rate', type=float, default=0.3, help='the dropout rate of embedding layers')
    arg_parser.add_argument('--learning_rate', type=float, default=1e-3, help='the learning rate of network')
    arg_parser.add_argument('--weight_decay', type=float, default=1e-7, help='the weight decay rate of network')
    arg_parser.add_argument('--extra', choices=['aa', 'bb'])
    args = arg_parser.parse_args()  # 进行参数解析
    # args, unk_args = arg_parser.parse_known_args()

    return args


if __name__ == '__main__':
    config = arg_parse()
    print(config.bs)
    print(config.epochs)
