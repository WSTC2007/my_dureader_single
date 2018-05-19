# -*- coding:utf8 -*-
from dataset import BRCDataset
from vocab import Vocab


def calculate_unk(train_files, target_files):
    brc_data = BRCDataset(5, 500, 60, train_files, target_files)
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)
    vocab.filter_tokens_by_cnt(min_cnt=2)
    overlap_num = 0
    dev_vocab = set()
    for word in brc_data.word_iter('dev'):
        dev_vocab.add(word)
    for word in dev_vocab:
        if word in vocab.token2id:
            overlap_num += 1
    print("over lap word is {} in {}".format(overlap_num, len(dev_vocab)))


if __name__ == '__main__':
    calculate_unk(['./data/preprocessed/trainset/zhidao.train.json'], ['./data/preprocessed/devset/zhidao.dev.json'])