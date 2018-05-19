# -*- coding:utf8 -*-
from fastText import load_model
import argparse
from dataset import BRCDataset
from vocab import Vocab
import codecs

"""
This module gets the fasttext word embedding of train_set.  
    train_files: The file including the training data
    pre_train_file: The file including the fasttext word embedding vec
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Get fasttext .vec file from .bin file'))
    parser.add_argument('--model_name', type=str, default='./data/wiki.zh.bin')
    parser.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    parser.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    parser.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    parser.add_argument('--train_files', nargs='+',
                               # default=['./data/demo/trainset/search.train.json'],
                               # default=['./data/preprocessed/trainset/search.train.json',
                               #          './data/preprocessed/trainset/zhidao.train.json'],
                               default=['./data/preprocessed/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    parser.add_argument('--pre_train_file', type=str,
                               default='./data/wiki.zh.new.vec', help='pre_train files')
    args = parser.parse_args()
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, args.train_files)
    vocab_set = set()
    for word in brc_data.word_iter('train'):
        vocab_set.add(word.lower())

    f = load_model(args.model_name)

    write_file = codecs.open(args.pre_train_file, 'w', 'utf-8')
    for vocab in vocab_set:
        value_str = [str(i) for i in f.get_word_vector(vocab)]
        write_file.write(vocab + " " + " ".join(value_str) + '\n')
    write_file.close()
    print("over!")
    # vocab_set = Vocab(lower=True)
    # for word in brc_data.word_iter('train'):
    #     vocab_set.add(word)
    # print("vocab size is ", vocab_set.size())
    # vocab_set.filter_tokens_by_cnt(min_cnt=2)
    # print("after filtered vocab size is ", vocab_set.size())
    # vocab_set.randomly_init_embeddings(300)
    # vocab_set.load_pretrained_embeddings('./data/wiki.zh.search.vec')
    # print("over")



