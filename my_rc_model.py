# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import json
import torch.nn as nn
from layers.eric_temp_layers import *
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from utils.dureader_eval import compute_bleu_rouge
from utils.dureader_eval import normalize


class MatchLSTM(nn.Module):
    def __init__(self, hidden_size, dropout_prob, max_p_num, max_p_len, max_q_len, max_a_len, vocab):
        super(MatchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.vocab = vocab
        self.use_dropout = self.dropout_prob > 0
        self._def_layers()
        self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def _def_layers(self):
        self.word_embedding = nn.Embedding(self.vocab.size(), self.vocab.embed_dim)
        # initial word embedding
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.vocab.embeddings))
        # self.word_embedding.weight.requires_grad = False
        self.q_encode = nn.LSTM(self.vocab.embed_dim, self.hidden_size, bidirectional=True)
        self.p_encode = nn.LSTM(self.vocab.embed_dim, self.hidden_size, bidirectional=True)

        enc_output_size = self.hidden_size * 2
        self.match_lstm = BiMatchLSTM(input_p_dim=enc_output_size, input_q_dim=enc_output_size,
                                      nhids=self.hidden_size)

        self.fuse_p_encode = nn.LSTM(enc_output_size, self.hidden_size, bidirectional=True)

        self.decoder_init_state_generator = AttentionPooling(enc_output_size, self.hidden_size)

        self.boundary_decoder = BoundaryDecoder(input_dim=enc_output_size, hidden_dim=self.hidden_size)

    def forward(self, p, q):
        """
         input:
            p: batch_size x padded_p_len
            q: batch_size x padded_q_len
        output:
            output: batch_size x padded_p_len x 2
        """
        # get mask
        q_mask = torch.ne(q, 0).float()  # batch_size x padded_q_len
        p_mask = torch.ne(p, 0).float()  # batch_size x padded_p_len

        # encode questions
        # sort_q_len, q_perm_idx = q_length.sort(0, descending=True)
        # _, q_recover_idx = q_perm_idx.sort(0, descending=False)
        # q_perm_idx = q_perm_idx.cuda()
        # q_recover_idx = q_recover_idx.cuda()
        # q = q[q_perm_idx]  # batch_size * max_passage_num x padded_q_len
        q = q.transpose(0, 1).contiguous()  # padded_q_len x batch_size
        q_emb = self.word_embedding(q)  # padded_q_len x batch_size x embed_dim
        # packed_q_in = pack_padded_sequence(q_emb, sort_q_len.data.cpu().numpy())
        # padded_q_len x batch_size x hidden_size * 2
        q_output, _ = self.q_encode(q_emb)
        # batch_size x padded_q_len x hidden_size * 2
        q_output = q_output.transpose(0, 1).contiguous()
        # batch_size x padded_q_len x hidden_size * 2
        q_output = q_output * q_mask.unsqueeze(-1)
        if self.use_dropout:
            q_output = torch.nn.functional.dropout(q_output, p=self.dropout_prob, training=self.training)

        # encode passages
        # sort_p_len, p_perm_idx = p_length.sort(0, descending=True)
        # _, p_recover_idx = p_perm_idx.sort(0, descending=False)
        # p_perm_idx = p_perm_idx.cuda()
        # p_recover_idx = p_recover_idx.cuda()
        # p = p[p_perm_idx]  # batch_size * max_passage_num x padded_p_len
        p = p.transpose(0, 1).contiguous()  # padded_p_len x batch_size
        p_emb = self.word_embedding(p)  # padded_p_len x batch_size x embed_dim
        p_output, _ = self.p_encode(p_emb)
        # padded_p_len x batch_size * max_passage_num x hidden_size * 2
        # p_output, _ = pad_packed_sequence(packed_p_out)
        # batch_size x padded_p_len x hidden_size * 2
        p_output = p_output.transpose(0, 1).contiguous()
        # batch_size x padded_p_len x hidden_size * 2
        p_output = p_output * p_mask.unsqueeze(-1)
        if self.use_dropout:
            p_output = torch.nn.functional.dropout(p_output, p=self.dropout_prob, training=self.training)

        # match question with passage
        # p_q_out: batch_size x padded_p_len x hidden_size * 2
        p_q_out, _ = self.match_lstm(p_output, p_mask, q_output, q_mask)
        if self.use_dropout:
            p_q_out = torch.nn.functional.dropout(p_q_out, p=self.dropout_prob, training=self.training)

        # fuse passage embedding
        # padded_p_len x batch_size x hidden_size * 2
        p_q_out = p_q_out.transpose(0, 1).contiguous()
        fuse_out, _ = self.fuse_p_encode(p_q_out)
        # batch_size x padded_p_len x hidden_size * 2
        fuse_out = fuse_out.transpose(0, 1).contiguous()
        if self.use_dropout:
            fuse_out = torch.nn.functional.dropout(fuse_out, p=self.dropout_prob, training=self.training)

        # decode
        # batch_size x hidden_size
        init_decode_vec = self.decoder_init_state_generator(q_output, q_mask)
        # batch_size x padded_p_len x 2
        output = self.boundary_decoder(fuse_out, p_mask, init_decode_vec)
        return output


class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):
        # basic config
        self.hidden_size = args.hidden_size
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        # self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # the vocab
        self.vocab = vocab

        self.model = MatchLSTM(self.hidden_size, args.dropout_prob, self.max_p_num, self.max_p_len, self.max_q_len,
                               self.max_a_len, self.vocab).cuda()

        # optimizer
        init_learning_rate = args.learning_rate
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if args.optim == 'sgd':
            self.optimizer = torch.optim.SGD(parameters, lr=init_learning_rate)
        elif args.optim == 'adam':
            self.optimizer = torch.optim.Adam(parameters, lr=init_learning_rate)

    def _train_epoch(self, train_batches):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
        """
        total_loss, num_of_batch = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        self.model.train()
        for bitx, batch in enumerate(train_batches, 1):
            num_of_batch += 1
            # batch_size x padded_p_len
            p = Variable(torch.LongTensor(batch['passage_token_ids'])).cuda()
            # batch_size x padded_q_len
            q = Variable(torch.LongTensor(batch['question_token_ids'])).cuda()
            # batch_size
            start_label = Variable(torch.LongTensor(batch['start_id'])).cuda()
            # batch_size
            end_label = Variable(torch.LongTensor(batch['end_id'])).cuda()

            self.optimizer.zero_grad()
            self.model.zero_grad()

            # batch_size x padded_p_len x 2
            answer_prob = self.model(p, q)

            # batch_size x padded_p_len
            answer_begin_prob = answer_prob[:, :, 0].contiguous()
            # batch_size x padded_p_len
            answer_end_prob = answer_prob[:, :, 1].contiguous()

            # batch_size
            answer_begin_prob = torch.log(answer_begin_prob[range(start_label.size(0)),
                                                            start_label.data] + 1e-6)
            # batch_size
            answer_end_prob = torch.log(answer_end_prob[range(end_label.size(0)),
                                                        end_label.data] + 1e-6)
            # batch_size
            total_prob = -(answer_begin_prob + answer_end_prob)
            loss = torch.mean(total_prob)
            total_loss += loss.data[0]
            n_batch_loss += loss.data[0]
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                print('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
            loss.backward()
            self.optimizer.step()
        return 1.0 * total_loss / num_of_batch

    def train(self, data, epochs, batch_size, save_dir, save_prefix, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_rouge_l = 0
        be_patient = 0
        for epoch in range(1, epochs + 1):
            print('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True, train=True)
            train_loss = self._train_epoch(train_batches)
            print('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                print('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False, dev=True)
                    bleu_rouge = self.evaluate(eval_batches)
                    print('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Rouge-L'] > max_rouge_l:
                        self.save(save_dir, save_prefix + '_' + str(epoch))
                        max_rouge_l = bleu_rouge['Rouge-L']
                        be_patient = 0
                    else:
                        be_patient += 1
                        if be_patient >= 5:
                            self.learning_rate *= 0.8
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.learning_rate
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_num, num_of_batch, correct_p_num, select_total_num, select_true_num = 0, 0, 0, 0, 0
        self.model.eval()
        for b_itx, batch in enumerate(eval_batches):
            print("aaaaaaaaa")
            num_of_batch += 1
            # print("now is batch: ", b_itx)
            # batch_size * max_passage_num x padded_p_len
            p = Variable(torch.LongTensor(batch['passage_token_ids']), volatile=True).cuda()
            # batch_size * max_passage_num x padded_q_len
            q = Variable(torch.LongTensor(batch['question_token_ids']), volatile=True).cuda()
            # batch_size
            start_label = Variable(torch.LongTensor(batch['start_id']), volatile=True).cuda()
            # batch_size
            # end_label = Variable(torch.LongTensor(batch['end_id']), volatile=True).cuda()
            # batch_size * max_passage_num x padded_p_len x 2
            answer_prob = self.model(p, q)
            # batch_size * max_passage_num x padded_p_len
            answer_begin_prob = answer_prob[:, :, 0].contiguous()
            # batch_size * max_passage_num x padded_p_len
            answer_end_prob = answer_prob[:, :, 1].contiguous()
            total_num += len(batch['raw_data'])
            # padded_p_len = len(batch['passage_token_ids'][0])
            max_passage_num = p.size(0) // start_label.size(0)
            for idx, sample in enumerate(batch['raw_data']):
                select_total_num += 1
                # max_passage_num x padded_p_len
                start_prob = answer_begin_prob[idx * max_passage_num: (idx + 1) * max_passage_num, :]
                end_prob = answer_end_prob[idx * max_passage_num: (idx + 1) * max_passage_num, :]

                best_answer, best_p_idx = self.find_best_answer(sample, start_prob, end_prob)
                if best_p_idx in sample['answer_passages']:
                    correct_p_num += 1
                if sample['passages'][best_p_idx]['is_selected']:
                    select_true_num += 1

                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                        'question_type': sample['question_type'],
                                        'answers': sample['answers'],
                                        'entity_answers': [[]],
                                        'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            print('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        # ave_loss = 1.0 * total_loss / num_of_batch
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        print('correct selected passage num is {} in {}'.format(select_true_num, select_total_num))
        print('correct passage num is {} in {}'.format(correct_p_num, total_num))
        return bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob):
        #  start_prob: max_passage_num x padded_p_len
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= start_prob.size(0):
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(start_prob[p_idx], end_prob[p_idx], passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer, best_p_idx

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                # print("prob.data[0] is ")
                # print(prob.data[0])
                # print("max_prob is ")
                # print(max_prob)
                if prob.data[0] > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob.data[0]
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_prefix))
        print('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.model.load_state_dict(torch.load(os.path.join(model_dir, model_prefix)))
        print('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
