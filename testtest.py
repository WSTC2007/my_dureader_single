# -*- coding:utf8 -*-
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import codecs

# rnn = nn.LSTM(8, 10, 1, bidirectional=True)
# # input = Variable(torch.randn(5, 3, 8))
# # length = np.array([5, 3, 1])
# # input = pack_padded_sequence(input, length)
# # # h0 = Variable(torch.randn(2, 3, 20))
# # # c0 = Variable(torch.randn(2, 3, 20))
# # output, hn = rnn(input)
# # output, _ = pad_packed_sequence(output)
# # print(output)
# a = [[1, 3, 4, 2, 0, 0, 0], [2, 2, 3, 2, 0, 0, 0]]
# b = Variable(torch.Tensor(a))
# c = torch.ne(b, 0).float()
# print(c)
# # print(b)
# G_p = self.W_p(input_p).unsqueeze(1)  # batch x None x out
# G_q = self.W_q(input_q)  # batch x time x out
# G_r = self.W_r(h_tm1).unsqueeze(1)  # batch x None x out
# G = F.tanh(G_p + G_q + G_r)  # batch x time x out
# alpha = torch.matmul(G, self.w[depth])  # batch x time

a = torch.Tensor([1, 2])
print(a.size(0))