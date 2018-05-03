import torch
import numpy as np
import torch.nn.functional as F
import codecs


def masked_softmax(x, m=None, axis=-1):
    """
    Softmax with mask (optional)
    """
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


class AttentionPooling(torch.nn.Module):
    """
     inputs: no_dup_q_encode:     batch_size x padded_q_len x hidden_size * 2
             q_mask:              batch_size x padded_q_len
     outputs: z:   batch_size x hidden_size
     """

    def __init__(self, input_dim, output_dim):
        super(AttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.W_r = torch.nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.W_mask = torch.nn.Linear(self.output_dim, 1, bias=False)
        self.out_l = torch.nn.Linear(self.input_dim, self.output_dim)
        self.w = torch.nn.Parameter(torch.FloatTensor(1, self.output_dim))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.W.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.W_r.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.out_l.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.W_mask.weight.data, gain=1)
        torch.nn.init.normal(self.w.data, mean=0, std=0.05)

    def forward(self, no_dup_q_encode, q_mask):
        G_q = self.W(no_dup_q_encode)  # batch_size x padded_q_len x output_dim
        G_r = self.W_r(self.w)  # 1 x output_dim
        G = F.tanh(G_q + G_r)  # batch_size x padded_q_len x output_dim
        alpha = self.W_mask(G).squeeze(-1)  # batch_size x padded_q_len
        alpha = masked_softmax(alpha, q_mask, axis=-1)  # batch_size x padded_q_len
        alpha = alpha.unsqueeze(1)  # batch_size x 1 x padded_q_len
        z = torch.bmm(alpha, no_dup_q_encode)  # batch_size x 1 x hidden_size * 2
        z = z.squeeze(1)  # batch_size x hidden_size * 2
        z = self.out_l(z)  # batch_size x hidden_size
        return z


class MatchLSTMAttention(torch.nn.Module):
    """
        input:  p:          batch x inp_p
                p_mask:     batch
                q:          batch x time x inp_q
                q_mask:     batch x time
                h_tm1:      batch x out
        output: z:          batch x inp_p+inp_q
    """

    def __init__(self, input_p_dim, input_q_dim, output_dim):
        super(MatchLSTMAttention, self).__init__()
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.output_dim = output_dim

        self.W_p = torch.nn.Linear(self.input_p_dim, self.output_dim, bias=False)
        self.W_q = torch.nn.Linear(self.input_q_dim, self.output_dim, bias=False)
        self.W_r = torch.nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.w = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.match_b = torch.nn.Parameter(torch.FloatTensor(1))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.W_p.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.W_q.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.W_r.weight.data, gain=1)
        torch.nn.init.normal(self.w.data, mean=0, std=0.05)
        self.match_b.data.fill_(1.0)

    def forward(self, input_p, input_q, mask_q, h_tm1):
        G_p = self.W_p(input_p).unsqueeze(1)  # batch x None x out
        G_q = self.W_q(input_q)  # batch x time x out
        G_r = self.W_r(h_tm1).unsqueeze(1)  # batch x None x out
        G = F.tanh(G_p + G_q + G_r)  # batch x time x out
        alpha = torch.matmul(G, self.w)  # batch x time
        alpha = alpha + self.match_b.unsqueeze(0)  # batch x time
        alpha = masked_softmax(alpha, mask_q, axis=-1)  # batch x time
        alpha = alpha.unsqueeze(1)  # batch x 1 x time
        # batch x time x input_q, batch x 1 x time
        z = torch.bmm(alpha, input_q)  # batch x 1 x input_q
        z = z.squeeze(1)  # batch x input_q
        z = torch.cat([input_p, z], 1)  # batch x input_p+input_q
        return z


class MatchLSTM(torch.nn.Module):
    """
    inputs: p:          batch x time x inp_p
            mask_p:     batch x time
            q:          batch x time x inp_q
            mask_q:     batch x time
    outputs:
            encoding:   batch x time x h
            mask_p:     batch x time
    """

    def __init__(self, input_p_dim, input_q_dim, nhids, attention_layer):
        super(MatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.attention_layer = attention_layer
        self.lstm_cell = torch.nn.LSTMCell(self.input_p_dim + self.input_q_dim, self.nhids)


    def get_init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(torch.autograd.Variable(weight.new(bsz, self.nhids).zero_()).cuda(),
                 torch.autograd.Variable(weight.new(bsz, self.nhids).zero_()).cuda())]

    def forward(self, input_p, mask_p, input_q, mask_q):
        batch_size = input_p.size(0)
        state_stp = self.get_init_hidden(batch_size)

        for t in range(input_p.size(1)):

            input_mask = mask_p[:, t]  # batch_size
            input_mask = input_mask.unsqueeze(1)  # batch_size x None
            curr_input = input_p[:, t]  # batch_size x inp_p
            previous_h, previous_c = state_stp[t]
            drop_input = self.attention_layer(curr_input, input_q, mask_q, h_tm1=previous_h)
            new_h, new_c = self.lstm_cell(drop_input, (previous_h, previous_c))
            new_h = new_h * input_mask + previous_h * (1 - input_mask)
            new_c = new_c * input_mask + previous_c * (1 - input_mask)
            state_stp.append((new_h, new_c))

        states = [h[0] for h in state_stp[1:]]  # list of batch x hid
        states = torch.stack(states, 1)  # batch x time x hid
        return states


class BiMatchLSTM(torch.nn.Module):
    """
    inputs: input_p:    batch_size * max_passage_num x padded_p_len x hidden_size * 2
            mask_p:     batch_size * max_passage_num x padded_p_len
            input_q:    batch_size * max_passage_num x padded_q_len x hidden_size * 2
            mask_q:     batch_size * max_passage_num x padded_q_len

    outputs: encoding:   batch x time x hid
             last state: batch x hid
    """

    def __init__(self, input_p_dim, input_q_dim, nhids):
        super(BiMatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.attention_layer = MatchLSTMAttention(input_p_dim, input_q_dim, output_dim=self.nhids)

        self.forward_rnn = MatchLSTM(input_p_dim=self.input_p_dim, input_q_dim=self.input_q_dim,
                                     nhids=self.nhids, attention_layer=self.attention_layer)

        self.backward_rnn = MatchLSTM(input_p_dim=self.input_p_dim, input_q_dim=self.input_q_dim,
                                      nhids=self.nhids, attention_layer=self.attention_layer)

    def flip(self, tensor, flip_dim=0):
        # flip
        idx = [i for i in range(tensor.size(flip_dim) - 1, -1, -1)]
        idx = torch.autograd.Variable(torch.LongTensor(idx))
        idx = idx.cuda()
        inverted_tensor = tensor.index_select(flip_dim, idx)
        return inverted_tensor

    def forward(self, input_p, mask_p, input_q, mask_q):

        # forward pass
        forward_states = self.forward_rnn.forward(input_p, mask_p, input_q, mask_q)
        forward_last_state = forward_states[:, -1]  # batch x hid

        # backward pass
        input_p_inverted = self.flip(input_p, flip_dim=1)  # batch x time x p_dim (backward)
        mask_p_inverted = self.flip(mask_p, flip_dim=1)  # batch x time (backward)
        backward_states = self.backward_rnn.forward(input_p_inverted, mask_p_inverted, input_q, mask_q)
        backward_last_state = backward_states[:, -1]  # batch x hid
        backward_states = self.flip(backward_states, flip_dim=1)  # batch x time x hid

        concat_states = torch.cat([forward_states, backward_states], -1)  # batch x time x hid * 2
        concat_states = concat_states * mask_p.unsqueeze(-1)  # batch x time x hid * 2
        concat_last_state = torch.cat([forward_last_state, backward_last_state], -1)  # batch x hid * 2

        return concat_states, concat_last_state


class BoundaryDecoderAttention(torch.nn.Module):
    """
        input:  H_r:        batch x time x input_dim
                mask_r:     batch x time
                h_tm1:      batch x out
        output: z:          batch x input_dim
    """

    def __init__(self, input_dim, output_dim,):
        super(BoundaryDecoderAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.V = torch.nn.Linear(self.input_dim, self.output_dim)
        self.W_a = torch.nn.Linear(self.output_dim, self.output_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.c = torch.nn.Parameter(torch.FloatTensor(1))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.V.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.W_a.weight.data, gain=1)
        self.V.bias.data.fill_(0)
        self.W_a.bias.data.fill_(0)
        torch.nn.init.normal(self.v.data, mean=0, std=0.05)
        self.c.data.fill_(1.0)

    def forward(self, H_r, mask_r, h_tm1):
        # H_r: batch x time x inp
        # mask_r: batch x time
        # h_tm1: batch x out
        batch_size, time = H_r.size(0), H_r.size(1)
        Fk = self.V.forward(H_r.view(-1, H_r.size(2)))  # batch*time x out
        Fk_prime = self.W_a.forward(h_tm1)  # batch x out
        Fk = Fk.view(batch_size, time, -1)  # batch x time x out
        Fk = torch.tanh(Fk + Fk_prime.unsqueeze(1))  # batch x time x out

        beta = torch.matmul(Fk, self.v)  # batch x time
        beta = beta + self.c.unsqueeze(0)  # batch x time
        beta = masked_softmax(beta, mask_r, axis=-1)  # batch x time
        z = torch.bmm(beta.view(beta.size(0), 1, beta.size(1)), H_r)  # batch x 1 x inp
        z = z.view(z.size(0), -1)  # batch x inp
        return z, beta


class BoundaryDecoder(torch.nn.Module):
    """
    input:  encoded stories:    batch_size x padded_p_len x hidden_size * 2
            story mask:         batch_size x padded_p_len
            init states:        batch_size x hidden_size
    output: res:                batch_size x padded_p_len x 2
    """

    def __init__(self, input_dim, hidden_dim):
        super(BoundaryDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_layer = BoundaryDecoderAttention(input_dim=input_dim, output_dim=hidden_dim)

        # self.rnn = LSTMCell(self.input_dim, self.hidden_dim, use_layernorm=False, use_bias=True)
        self.rnn = torch.nn.LSTMCell(self.input_dim, self.hidden_dim)

    def forward(self, x, x_mask, h_0):

        state_stp = [(h_0, h_0)]
        beta_list = []
        for t in range(2):

            previous_h, previous_c = state_stp[t]
            curr_input, beta = self.attention_layer(x, x_mask, h_tm1=previous_h)
            new_h, new_c = self.rnn(curr_input, (previous_h, previous_c))
            state_stp.append((new_h, new_c))
            beta_list.append(beta)

        # beta list: list of batch x time
        res = torch.stack(beta_list, 2)  # batch x time x 2
        res = res * x_mask.unsqueeze(2)  # batch x time x 2
        return res


