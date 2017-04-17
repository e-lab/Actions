import math
import torch
import torch.nn as nn
from torch.autograd import Variable


# Create Model
class ModelDef(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, x_n, h_n, rnn_type='RNN'):
        """
        Model initialization

        :param x_n: number of input neurons
        :type x_n: int
        :param h_n: # of output neurons for each layer
        :type h_n: list
        :param rnn_type: RNN or LSTM/GRU cell
        :type rnn_type: str
        """
        super(ModelDef, self).__init__()
        self.rnn = nn.ModuleList()

        self.h_n = h_n
        self.n_classes = h_n[-1]

        self.rnn_type = rnn_type + 'Cell'
        if rnn_type == 'RNN':
            for i in range(len(h_n)):
                if i > 0:
                    x_n = h_n[i-1]
                self.rnn.append(nn.RNNCell(x_n, h_n[i], nonlinearity='tanh'))
        else:
            for i in range(len(h_n)):
                if i > 0:
                    x_n = h_n[i-1]
                self.rnn.append(getattr(nn, self.rnn_type)(x_n, h_n[i]))

    def forward(self, x, h0):
        hn = list()
        for i in range(len(self.h_n)):
            hn.append(self.rnn[i](x, h0[i]))
            x = hn[i] if self.rnn_type in ['RNNCell', 'GRUCell'] else hn[i][0]

        return hn

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        h0_default = list()

        for i in range(len(self.h_n)):
            if self.rnn_type in ['RNNCell', 'GRUCell']:
                h0_default.append(Variable(weight.new(bsz, self.h_n[i]).zero_()))
            else:
                h0_default.append((Variable(weight.new(bsz, self.h_n[i]).zero_()),
                        Variable(weight.new(bsz, self.h_n[i]).zero_())))

        return h0_default
