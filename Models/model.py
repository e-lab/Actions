import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Create Model
class ModelDef(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, h_n, rnn_type):
        """
        Model initialization
        """
        super(ModelDef, self).__init__()
        self.h_n = h_n

        self.branch1 = ModelBranch()
        self.branch2 = ModelBranch()

        self.rnn = nn.ModuleList()
        x_n = 512
        self.rnn_type = rnn_type + 'Cell'
        if rnn_type == 'RNN':
            for i in range(len(h_n)):
                if i > 0:
                    x_n = h_n[i-1]

                nonlinearity = 'tanh'
                dropout = True
                if(i == len(h_n) - 1):
                    nonlinearity = False
                    dropout = False
                self.rnn.append(RNNCell(x_n, h_n[i], h_n[i], nonlinearity, dropout))
        else:
            for i in range(len(h_n)):
                if i > 0:
                    x_n = h_n[i-1]
                self.rnn.append(getattr(nn, self.rnn_type)(x_n, h_n[i]))

    def forward(self, x, h0):

        x1 = self.branch1(x).view(1,256)
        x2 = self.branch2(x).view(1,256)
        y = torch.cat((x1, x2), 1)

        hn = list()
        for i in range(len(self.h_n)):
            hn.append(self.rnn[i](y, h0[i]))
            y = hn[i] if self.rnn_type in ['RNNCell', 'GRUCell'] else hn[i][0]

        return y, hn

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


class RNNCell(nn.Module):
    def __init__(self, n_ip, n_h, n_op, nonlinearity, dropout):
        super(RNNCell, self).__init__()
        self.nonlinearity = nonlinearity
        self.linear = nn.Linear(n_ip+n_h, n_op)
        self.dropout = dropout
        self.dpout = nn.Dropout(p=0.1)

    def forward(self, x, h):
        x_cat = torch.cat((x, h), 1)
        y = self.linear(x_cat)
        if self.dropout:
            y = self.dpout(y)
        if self.nonlinearity:
            y = F.relu(y, inplace=True)

        return y


class ModelBranch(nn.Module):
    """
    Create one branch of the model with 3D and 2D convolution
    """
    def __init__(self):
        super(ModelBranch, self).__init__()
        self.Conv3D1 = BasicConv3D(3, 32, (3,3,3), stride=(3,1,1), pad=(0,1,1))
        self.Conv3D2 = BasicConv3D(32, 64, (2,3,3), stride=(1,1,1), pad=(0,1,1))

        self.Conv2D1 = BasicConv2D(64, 128, (3,3), stride=(1,1), pad=(1,1), pool=True, dropout=True)
        self.Conv2D2 = BasicConv2D(128, 256, (3,3), stride=(1,1), pad=(1,1), pool=True, dropout=True)
        self.Conv2D3 = BasicConv2D(256, 256, (3,3), stride=(1,1), pad=(1,1), pool=True, dropout=True)
        self.Conv2D4 = nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(1,1))

        self.avgpool2D = nn.AvgPool2d((2,3), stride=(1,1), padding=(0,0))

    def forward(self, x):
        # x = 3x6x160x120
        x = self.Conv3D1(x)
        # x = 32x2x80x60
        x = self.Conv3D2(x).squeeze(2)
        # x = 64x40x30
        x = self.Conv2D1(x)
        # x = 128x20x15
        x = self.Conv2D2(x)
        # x = 256x10x8
        x = self.Conv2D3(x)
        # x = 256x5x4
        x = self.Conv2D4(x)
        # x = 256x5x4
        x = self.avgpool2D(x)
        # x = 256x1x1
        x = F.relu(x, inplace=True)

        return x


class BasicConv3D(nn.Module):
    def __init__(self, iChannels, oChannels, kernel, stride, pad):
        super(BasicConv3D, self).__init__()
        self.Conv3D = nn.Conv3d(iChannels, oChannels, kernel, stride, pad)
        self.pool3D = nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))

    def forward(self, x):
        x = self.Conv3D(x)
        x = self.pool3D(x)
        x = F.relu(x, inplace=True)
        return x


class BasicConv2D(nn.Module):
    def __init__(self, iChannels, oChannels, kernel, stride, pad, pool, dropout):
        super(BasicConv2D, self).__init__()
        self.Conv2D = nn.Conv2d(iChannels, oChannels, kernel, stride, pad)
        self.dropout = dropout
        self.dpout = nn.Dropout2d(p=0.1)
        self.pool = pool
        if pool:
            self.pool2D = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))

    def forward(self, x):
        x = self.Conv2D(x)
        if self.dropout:
            x = self.dpout(x)
        if self.pool:
            x = self.pool2D(x)
        x = F.relu(x, inplace=True)
        return x
