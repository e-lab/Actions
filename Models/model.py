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

    def __init__(self, n_classes):
        """
        Model initialization
        """
        super(ModelDef, self).__init__()
        self.branch1 = ModelBranch()
        self.branch2 = ModelBranch()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, n_classes)

    def forward(self, x, h0):

        x1 = self.branch1(x).view(1,256)
        x2 = self.branch2(x).view(1,256)
        y = torch.cat((x1, x2, h0), 1)
        y = self.linear1(y)
        y = F.relu(y, inplace=True)
        output = self.linear2(y)

        return output, y

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        h0_default = Variable(weight.new(bsz, 512).zero_())

        return h0_default


class ModelBranch(nn.Module):
    """
    Create one branch of the model with 3D and 2D convolution
    """
    def __init__(self):
        super(ModelBranch, self).__init__()
        self.Conv3D1 = BasicConv3D(3, 32, (3,3,3), stride=(3,1,1), pad=(0,1,1))
        self.Conv3D2 = BasicConv3D(32, 64, (2,3,3), stride=(1,1,1), pad=(0,1,1))

        self.Conv2D1 = BasicConv2D(64, 128, (3,3), stride=(1,1), pad=(1,1), pool=True)
        self.Conv2D2 = BasicConv2D(128, 256, (3,3), stride=(1,1), pad=(1,1), pool=True)
        self.Conv2D3 = BasicConv2D(256, 256, (3,3), stride=(1,1), pad=(1,1), pool=True)
        self.Conv2D4 = nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(1,1))

        self.avgpool2D = nn.AvgPool2d((4,5), stride=(1,1), padding=(0,0))

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
    def __init__(self, iChannels, oChannels, kernel, stride, pad, pool):
        super(BasicConv2D, self).__init__()
        self.Conv2D = nn.Conv2d(iChannels, oChannels, kernel, stride, pad)
        self.pool = pool
        if pool:
            self.pool2D = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))

    def forward(self, x):
        x = self.Conv2D(x)
        if self.pool:
            x = self.pool2D(x)
        x = F.relu(x, inplace=True)
        return x
