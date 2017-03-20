import torch.nn as nn
from torch.autograd import Variable


# Create Model
class ModelDef(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, i_height, i_width, n_classes):
        """
        Model initialization

        :param i_height: height of input image
        :type i_height: int
        :param i_width: width of input image
        :type i_width: int
        """
        super(ModelDef, self).__init__()
        # Output size of ResNet18
        self.o_height = i_height // 64
        self.o_width = i_width // 64
        self.n_layers = 1
        self.n_classes = n_classes
        self.n_inp = 512 * self.o_height * self.o_width  # Input neurons of rnn

        self.rnn = nn.RNNCell(self.n_inp, self.n_classes, nonlinearity='tanh')

    def forward(self, x, h0):
        hn = self.rnn(x, h0)

        return hn

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return Variable(weight.new(bsz, self.n_classes).zero_())
