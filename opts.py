from argparse import ArgumentParser


def get_args():
    """
    Parse input arguments for the training script

    :return: all the input arguments
    """
    parser = ArgumentParser(description='e-Lab Gesture Recognition Script')
    _ = parser.add_argument
    _('--data',     type=str,   default='/media/HDD2/Models/', help='dataset location')
    _('--save',     type=str,   default='/media/HDD2/Models/', help='folder to save outputs')
    _('--model',    type=str,   default='models/model.py')
    _('--rnn_type', type=str,   default='LSTM', help='RNN | LSTM | GRU')
    _('--dim',      type=int,   default=(176, 120), nargs=2, help='input image dimension as tuple (HxW)', metavar=('W', 'H'))
    _('--seq',      type=int,   default=10, help='sequence length')
    _('--bs',       type=int,   default=1, help='batch size')
    _('--lr',       type=float, default=1e-4, help='learning rate')
    _('--eta',      type=float, default=0.9, help='momentum')
    _('--seed',     type=int,   default=1, help='seed for random number generator')
    _('--epochs',   type=int,   default=300, help='# of epochs you want to run')
    _('--devID',    type=int,   default=0, help='GPU ID to be used')
    _('--workers',  type=int,   default=0, help='number of workers for data loader')
    _('--cuda',     action='store_true', help='use CUDA')
    args = parser.parse_args()
    return args
