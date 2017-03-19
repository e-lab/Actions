from argparse import ArgumentParser


def get_args():
    """
    Parse input arguments for the training script

    :return: all the input arguments
    """
    parser = ArgumentParser(description='e-Lab Gesture Recognition Script')
    _ = parser.add_argument
    _('--data',  type=str, default='/media/HDD2/Models/', help='dataset location')
    _('--save',  type=str, default='/media/HDD2/Models/', help='folder to save outputs')
    _('--model', type=str, default='models/model1.py')
    _('--dim',   type=int, default=(128, 256), nargs=2, help='input image dimension as tuple (HxW)', metavar=('H', 'W'))
    _('--seq',   type=int, default=10, help='sequence length')
    _('--bs',    type=int, default=1, help='batch size')
    _('--lr',    type=float, default=1e-4, help='learning rate')
    _('--cuda',  action='store_true', help='use CUDA')
    _('--seed',  type=int, default=1, help='seed for random number generator')
    args = parser.parse_args()
    return args
