# Python imports
import os
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import trange

# Local imports
from opts import get_args           # Get all the input arguments
from Models.model import ModelDef
import generateData

print('\033[0;0f\033[0J')
# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

args = get_args()                   # Holds all the input argument

#ModelDef = getattr(__import__(args.model, fromlist=['ModelDef']), 'ModelsDef')  # Get the model definition

if not os.path.exists(args.save):
    os.makedirs(args.save)

args_log = open(args.save + '/args.log', 'w')
args_log.write(str(args))
args_log.close()

seq_len = args.seq
data_dir = args.data
i_width, i_height = args.dim

torch.manual_seed(args.seed)        # Set random seed manually
if torch.cuda.is_available():
    if not args.cuda:
        print(CP_G + "WARNING: You have a CUDA device, so you should probably run with --cuda" + CP_C)
    else:
        torch.cuda.set_device(args.devID)
        torch.cuda.manual_seed(args.seed)
        print("\033[41mGPU({:}) is being used!!!{}".format(torch.cuda.current_device(), CP_C))

# Acquire dataset loader object
data_obj = generateData.TensorFolder(root=data_dir)
data_loader = DataLoader(data_obj, batch_size=args.bs, shuffle=True, num_workers=args.workers)
n_classes = len(data_obj.classes)

# Load pretrained ResNet18 model
pretrained_rn18 = models.resnet18(pretrained=True)
# Remove last layer of pretrained network
model_rn18 = nn.Sequential(*list(pretrained_rn18.children())[:-2])
avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

# Output size of ResNet
n_inp = 512 * math.ceil(i_height/64) * math.ceil(i_width/64)    # input neurons of RNN

# Load model
model = ModelDef(n_inp, [256, n_classes], args.rnn_type)        # Network architecture is stored here

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.eta)
loss_fn = nn.CrossEntropyLoss()


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

if args.cuda:
    model_rn18.cuda()
    avg_pool.cuda()
    model.cuda()

state = model.init_hidden(args.bs)

# Error logger
logger_bw = open(args.save + '/error_bw.log', 'w')
logger_bw.write('{:10}'.format('Train Error'))
logger_bw.write('\n{:-<10}'.format(''))

def train(epoch):
    model.train()
    total_error = 0

    pbar = trange(len(data_loader.dataset), desc='Epoch {:03}'.format(epoch))
    for batch_idx, (data_batch_seq, target_batch_seq) in enumerate(data_loader):
        # Data is of the dimension: batch_size x frames x 3 x height x width
        n_frames = data_batch_seq.size(1)
        # RNN input should be: batch_size x frames x neurons
        rnn_inputs = torch.FloatTensor(args.bs, n_frames, n_inp)
        state = model.init_hidden(args.bs)

        if args.cuda:  # Convert into CUDA tensors
            target_batch_seq = target_batch_seq.cuda()
            data_batch_seq = data_batch_seq.cuda()
            rnn_inputs = rnn_inputs.cuda()

        # Get the output of resnet-18 for individual batch per video
        for seq_idx in range(n_frames):
            temp_variable = avg_pool(model_rn18(Variable(data_batch_seq[:, seq_idx])))
            rnn_inputs[:, seq_idx] = temp_variable.data.view(-1, n_inp)

        optimizer.zero_grad()
        for seq_idx in range(n_frames):
            state = repackage_hidden(state)
            state = model(Variable(rnn_inputs[:, seq_idx]), state)
            temp_loss = 0
            if args.rnn_type == 'LSTM':
                temp_loss = loss_fn(state[-1][0], Variable(target_batch_seq))
            else:
                temp_loss = loss_fn(state[-1], Variable(target_batch_seq))
            # Log batchwise error
            logger_bw.write('\n{:.6f}'.format(temp_loss.data[0]))

        loss = 0
        if args.rnn_type == 'LSTM':
            loss = loss_fn(state[-1][0], Variable(target_batch_seq))
        else:
            loss = loss_fn(state[-1], Variable(target_batch_seq))
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        if batch_idx % 10 == 0:
            if (batch_idx*len(data_batch_seq) + 10) <= len(data_loader.dataset):
                pbar.update(10)
            else:
                pbar.update(len(data_loader.dataset) - batch_idx*len(data_batch_seq))
            '''print('Train Epoch: {:03} [{:03}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_batch_seq), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))
            '''
        total_error += loss.data[0]         # Total loss
    total_error = total_error/math.ceil(len(data_obj)/args.bs)
    pbar.close()
    return total_error


def main():
    print("\n\033[94m\033[1me-Lab Gesture Recognition Training Script\033[0m\n")
    error_log = list()
    for epoch in range(1, args.epochs):
        total_error = train(epoch)
        print('{}{:-<50}{}'.format(CP_R, '', CP_C))
        print('{}Epoch #: {}{:03} | {}Training Error: {}{:.6f}'.format(
            CP_B, CP_C, epoch, CP_B, CP_C, total_error))
        print('{}{:-<50}{}\n'.format(CP_R, '', CP_C))
        error_log.append(total_error)
        with open(args.save + "/model.pyt", 'wb') as f:
            torch.save(model, f)

    logger_bw.close()

    # Log batchwise error
    logger = open(args.save + '/error.log', 'w')
    logger.write('{:10}'.format('Train Error'))
    logger.write('\n{:-<10}'.format(''))
    for total_error in error_log:
        logger.write('\n{:.6f}'.format(total_error))
    logger.close()


if __name__ == "__main__":
    main()
