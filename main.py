# Python imports
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models

# Local imports
from opts import get_args           # Get all the input arguments
from Models.model import ModelDef   # Get the model definition
import generateData

print('\033[0;0f\033[0J')

args = get_args()                   # Holds all the input argument

seq_len = args.seq
data_dir = args.data
i_height, i_width = args.dim

torch.manual_seed(args.seed)        # Set random seed manually
if torch.cuda.is_available():
    if not args.cuda:
        print("\033[32mWARNING: You have a CUDA device, so you should probably run with --cuda\033[0m")
    else:
        torch.cuda.set_device(args.devID)
        torch.cuda.manual_seed(args.seed)
        print("\033[41mGPU({:}) is being used!!!\033[0m".format(torch.cuda.current_device()))

# Acquire dataset loader object
data_obj = generateData.TensorFolder(root=data_dir)
data_loader = DataLoader(data_obj, batch_size=args.bs, shuffle=True, num_workers=args.workers)
n_classes = len(data_obj.classes)

# Load model
model = ModelDef(i_height, i_width, n_classes, args.rnn_type)        # Network architecture is stored here

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.eta)
loss_fn = nn.MSELoss()


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# Load pretrained ResNet18 model
pretrained_rn18 = models.resnet18(pretrained=True)
# Remove last layer of pretrained network
model_rn18 = nn.Sequential(*list(pretrained_rn18.children())[:-2])
avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

if args.cuda:
    model_rn18.cuda()
    avg_pool.cuda()
    model.cuda()

h0 = model.init_hidden(args.bs)
c0 = model.init_hidden(args.bs)

def train(epoch):
    model.train()
    for batch_idx, (data_batch_seq, target_batch_seq) in enumerate(data_loader):
        # Data is of the dimension: batch_size x frames x 3 x height x width
        n_frames = data_batch_seq.size(1)
        n_inp = 512 * (i_height // 64) * (i_width // 64)    # input neurons of RNN
        # RNN input should be: batch_size x frames x neurons
        rnn_inputs = torch.FloatTensor(args.bs, n_frames, n_inp)
        h0 = model.init_hidden(args.bs)
        if args.rnn_type == 'LSTM':
            c0 = model.init_hidden(args.bs)

        if args.cuda:  # Convert into CUDA tensors
            target_batch_seq = target_batch_seq.cuda()
            data_batch_seq = data_batch_seq.cuda()
            rnn_inputs = rnn_inputs.cuda()

        # Get the output of resnet-18 for individual batch per video
        for seq_idx in range(n_frames):
            temp_variable = avg_pool(model_rn18(Variable(data_batch_seq[:, seq_idx])))
            rnn_inputs[:, seq_idx] = temp_variable.data.view(-1, n_inp)

        optimizer.zero_grad()
        model.zero_grad()
        for seq_idx in range(n_frames):
            if args.rnn_type == 'LSTM':
                h0 = repackage_hidden(h0)
                c0 = repackage_hidden(c0)
                h0, c0 = model.forward(Variable(rnn_inputs[:, seq_idx]), h0, c0)
            else:
                h0 = repackage_hidden(h0)
                h0 = model.forward(Variable(rnn_inputs[:, seq_idx]), h0, None)

        loss = loss_fn(h0, Variable(target_batch_seq))
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_batch_seq), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))


def main():
    print("\n\033[94m\033[1me-Lab Gesture Recognition Training Script\033[0m\n")
    for epoch in range(1, 100):
        train(epoch)
        with open(args.save + "model.net", 'wb') as f:
            torch.save(model, f)


if __name__ == "__main__":
    main()
