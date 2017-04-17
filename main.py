# Python imports
import os
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Local imports
from opts import get_args           # Get all the input arguments
from Models.model import ModelDef
from train import train as trainClass
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
print(data_obj)
data_loader = DataLoader(data_obj, batch_size=args.bs, shuffle=True, num_workers=args.workers)
n_classes = len(data_obj.classes)
data_len = len(data_obj)

# Load pretrained ResNet18 model
pretrained_rn18 = models.resnet18(pretrained=True)
# Remove last layer of pretrained network
model_rn18 = nn.Sequential(*list(pretrained_rn18.children())[:-2])
avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

# Output size of ResNet
n_inp = 512 * math.ceil(i_height/64) * math.ceil(i_width/64)    # input neurons of RNN

# Load model
model = ModelDef(n_inp, [1024, 256, n_classes], args.rnn_type)        # Network architecture is stored here

if args.cuda:
    model_rn18.cuda()
    avg_pool.cuda()
    model.cuda()

state = model.init_hidden(args.bs)

def main():
    print("\n\033[94m\033[1me-Lab Gesture Recognition Training Script\033[0m\n")
    error_log = list()
    prev_error = 1000

    train = trainClass(model, data_loader, data_len, n_inp, avg_pool, model_rn18, args)
    for epoch in range(1, args.epochs):
        total_error = train.forward(epoch)
        print('{}{:-<50}{}'.format(CP_R, '', CP_C))
        print('{}Epoch #: {}{:03} | {}Training Error: {}{:.6f}'.format(
            CP_B, CP_C, epoch, CP_B, CP_C, total_error))
        error_log.append(total_error)
        if total_error <= prev_error:
            prev_error = total_error
            print(CP_G + "Saving model!!!" + CP_C)
            print('{}{:-<50}{}\n'.format(CP_R, '', CP_C))
            with open(args.save + "/model.pyt", 'wb') as f:
                torch.save(model, f)

    train.logger_bw.close()

    # Log batchwise error
    logger = open(args.save + '/error.log', 'w')
    logger.write('{:10}'.format('Train Error'))
    logger.write('\n{:-<10}'.format(''))
    for total_error in error_log:
        logger.write('\n{:.6f}'.format(total_error))
    logger.close()


if __name__ == "__main__":
    main()
