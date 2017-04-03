import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models
from argparse import ArgumentParser
import time
from skimage.io import imsave
from skvideo.io import FFmpegReader
from skimage.transform import resize 

import cv2
import numpy as np
import time

#Local imports
from Models.model import ModelDef

filename = "1.mp4"
#cam = cv2.VideoCapture(filename)

#Arguments
parser = ArgumentParser(description='e-Lab Gesture Recognition Visualization Script')
_ = parser.add_argument
_('--loc', type=str, default='/media/HDD1/Models/Action/niki/Models/', help='Trained Model file')
_('--model', type=str, default='/model.pyt')
_('--rnn_type', type=str, default='RNN', help='rnn | lstm | gru')
_('--dim', type=int, default=(256, 144), nargs=2, help='input image dimension as tuple (WxH)', metavar=('W','H'))
_('--bs', type=int, default=1, help='batch size')
_('--seed', type=int, default=1, help='seed for random number generator')
_('--devID', type=int, default=0, help='GPU ID to be used')
_('--cuda', action='store_true', help="use CUDA")

args = parser.parse_args()
#args.cuda = True
print("CUDA:" + str(args.cuda))
iHeight = args.dim[0]
iWidth = args.dim[1]

reader = FFmpegReader(filename)
n_frames = reader.getShape()[0]

#TODO: Read batch size number of videos at a time.
frameCount = 0
frameCollection = torch.FloatTensor(n_frames, 3, iHeight, iWidth)
for frame in reader.nextFrame():
    # Original resolution -> desired resolution
    tempImg = resize(frame, (iHeight, iWidth))
    
    #(height, width, channel) -> (channel, height, width)
    frameCollection[frameCount] = torch.from_numpy(np.transpose(tempImg, (2, 0, 1)))
    frameCount += 1
print("Number of Frames:" + str(n_frames))

#count = 0
#while cam.isOpened():
#    ret, frames = cam.read()
#    print(count)
#    break
#    count += 1

#print(count)
    
#while(cam.isOpened()):
#    ret, frame = cam.read()
#    if ret==True:
#        frame = cv2.flip(frame, 0)
#    else:
#        break
#cam.release()

#Check for GPU
if torch.cuda.is_available():
    if not args.cuda:
        print("\033[32mWARNING: You have a CUDA device, so you should probably run with --cuda\033[0m")
    else:
        torch.cuda.set_device(args.devID)
        torch.cuda.manual_seed(args.seed)
        print("\033[41mGPU({:}) is being used!!!\033[0m".format(torch.cuda.current_device()))

#Resnet-18 pretrained model
torch.manual_seed(args.seed)
pretrained_rn18 = models.resnet18(pretrained=True)
model_rn18 = nn.Sequential(*list(pretrained_rn18.children())[:-2])
avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

# Loading the rnn_type model
trained_model_loc = args.loc + args.rnn_type.lower() + args.model
print("Trained Model Location : " + trained_model_loc)
model = torch.load(trained_model_loc)

if args.cuda:
    model_rn18.cuda()
    avg_pool.cuda()
    model.cuda()

def repackage_hidden(h):
    """Wraps hidden states in new Variables, tp detach from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate():
    model.eval()

    #Data is converted to dimension: batch_size x frames x 3 x height x width
    collection = torch.FloatTensor(args.bs, n_frames, 3, iHeight, iWidth)
    collection[0] = frameCollection

    rnn_input = torch.FloatTensor(args.bs, n_frames, model.n_inp)
    
    #Get output for resnet-18
    for frame_id in range(n_frames):
        temp_var = avg_pool(model_rn18(Variable(collection[:, frame_id])))
        rnn_input[:, frame_id] = temp_var.data.view(-1, model.n_inp)
    
    #if args.cuda:   # Convert into CUDA tensors
    collection = collection.cuda()
    rnn_input = rnn_input.cuda()


   #Get ouptut for rnn model
    h0 = model.init_hidden(args.bs)
    c0 = model.init_hidden(args.bs)

    for frame_id in range(n_frames):
        if args.rnn_type=='LSTM':
            h0 = repackage_hidden(h0)
            c0 = repackage_hidden(c0)
            h0, c0 = model.forward(Variable(rnn_input[:, frame_id]), h0, c0)
        else:
            h0 = repackage_hidden(h0)
            h0 = model.forward(Variable(rnn_input[:, frame_id]), h0, None)
            
    return repackage_hidden(h0)


def predict(n_predictions = 3):
    output = evaluate()
    
    #Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' %(value, category_index))
        predictions.append([value, category_index])
    return predictions

predict()
