import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models
from argparse import ArgumentParser
from skimage.io import imsave
from skvideo.io import FFmpegReader
from skimage.transform import resize 
import cv2
import numpy as np
import time
import os
from Models.model import ModelDef


#Arguments
parser = ArgumentParser(description='e-Lab Gesture Recognition Visualization Script')
_ = parser.add_argument
_('--loc', type=str, default='/media/HDD1/Models/Action/abhi/Models/', help='Trained Model file folder')
_('--model', type=str, default='/model.pyt')
_('--rnn_type', type=str, default='LSTM', help='rnn | lstm | gru')
_('--dim', type=int, default=(256, 144), nargs=2, help='input image dimension as tuple (WxH)', metavar=('W','H'))
_('--bs', type=int, default=1, help='batch size')
_('--seed', type=int, default=1, help='seed for random number generator')
_('--devID', type=int, default=1, help='GPU ID to be used')
_('--cuda', action='store_true', help="use CUDA")
_('--test_filename', type=str, default='1.mp4', help='Path of the test video file')
_('--classname_path', type=str, default='/media/HDD1/Models/Action/niki/cache/classList.txt', help='path of file containing the map between class label and class name')
args = parser.parse_args()

#args.cuda=True
iHeight = args.dim[0]
iWidth = args.dim[1]

reader = FFmpegReader(args.test_filename)
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
    #model_rn18.cuda()
    avg_pool.cuda(1)
    model.cuda(1)

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
    collection = collection.cuda(1)
    rnn_input = rnn_input.cuda(1)


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

def get_class_name(classnamelist_file_path):
    class_map = {}
    file = open(classnamelist_file_path, "r")
    for line in file.readlines():
        num, name = line.strip("\n").split(",")
        class_map[int(num)] = name
    file.close()  
    return class_map          

def predict(n_predictions = 5):
    output = evaluate().float()
       
    #Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []
    class_map = get_class_name(args.classname_path)

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = int(topi[0][i])+1
        print('(%.2f) %s' %(value, class_map[category_index]))
        predictions.append([value, class_map[category_index]])
    return predictions

predict()
#get_class_name(args.classname_path)
