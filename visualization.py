import math
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models
from argparse import ArgumentParser
from skimage.io import imsave
from skvideo.io import FFmpegReader
from skimage.transform import resize
from PIL import Image
import cv2
import numpy as np
import time
import os
from Models.model import ModelDef

#Arguments
parser = ArgumentParser(description='e-Lab Gesture Recognition Visualization Script')
_ = parser.add_argument
_('--loc', type=str, default='/media/HDD1/Models/Action/abhi/Models/rnn3', help='Trained Model file folder')
_('--model', type=str, default='/model.pyt')
_('--rnn_type', type=str, default='RNN', help='rnn | lstm | gru')
_('--dim', type=int, default=(256, 144), nargs=2, help='input image dimension as tuple (WxH)', metavar=('W','H'))
_('--bs', type=int, default=1, help='batch size')
_('--seed', type=int, default=1, help='seed for random number generator')
_('--devID', type=int, default=1, help='GPU ID to be used')
_('--cuda', action='store_true', help="use CUDA")
_('--test_filename', type=str, default='1.mp4', help='Path of the test video file')
_('--classname_path', type=str, default='/media/HDD1/Models/Action/niki/cache/classList.txt', help='path of file containing the map between class label and class name')
args = parser.parse_args()

iHeight = args.dim[0]
iWidth = args.dim[1]

reader = FFmpegReader(args.test_filename)
n_frames = reader.getShape()[0]

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
trained_model_loc = args.loc + args.model
print("Trained Model Location : " + trained_model_loc)
model = torch.load(trained_model_loc)
model.eval()

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

def evaluate(n_predictions = 3):
    #Data is converted to dimension: batch_size x frames x 3 x height x width
    collection = torch.FloatTensor(args.bs, n_frames, 3, iHeight, iWidth)
    collection[0] = frameCollection

    #print(model.n_inp)
    rnn_input = torch.FloatTensor(args.bs, n_frames, model.n_inp)

    if args.cuda:   # Convert into CUDA tensors
        collection = collection.cuda()
        rnn_input = rnn_input.cuda()

    #Get output for resnet-18
    for frame_id in range(n_frames):
        temp_var = avg_pool(model_rn18(Variable(collection[:, frame_id])))
        rnn_input[:, frame_id] = temp_var.data.view(-1, model.n_inp)

    #plt.show()
   #Get ouptut for rnn model
    h0 = model.init_hidden(1)
    c0 = model.init_hidden(1)

    figure = plt.figure()
    img = plt.imshow(frameCollection[0].numpy().reshape(iHeight, iWidth, 3), animated=True)
    figure.show()

   # colors = ['red', 'blue', 'green', 'yellow', 'orange']
    for frame_id in range(n_frames):
        #plt.clf()
        plt.imshow(frameCollection[frame_id].numpy().reshape(iHeight, iWidth, 3))
        plt.title("Frame" + str(frame_id))
        figure.canvas.draw()

        h0 = repackage_hidden(h0)
        h0 = model.forward(Variable(rnn_input[:, frame_id]), h0)
        '''if args.rnn_type=='LSTM':
            h0 = repackage_hidden(h0)
            #c0 = repackage_hidden(c0)
            h0, c0 = model.forward(Variable(rnn_input[:, frame_id]),h0)
        else:
            h0 = repackage_hidden(h0)
            h0 = model.forward(Variable(rnn_input[:, frame_id]), h0, None)
        '''
        #Get top N categories
        topv, topi = h0.data.topk(n_predictions, 1, True)
        predictions = []
        class_map = get_class_name(args.classname_path)

        #ax = figure.add_subplot(212)
        for i in range(n_predictions):
            value = topv[0][i]
            category_index = int(topi[0][i])+1
            print('(%.2f) %s' %(value, class_map[category_index]))
            predictions.append([value, class_map[category_index]])
        pred_txt = ""
        for v,clazz in predictions:
            pred_txt = pred_txt + clazz +"  ,  "

        print(pred_txt)
        txt = figure.text(0,0,pred_txt,
                    bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
            #txt.remove()
        figure.canvas.draw()
        time.sleep(1)
        txt.remove()
        print('{:-<40}\n'.format(''))


def get_class_name(classnamelist_file_path):
    class_map = {}
    file = open(classnamelist_file_path, "r")
    for line in file.readlines():
        num, name = line.strip("\n").split(",")
        class_map[int(num)] = name
    file.close()
    return class_map

evaluate()
#get_class_name(args.classname_path)
