import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models
from argparse import ArgumentParser
from skvideo.io import FFmpegReader
from skimage.transform import resize
import numpy as np
import time
#from Models.model import ModelDef

#Arguments
parser = ArgumentParser(description='e-Lab Gesture Recognition Visualization Script')
_ = parser.add_argument
_('--loc', type=str, default='/media/HDD2/Models/Action/Models/KTH/', help='Trained Model file folder')
_('--model', type=str, default='/lstm4/')
_('--rnn_type', type=str, default='LSTM', help='rnn | lstm | gru')
_('--dim', type=int, default=(160, 120), nargs=2, help='input image dimension as tuple (WxH)', metavar=('W','H'))
_('--bs', type=int, default=1, help='batch size')
_('--seed', type=int, default=1, help='seed for random number generator')
_('--devID', type=int, default=1, help='GPU ID to be used')
_('--cuda', action='store_true', help="use CUDA")
_('--test_filename', type=str, default='/media/HDD1/Datasets/KTH/walking/person01_walking_d1_uncomp.avi', help='Path of the test video file')
_('--classname_path', type=str, default='/media/HDD2/Models/Action/cacheKTH/categories.txt', help='path of file containing the map between class label and class name')
args = parser.parse_args()

#Check for GPU
if torch.cuda.is_available():
    if not args.cuda:
        print("\033[32mWARNING: You have a CUDA device, so you should probably run with --cuda\033[0m")
    else:
        torch.cuda.set_device(args.devID)
        torch.cuda.manual_seed(args.seed)
        print("\033[41mGPU({:}) is being used!!!\033[0m".format(torch.cuda.current_device()))

iWidth = args.dim[0]
iHeight = args.dim[1]

reader = FFmpegReader(args.test_filename)
n_frames = reader.getShape()[0]

frameCount = 0
input_img = torch.cuda.FloatTensor(1, 3, iHeight, iWidth)

print("Number of Frames:" + str(n_frames))

#Resnet-18 pretrained model
torch.manual_seed(args.seed)
n_inp = 512 * math.ceil(iHeight/64) * math.ceil(iWidth/64)    # input neurons of RNN

pretrained_rn18 = models.resnet18(pretrained=True)
pretrained_rn18.eval()
model_rn18 = nn.Sequential(*list(pretrained_rn18.children())[:-2])
avg_pool = nn.AvgPool2d(7, stride=2, padding=3)

# Loading the rnn_type model
model_param = args.loc + args.model + '/model.pt'
model_def = args.loc + args.model + '/modelDef.pt'
print("Trained Model Location : " + model_param)
# Use it if model was not trained on gpu id 0
modelDef = torch.load(model_def)
model = modelDef(n_inp, [512, 512, 6], 'LSTM')
xxx = torch.load(model_param, map_location={'cuda:2':'cuda:1'})
# xxx = torch.load(model_param, map_location={'cuda:3':'cuda:1'})
model.load_state_dict(xxx)
#model = torch.load(trained_model_loc, map_location={'cuda:3':'cuda:1'})
#model = torch.load(trained_model_loc)
model.eval()
trn = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
    state = model.init_hidden(args.bs)

    figure = plt.figure()
    figure.show()

   # colors = ['red', 'blue', 'green', 'yellow', 'orange']
    for frame in reader.nextFrame():
        # Original resolution -> desired resolution
        img_to_show = resize(frame, (iHeight, iWidth), mode='reflect')

        #(height, width, channel) -> (channel, height, width)
        input_img[0] = trn(torch.from_numpy(np.transpose(img_to_show, (2, 0, 1))))

        temp_var = avg_pool(model_rn18(Variable(input_img)))
        rnn_input = temp_var.data.view(-1, n_inp)

        #plt.clf()
        plt.imshow(img_to_show)
        figure.canvas.draw()

        state = model.init_hidden(args.bs)
        state = model.forward(Variable(rnn_input), state)

        #Get top N categories
        h0 = state[-1]
        if args.rnn_type == 'LSTM':
            h0 = state[-1][0]
        topv, topi = h0.data.topk(n_predictions, 1, True)
        predictions = []
        class_map = get_class_name(args.classname_path)

        #ax = figure.add_subplot(212)
        for i in range(n_predictions):
            value = topv[0][i]
            category_index = int(topi[0][i])+1
            print('(%.2f) %s' %(value, class_map[category_index]))
            predictions.append([value, class_map[category_index]])
        pred_txt = "| "
        for v,clazz in predictions:
            pred_txt = pred_txt + clazz +"  |  "

        #plt.title("Frame" + str(frame_id))
        plt.title(pred_txt)
        #txt = figure.text(0,0,pred_txt,
        #            bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
            #txt.remove()
        figure.canvas.draw()
        plt.axis('off')
        #time.sleep(1)
        #txt.remove()
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
