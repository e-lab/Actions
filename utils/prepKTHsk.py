import numpy as np
import os
import torch
from skvideo.io import FFmpegReader
from skimage.transform import resize
from tqdm import trange
from argparse import ArgumentParser

parser = ArgumentParser(description='e-Lab Gesture Recognition Script')
_ = parser.add_argument
_('--data',  type=str, default='/media/HDD2/Models/', help='dataset location')
_('--save',  type=str, default='/media/HDD2/Models/', help='folder to save outputs')
_('--skip',  type=int, default=5, help='# of frames to skip')
_('--dim',   type=int, default=(160, 120), nargs=2, help='input image dimension as tuple (WxH)', metavar=('W', 'H'))

print('\033[0;0f\033[0J')
args = parser.parse_args()

f = open('utils/tvt_split.txt', 'r')

# maps personID to tvt set
split_dict = {}
for i in range(3):
    file_line = f.readline().rstrip().split(" ")
    for j in range(len(file_line) - 1):
        split_dict.update({file_line[j+1]: file_line[0]})


iWidth = args.dim[0]
iHeight = args.dim[1]

rootDirSave = args.save
if not os.path.exists(rootDirSave):
    os.makedirs(rootDirSave)
    os.makedirs(rootDirSave + '/Train')
    os.makedirs(rootDirSave + '/Val')
    os.makedirs(rootDirSave + '/Test')

rootDirLoad = args.data
pbar1 = trange(600, ncols=100, position=0, desc='Overall progress      ')
classes = []

dataList = open(rootDirSave + "/dataList.txt", "w")
dataList.write('{:12} {:12} {:12}'.format('Class name', '# of Frames', 'Filename'))
dataList.write('\n{:-<47}'.format(''))


nVideos = 0
file_line = f.readline()
while file_line:
    file_line = file_line.rstrip().split(" ")
    personID, subDir, _ = file_line[0].split('_')
    personID = personID[-2:]

    # Prepare input file name and directory to save output tensors
    videoname = os.path.join(rootDirLoad, subDir, file_line[0]) + '_uncomp.avi'
    classDir = os.path.join(rootDirSave, split_dict[personID], subDir)

    if not os.path.exists(classDir):
        os.makedirs(classDir)

    if os.path.isfile(videoname):
        reader = FFmpegReader(videoname)
        nFrames = reader.getShape()[0]

        nVideos += 1
        dataList.write('\n{:12} {:<12} {:02}.pyt'.format(subDir, nFrames, nVideos))
        pbar2 = trange(nFrames, ncols=100, position=2, desc='Video progress        ')

        frameCount = 0
        # Tensor to save all the fames of a video
        frameCollection = torch.FloatTensor()
        for frame in reader.nextFrame():        # Garb each frame
            frameCount += 1
            if (frameCount % (args.skip+1)) == 0:
                # Original resolution -> desired resolution
                tempImg = resize(frame, (iHeight, iWidth), mode='reflect')
                # (height, width, channel) -> (channel, height, width)
                frameCollection = torch.cat((frameCollection,
                                  torch.unsqueeze(
                                  torch.from_numpy(
                                  np.transpose(tempImg, (2, 0, 1))).float(), 0)), 0)

            pbar2.update(1)

        # Split video into chunks and save them
        for sequenceIdx in range((len(file_line)-1)//2):
            startIdx = (int(file_line[2*sequenceIdx + 1])-1) // (args.skip+1)
            stopIdx = int(file_line[2*sequenceIdx + 2]) // (args.skip+1)
            torch.save(frameCollection[startIdx : stopIdx], os.path.join(classDir, '{:02}_{:02}.pyt'.format(nVideos, (sequenceIdx + 1))))

    pbar1.update(1)
    file_line = f.readline()

dataList.close()
f.close()
pbar1.close()
print('\n\n')
