import numpy as np
import os
import torch
from skimage.io import imsave
from skvideo.io import FFmpegReader
from skimage.transform import resize
from tqdm import trange
from argparse import ArgumentParser
import warnings

parser = ArgumentParser(description='e-Lab Gesture Recognition Script')
_ = parser.add_argument
_('--data',  type=str, default='/media/HDD2/Models/', help='dataset location')
_('--save',  type=str, default='/media/HDD2/Models/', help='folder to save outputs')
_('--skip',  type=int, default=5, help='# of frames to skip')
_('--dim',   type=int, default=(160, 120), nargs=2, help='input image dimension as tuple (WxH)', metavar=('W', 'H'))

print('\033[0;0f\033[0J')
args = parser.parse_args()

iWidth = args.dim[0]
iHeight = args.dim[1]

getImgs = False
rootDirSave = args.save
if not os.path.exists(rootDirSave):
    os.makedirs(rootDirSave)

rootDirLoad = args.data
subDirList = os.listdir(rootDirLoad)
pbar1 = trange(len(subDirList), ncols=100, position=0, desc='Overall progress      ')
classes = []

dataList = open(rootDirSave + "/dataList.txt", "w")
dataList.write('{:8} {:12} {:12} {:12}'.format('Class #', 'Class name', '# of Frames', 'Filename'))
dataList.write('\n{:-<47}'.format(''))

nClasses = 0
for subDir in subDirList:                               # Walk through all the classes
    pbar1.update(1)
    nClasses += 1

    if os.path.isdir(os.path.join(rootDirLoad, subDir)):  # Check if it is a folder
        classes.append(subDir)                          # Create a path
        files = os.listdir(os.path.join(rootDirLoad, subDir))
        nVideos = 0                                     # Videos in class X

        pbar2 = trange(len(files), ncols=100, position=2, desc='Within-class progress ')

        for file in files:                              # Get all the videos
            if file.lower().endswith('.avi') or file.lower().endswith('.mp4'):
                filename = os.path.join(rootDirLoad, subDir, file)
                reader = FFmpegReader(filename)
                nFrames = reader.getShape()[0]

                nVideos += 1
                dataList.write('\n{:<8} {:12} {:<12} {:02}.pyt'.format(nClasses, subDir, nFrames, nVideos))

                # Create class directories if they do not exist
                classDir = os.path.join(rootDirSave, subDir)
                if not os.path.exists(classDir):
                    os.makedirs(classDir)

                pbar3 = trange(nFrames, ncols=100, position=4, desc='Video progress        ')
                frameCount = 0
                # Tensor to save all the frames of a video
                frameCollection = torch.FloatTensor(nFrames//args.skip, 3, iHeight, iWidth)
                for frame in reader.nextFrame():        # Garb each frame
                    frameCount += 1
                    if (frameCount % args.skip) == 0:
                        # Original resolution -> desired resolution
                        tempImg = resize(frame, (iHeight, iWidth))


                        if getImgs:
                            imgName = '{:02}_{:04}.png'.format(nVideos, frameCount)
                            # Ignore warning regarding float64 being converted into uint8
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                imsave(os.path.join(rootDirSave, subDir, imgName), tempImg)

                        # (height, width, channel) -> (channel, height, width)
                        frameCollection[frameCount//args.skip-1] = torch.from_numpy(np.transpose(tempImg, (2, 0, 1)))
                    pbar3.update(1)

                # Save tensors class wise
                torch.save(frameCollection, os.path.join(classDir, '{:02}.pyt'.format(nVideos)))

                pbar3.close()
            pbar2.update(1)
        pbar2.close()


dataList.close()
pbar1.close()
print('\n\n\n')
