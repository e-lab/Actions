import numpy as np
import os
import torch
from skimage.io import imsave
from skvideo.io import FFmpegReader
from skimage.transform import resize
from tqdm import trange
import warnings

iHeight = 128
iWidth = 128

rootDirSave = "./dataSave/"
if not os.path.exists(rootDirSave):
    os.makedirs(rootDirSave)

rootDirLoad = "./data/"
subDirList = os.listdir(rootDirLoad)
pbar1 = trange(len(subDirList), ncols=150, position=0, desc='Overall progress      ')
classes = []

dataList = open("./dataList.txt", "w")
dataList.write('{:12}{:12}{:12}{:12}'.format('Class Index', 'Class name', '# of Frames', 'Filename'))
dataList.write('\n{:-<48}'.format(''))

nClasses = 0
for subDir in subDirList:                               # Walk through all the classes
    pbar1.update(1)
    nClasses += 1

    if os.path.isdir(os.path.join(rootDirLoad, subDir)):  # Check if it is a folder
        classes.append(subDir)                          # Create a path
        files = os.listdir(os.path.join(rootDirLoad, subDir))
        nVideos = 0                                     # Videos in class X

        pbar2 = trange(len(files), ncols=150, position=2, desc='Within-class progress ')

        for file in files:                              # Get all the videos
            if file.lower().endswith('.avi') or file.lower().endswith('.mp4'):
                filename = os.path.join(rootDirLoad, subDir, file)
                reader = FFmpegReader(filename)
                nFrames = reader.getShape()[0]

                nVideos += 1
                dataList.write('\n{:12}{:12}{:12}{:02}.pyt'.format(nClasses, subDir, str(nFrames), nVideos))

                # Create class directories if they do not exist
                classDir = os.path.join(rootDirSave, subDir)
                if not os.path.exists(classDir):
                    os.makedirs(classDir)

                pbar3 = trange(nFrames, ncols=150, position=4, desc='Video progress        ')
                frameCount = 0
                # Tensor to save all the frames of a video
                frameCollection = torch.FloatTensor(nFrames, 3, iHeight, iWidth)
                for frame in reader.nextFrame():        # Garb each frame
                    # Original resolution -> desired resolution
                    tempImg = resize(frame, (iHeight, iWidth))

                    frameCount += 1

                    imgName = '{:02}_{:04}.png'.format(nVideos, frameCount)
                    # Ignore warning regarding float64 being converted into uint8
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        imsave(os.path.join(rootDirSave, subDir, imgName), tempImg)

                    # (height, width, channel) -> (channel, height, width)
                    frameCollection[frameCount-1] = torch.from_numpy(np.transpose(tempImg, (2, 0, 1)))
                    pbar3.update(1)

                # Save tensors class wise
                torch.save(frameCollection, os.path.join(classDir, '{:02}.pyt'.format(nVideos)))

                pbar3.close()
            pbar2.update(1)
        pbar2.close()


dataList.close()
pbar1.close()
print('\n\n\n')
