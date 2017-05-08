import numpy as np
import cv2
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
_('--data',  type=str, default='/media/HDD1/Datasets/HMDB/', help='dataset location')
_('--save',  type=str, default='/media/HDD2/Models/', help='folder to save outputs')
_('--skip',  type=int, default=1, help='# of frames to skip')
_('--dim',   type=int, default=(176, 120), nargs=2, help='input image dimension as tuple (WxH)', metavar=('W', 'H'))

print('\033[0;0f\033[0J')
args = parser.parse_args()

iWidth = args.dim[0]
iHeight = args.dim[1]

getImgs = False
rootDirSave = args.save
if not os.path.exists(rootDirSave):
    os.makedirs(rootDirSave)

dataList = open(rootDirSave + "/dataList.txt", "w")
dataList.write('{:8} {:8} {:12} {:12} {:12}'.format('Class #', 'Type', 'Class name', '# of Frames', 'Filename'))
dataList.write('\n{:-<55}'.format(''))

rootDirLoad = args.data + '/hmdb51/'
subDirList = os.listdir(rootDirLoad)

data_fname_dir = os.path.join(args.data, 'tvt_split1')
data_fname_list = os.listdir(data_fname_dir)    # Get all the list for files in tvt_split dir
pbar1 = trange(len(data_fname_list), position=0, desc='Overall Progress')

tvt = ['val', 'train', 'test']

nClasses = 0
for data_fname in data_fname_list:
    pbar1.update(1)
    nClasses += 1
    classname = data_fname.split('_test')       # Get class name
    subDir = os.path.join(rootDirLoad, classname[0])

    # Create class directories if they do not exist
    classDir = list()
    for data_type in tvt:
        classDir.append(os.path.join(rootDirSave, data_type, classname[0]))
        if not os.path.exists(classDir[len(classDir)-1]):
            os.makedirs(classDir[len(classDir)-1])

    data_fname = os.path.join(data_fname_dir, data_fname)
    with open(data_fname) as f:
        filename_list = f.readlines()
        pbar2 = trange(len(filename_list), position=2, desc='Within-class progress ')
        nVideos = 0
        for content in filename_list:
            pbar2.update(1)
            content = content.split()

            if content[0].lower().endswith('.avi') or content[0].lower().endswith('.mp4'):
                # Get the filename with full path
                filename = os.path.join(subDir, content[0])
                cap = cv2.VideoCapture(filename)
                nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                #reader = FFmpegReader(filename)
                #nFrames = reader.getShape()[0]

                nVideos += 1
                dataList.write('\n{:<8} {:8} {:12} {:<12} {:02}.pyt'.format(nClasses, content[1], classname[0], nFrames, nVideos))


                pbar3 = trange(nFrames, position=4, desc='Video progress        ')
                frameCount = [0, 0, 0]
                # Tensor to save all the frames of a video
                frameCollection = torch.FloatTensor()
                while(cap.isOpened()):
                #for frame in reader.nextFrame():        # Garb each frame
                    ret, frame = cap.read()
                    frameCount[int(content[1])] += 1
                    if not ret:
                        if frameCollection.size(0) == 0:
                            print(filename)
                        break

                    if (frameCount[int(content[1])] % args.skip) == 0:
                        # Original resolution -> desired resolution
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        tempImg = cv2.resize(frame, args.dim, interpolation=cv2.INTER_AREA)


                        if getImgs:
                            imgName = '{:02}_{:04}.png'.format(nVideos, frameCount[int(content[1])])
                            # Ignore warning regarding float64 being converted into uint8
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                imsave(os.path.join(rootDirSave, subDir, imgName), tempImg)

                        # (height, width, channel) -> (channel, height, width)
                        frameCollection = torch.cat((frameCollection,
                                          torch.unsqueeze(
                                          torch.from_numpy(
                                          np.transpose(tempImg, (2, 0, 1))).float(), 0)), 0)
                    pbar3.update(1)

                # Save tensors class wise
                torch.save(frameCollection, os.path.join(classDir[int(content[1])], '{:02}.pyt'.format(nVideos)))

                cap.release()
                pbar3.close()
        pbar2.close()

dataList.close()
pbar1.close()
print('\n\n\n')


"""
nClasses = 0
for subDir in subDirList:                               # Walk through all the classes
    pbar1.update(1)
    nClasses += 1

    if os.path.isdir(os.path.join(rootDirLoad, subDir)):  # Check if it is a folder
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
"""
