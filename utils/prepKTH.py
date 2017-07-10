import numpy as np
import os
import torch
import cv2
from skimage.transform import resize
from tqdm import trange
from argparse import ArgumentParser

parser = ArgumentParser(description='e-Lab Gesture Recognition Script')
_ = parser.add_argument
_('--data',      type=str, default='/media/HDD2/Models/', help='dataset location')
_('--save',      type=str, default='/media/HDD2/Models/', help='folder to save outputs')
_('--startfrom', type=int, default=0, help='Start from this video')
_('--skip',      type=int, default=0, help='# of frames to skip')
_('--dim',       type=int, default=(160, 120), nargs=2, help='input image dimension as tuple (WxH)', metavar=('W', 'H'))

print('\033[0;0f\033[0J')
args = parser.parse_args()

f = open('tvt_split.txt', 'r')

# maps personID to tvt set
split_dict = {}
for i in range(3):
    file_line = f.readline().rstrip().split(" ")
    for j in range(len(file_line) - 1):
        split_dict.update({file_line[j+1]: file_line[0]})

classmap = {'boxing': 1,
            'handclapping': 2,
            'handwaving': 3,
            'jogging': 4,
            'running': 5,
            'walking': 6}

iWidth = args.dim[0]
iHeight = args.dim[1]

rootDirSave = args.save
if not os.path.exists(rootDirSave):
    os.makedirs(rootDirSave)
    os.makedirs(rootDirSave + '/Train')
    os.makedirs(rootDirSave + '/Val')
    os.makedirs(rootDirSave + '/Test')

rootDirLoad = args.data
pbar1 = trange(599, ncols=100, position=0, desc='Overall progress      ')
classes = []

dataList = open(rootDirSave + "/dataList.txt", "w")
dataList.write('{:8} {:12} {:12} {:12} {:12}'.format('Class ID', 'Class name', 'Mode', '# of Frames', 'Filename'))
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
        cap = cv2.VideoCapture(videoname)
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if nVideos >= args.startfrom:
            nVideos += 1
            pbar2 = trange(nFrames, ncols=100, position=2, desc='Video progress        ')

            frameCount = 0
            # Tensor to save all the fames of a video
            frameCollection = torch.FloatTensor()
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    if frameCollection.size(0) == 0:
                        print(videoname)
                    break

                frameCount += 1
                if (frameCount % (args.skip+1)) == 0:
                    # Original resolution -> desired resolution
                    if args.dim == (160, 120):
                        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        current_frame = cv2.resize(frame, args.dim, interpolation=cv2.INTER_AREA)

                    if frameCount > 1:
                        flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                        frame_temp = torch.unsqueeze(torch.from_numpy(current_frame), 0).float()
                        flow_temp = torch.from_numpy(np.transpose(flow, (2, 0, 1))).float()
                        gray_opt = torch.cat((frame_temp, flow_temp), 0)

                        # Force the values to lie between [0, 1] for each channel
                        for i in range(len(gray_opt)):
                            gray_opt[i] = (gray_opt[i] - gray_opt[i].min())/(gray_opt[i].max() - gray_opt[i].min())

                        frameCollection = torch.cat((frameCollection,
                                          torch.unsqueeze(gray_opt, 0)), 0)
                        # cv2.imshow('Image', flow[:,:,0])
                        # cv2.waitKey(0)

                    prev_frame = current_frame
                    # (height, width, channel) -> (channel, height, width)

                # cv2.destroyAllWindows()
                pbar2.update(1)

            pbar2.close()
            # Split video into chunks and save them
            for sequenceIdx in range((len(file_line)-1)//2):
                startIdx = (int(file_line[2*sequenceIdx + 1])-1) // (args.skip+1)
                stopIdx = int(file_line[2*sequenceIdx + 2]) // (args.skip+1)
                torch.save(frameCollection[startIdx : stopIdx], os.path.join(classDir, '{:02}_{:02}.pyt'.format(nVideos, (sequenceIdx + 1))))
                dataList.write('\n{:8} {:12} {:12} {:<12} {:02}_{:02}.pyt'
                        .format(classmap[subDir], subDir, split_dict[personID], (stopIdx - startIdx), nVideos, (sequenceIdx + 1)))

    pbar1.update(1)
    file_line = f.readline()

dataList.close()
f.close()
pbar1.close()
print('\n\n')