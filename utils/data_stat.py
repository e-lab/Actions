filepath = '/media/HDD1/Models/Actions/cache/dataList.txt'
import matplotlib.pyplot as plt

file = open(filepath, 'r')
file.readline()
file.readline()

n_frames = list()       # number of frames in each video
n_videos = list()       # number of videos in each class
for line in file:
    words = line.split()

    n_frames.append(words[2])
    n_videos.asppend(words[0])

file.close()
plt.xlabel('Classes')
plt.ylabel('# of videos')
plt.title('Data distribution over classes')
plt.axis([0, 12, 0, 200])
plt.grid(True)
