rnn_type = 'rnn'        # rnn | lstm | gru
filepath = '/media/HDD1/Models/Actions/abhi/Models/rnn/'
import matplotlib.pyplot as plt


def read_file(file_to_read, image_name, fig_xlabel, fig_title):
    file = open(file_to_read, 'r')
    file.readline()

    error_value = list()
    for line in file:
        error_value.append(int(line))

    plt.xlabel(fig_xlabel)
    plt.ylabel('Error')
    plt.title(fig_title)
    plt.grid(True)
    plt.savefig(image_name)

filename = ['error', 'error_bw']
fig_xlabel_list = ['epochs', 'frames']
fig_title_list = ['Error every epoch using ' + rnn_type.upper(), 'Error every frame using ' + rnn_type().upper()]
for i in range(2):
    file_to_read = filepath + filename[i] + '.txt'
    image_name = filepath + filename[i] + '.png'
    read_file(file_to_read, image_name, fig_xlabel_list[i], fig_title_list[i])
