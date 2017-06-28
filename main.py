import torch
from opts import get_args           # Get all the input arguments
args = get_args()                   # Holds all the input argument

torch.manual_seed(args.seed)        # Set random seed manually

# Python imports
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import math
from subprocess import call

# Local imports
from Models.model2 import ModelDef
from utils.confusionMatrix import ConfusionMatrix
from train import train as trainClass
from test import test as testClass
import generateData

print('\033[0;0f\033[0J')
# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

#ModelDef = getattr(__import__(args.model, fromlist=['ModelDef']), 'ModelsDef')  # Get the model definition

if not os.path.exists(args.save):
    os.makedirs(args.save)

# Save arguements used for training
args_log = open(args.save + '/args.log', 'w')
args_log.write(str(args))
args_log.close()
# Save model definiton script
call(["cp", args.model, args.save])

seq_len = args.seq
data_dir = args.data
i_width, i_height = args.dim

if torch.cuda.is_available():
    if not args.cuda:
        print(CP_G + "WARNING: You have a CUDA device, so you should probably run with --cuda" + CP_C)
    else:
        torch.cuda.set_device(args.devID)
        torch.cuda.manual_seed(args.seed)
        print("\033[41mGPU({:}) is being used!!!{}".format(torch.cuda.current_device(), CP_C))

# Acquire dataset loader object
data_obj_train = generateData.TensorFolder(root=data_dir + '/Train/')
data_loader_train = DataLoader(data_obj_train, batch_size=args.bs, shuffle=True, num_workers=args.workers)

data_obj_test = generateData.TensorFolder(root=data_dir + '/Test/')
data_loader_test = DataLoader(data_obj_test, batch_size=args.bs, shuffle=True, num_workers=args.workers)

n_classes = len(data_obj_train.classes)
data_len_train = len(data_obj_train)
data_len_test = len(data_obj_test)

log_classes = open(args.save + '/categories.txt', 'w')
for i in range(n_classes):
    log_classes.write(str(i+1) + ': ' + data_obj_train.classes[i] + '\n')
log_classes.close()


# Load model
model = ModelDef([512, n_classes], args.rnn_type)        # Network architecture is stored here
print(model)

if args.cuda:
    model.cuda()
# model = nn.DataParallel(net, device_ids=[0,1,2,3])

confusion_matrix_train = ConfusionMatrix(n_classes)
confusion_matrix_test = ConfusionMatrix(n_classes)


def main():
    print("\n\033[94m\033[1me-Lab Gesture Recognition Training Script\033[0m\n")
    prev_accuracy = -1

    # Log batchwise error
    logger = open(args.save + '/error.log', 'w')
    logger.write('{:10} {:10}'.format('Train Error', 'Test Error'))
    logger.write('\n{:-<20}'.format(''))
    logger.close()

    train = trainClass(model, data_loader_train, data_len_train, n_classes, args)
    test = testClass(model, data_loader_test, data_len_test, n_classes, args)
    for epoch in range(1, args.epochs):
        total_train_error = train.forward()
        confusion_matrix_train.generate_matrix(train.mtr.value())

        total_test_error = test.forward()
        confusion_matrix_test.generate_matrix(test.mtr.value())
        print('Class wise accuracy:')
        print(confusion_matrix_test.classwise_accuracy)

        print('{}{:-<50}{}'.format(CP_R, '', CP_C))
        print('{}Epoch #: {}{:03}'
                .format(CP_B, CP_C, epoch))
        print('{}Training Error: {}{:.6f} | {}Testing Error: {}{:.6f}'.format(
            CP_B, CP_C, total_train_error, CP_B, CP_C, total_test_error))

        logger = open(args.save + '/error.log', 'a')
        logger.write('\n{:.6f} {:.6f}'.format(total_train_error, total_test_error))
        logger.close()
        accuracy = confusion_matrix_test.accuracy
        print(accuracy)
        # Save weights and model definition
        if accuracy >= prev_accuracy:
            prev_accuracy = accuracy
            print(CP_G + "Saving model!!!" + CP_C)
            print('{}{:-<50}{}\n'.format(CP_R, '', CP_C))
            torch.save(model.state_dict(), args.save + "/model.pt")
            torch.save(ModelDef, args.save + "/modelDef.pt")
            conf_logger = open(args.save + '/conf.log', 'w')
            conf_logger.write('{:-<20}'.format(''))
            conf_logger.write('\nTrain:')
            conf_logger.write('\n{:-<20}'.format(''))
            conf_logger.write('\n{}'.format(confusion_matrix_train.classwise_accuracy))
            conf_logger.write('\n{:-<20}'.format(''))
            for value in confusion_matrix_train.classwise_metric.items():
                conf_logger.write('\n{}: {}'.format(value[0], value[1]))
            conf_logger.write('\n{:-<20}'.format(''))
            for value in confusion_matrix_train.metric.items():
                conf_logger.write('\n{}: {}'.format(value[0], value[1]))
            conf_logger.write('\nGlobal Accuracy: {}'.format(confusion_matrix_train.accuracy))
            conf_logger.write('\n{:-<20}'.format(''))
            conf_logger.write('\n{:-<20}'.format(''))
            conf_logger.write('\nTest:')
            conf_logger.write('\n{:-<20}'.format(''))
            conf_logger.write('\n{}'.format(confusion_matrix_test.classwise_accuracy))
            conf_logger.write('\n{:-<20}'.format(''))
            for value in confusion_matrix_test.classwise_metric.items():
                conf_logger.write('\n{}: {}'.format(value[0], value[1]))
            conf_logger.write('\n{:-<20}'.format(''))
            for value in confusion_matrix_test.metric.items():
                conf_logger.write('\n{}: {}'.format(value[0], value[1]))
            conf_logger.write('\nGlobal Accuracy: {}'.format(confusion_matrix_test.accuracy))
            conf_logger.write('\n{:-<20}'.format(''))
            conf_logger.close()


    # train.logger_bw.close()
    # test.logger_bw.close()


if __name__ == "__main__":
    main()
