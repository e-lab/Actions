import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Logger():
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        error_file = os.path.join(root_dir, 'error.log')

        error_log = open(error_file, 'w')
        error_log.write('{:10} {:10}'.format('Train Error', 'Test Error'))
        error_log.write('\n{:-<20}'.format(''))
        error_log.close()

        self.total_train_error = list()
        self.total_test_error = list()

    def update_error_log(self, train_error, test_error):
        error_file = os.path.join(self.root_dir, 'error.log')
        plot_file = os.path.join(self.root_dir, 'error_plot.png')

        f = open(error_file, 'a')
        f.write('\n{:.6f} {:.6f}'.format(train_error, test_error))
        f.close()

        self.total_train_error.append(train_error)
        self.total_test_error.append(test_error)
        plt.plot(range(len(self.total_train_error)), self.total_train_error, label='Train error')
        plt.plot(range(len(self.total_test_error)), self.total_test_error, label='Test error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Train/Test Error')
        plt.grid(True)
        plt.savefig(plot_file)
        # plt.show()


    def update_conf_log(self, conf_train, conf_test):
        conf_file = os.path.join(self.root_dir, 'conf.log')

        f = open(conf_file, 'w')
        f.write('{:-<20}'.format(''))
        f.write('\nTrain:')
        f.write('\n{:-<20}'.format(''))
        f.write('\n{}'.format(conf_train.classwise_accuracy))
        f.write('\n{:-<20}'.format(''))
        for value in conf_train.classwise_metric.items():
            f.write('\n{}: {}'.format(value[0], value[1]))
        f.write('\n{:-<20}'.format(''))
        for value in conf_train.metric.items():
            f.write('\n{}: {}'.format(value[0], value[1]))
        f.write('\nGlobal Accuracy: {}'.format(conf_train.accuracy))
        f.write('\n{:-<20}'.format(''))
        f.write('\n{:-<20}'.format(''))
        f.write('\nTest:')
        f.write('\n{:-<20}'.format(''))
        f.write('\n{}'.format(conf_test.classwise_accuracy))
        f.write('\n{:-<20}'.format(''))
        for value in conf_test.classwise_metric.items():
            f.write('\n{}: {}'.format(value[0], value[1]))
        f.write('\n{:-<20}'.format(''))
        for value in conf_test.metric.items():
            f.write('\n{}: {}'.format(value[0], value[1]))
        f.write('\nGlobal Accuracy: {}'.format(conf_test.accuracy))
        f.write('\n{:-<20}'.format(''))
        f.close()
