import numpy as np

class ConfusionMatrix():
    def __init__(self, n_classes):

        super(ConfusionMatrix, self).__init__()
        self.n_classes = n_classes
        self.classwise_accuracy = np.zeros(n_classes)
        self.metric = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}   # Dictionary to hold all 4 metric values
        self.classwise_metric = {}
        for i in range(n_classes):
            self.classwise_metric[i] = self.metric.copy()


    def generate_matrix(self, arr):
        total_elements = arr.sum()

        self.metric['tp'] = 0
        self.metric['tn'] = 0
        self.metric['fp'] = 0
        self.metric['fn'] = 0
        self.accuracy = 0

        for i in range(self.n_classes):
            tp = arr[i, i]
            fp = arr[:, i].sum() - tp
            fn = arr[i, :].sum() - tp
            tn = total_elements - (tp + fp + fn)

            if (tp+fp) == 0:
                self.classwise_accuracy[i] = 0
            else:
                self.classwise_accuracy[i] = (tp / (tp+fp)) * 100

            self.classwise_metric[i]['tp'] = tp
            self.classwise_metric[i]['tn'] = tn
            self.classwise_metric[i]['fp'] = fp
            self.classwise_metric[i]['fn'] = fn

            self.metric['tp'] += tp
            self.metric['tn'] += tn
            self.metric['fp'] += fp
            self.metric['fn'] += fn

        self.accuracy = (self.metric['tp'] / (self.metric['tp']+self.metric['fp']))*100
