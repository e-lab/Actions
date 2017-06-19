import numpy as np

class ConfusionMatrix():
    def __init__(self, n_classes):

        super(ConfusionMatrix, self).__init__()
        self.n_classes = n_classes
        self.classwise_accuracy = np.zeros(n_classes)
        self.metric = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}   # Dictionary to hold all 4 metric values


    def generate_matrix(self, arr):
        total_elements = arr.sum()

        for i in range(self.n_classes):
            tp = arr[i, i]
            fp = arr[:, i].sum() - tp
            fn = arr[i, :].sum() - tp
            tn = total_elements - (tp + fp + fn)

            self.classwise_accuracy[i] = ((tp+tn) / total_elements)*100

            self.metric['tp'] += tp
            self.metric['tn'] += tn
            self.metric['fp'] += fp
            self.metric['fn'] += fn
