import torch.utils.data as data

import torch
import os
import os.path


def find_classes(root_dir):
    classes = os.listdir(root_dir)
    classes.sort()
    for item in classes:
        if not os.path.isdir(os.path.join(root_dir, item)):
            classes.remove(item)

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root_dir, class_to_idx):
    tensors = []
    for target in os.listdir(root_dir):
        d = os.path.join(root_dir, target)
        if not os.path.isdir(d):
            continue

        for filename in os.listdir(d):
            if filename.endswith('.pyt'):
                path = '{0}/{1}'.format(target, filename)
                item = (path, class_to_idx[target])
                tensors.append(item)

    return tensors


def default_loader(path):
    return torch.load(path)


class TensorFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        """

        :param root: path to the root directory of data
        :type root: str
        :param transform: input transform
        :type transform: torch-vision transforms
        :param target_transform: target transform
        :type target_transform: torch-vision transforms
        :param loader: type of data loader
        :type loader: function
        """
        classes, class_to_idx = find_classes(root)
        tensors = make_dataset(root, class_to_idx)

        self.root = root
        self.tensors = tensors
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # One hot encoding of label
        self.class_map = torch.zeros(len(classes), len(classes))
        for i in range(len(classes)):
            self.class_map[i][i] = 1

    def __getitem__(self, index):
        path, target = self.tensors[index]
        input_tensor = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return input_tensor, self.class_map[target]

    def __len__(self):
        return len(self.tensors)
