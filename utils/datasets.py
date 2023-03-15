import os
import torch
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


class Datasets(datasets.VisionDataset):

    def __init__(self, args, source=False, transform=None, target_transform=None):
        super().__init__(args.root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.common_class = args.common_class
        self.source_private_class = args.source_private_class
        self.target_private_class = args.target_private_class
        self.source_class = self.common_class + self.source_private_class

        # add '/' in the last of root if necessary
        if self.root[-1] is not '/':
            self.root += '/'

        # create total root
        if source:
            file_name = self.root + args.source + '.txt'
        else:
            file_name = self.root + args.target + '.txt'

        self.samples = self.get_data_list(file_name, source=source)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

    def create_label_set(self):
        label = torch.tensor([self.samples[i][1] for i in range(len(self.samples))])
        return label

    def get_data_list(self, file_name, source=False):
        with open(file_name, "r") as f:
            data_list = []
            if source:
                class_set = [i for i in range(self.source_class)]
            else:
                class_set = [i for i in range(self.common_class)] + [self.source_class]
            for line in f.readlines():
                split_line = line.split()
                target = int(split_line[-1])

                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)

                # uniform unknown sample label
                if not source and target >= self.source_class:
                    target = self.source_class

                if target in class_set:
                    data_list.append((path, target))
        return data_list


