import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


class CelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        '''read attr.txt, select interested attributes(4), split train and test dataset, save
        train_filenames and train_labels(successively), and test_filenames[1.jpg 2.jpg ...] 
        and test_labels[[0 1 1] [1 0 1] ...]'''
    
        '''read attributes'''
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        '''read values for selected attributes and change -1 to 0'''
        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)

            if (i+1) < 2000:
            '''the first 1999 images for test, others for train'''
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

    def __getitem__(self, index):
        '''use index to return an item'''
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
            label = self.train_labels[index]
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
            label = self.test_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data


def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train'):
    """Build and return data loader."""
    '''return a data_loader, should contain dataset, batch_size and shuffle attr
    dataset in a class contains ???'''
    if mode == 'train':
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == 'CelebA':
        dataset = CelebDataset(image_path, metadata_path, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_path, transform)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
