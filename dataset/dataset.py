from torch.utils.data import Dataset
import torch
import numpy as np
from numpy.random import randint, choice
import glob

class bearing_dataset(Dataset):
    def __init__(
        self, path: str = './data/Image/train',
        class_lst: list = ['Ball', 'Normal', 'Inner_break', 'Outer_break'],
        train: bool = True
    ):
        self.train = train

        length = 0
        trainDB = [[] for _ in range(len(class_lst))]
        total_lst = []
        label_lst = []
        for i, Class in enumerate(class_lst):
            class_path = f'{path}/{Class}/*.npy'
            file_lst = glob.glob(class_path)
            trainDB[i] += file_lst
            total_lst += file_lst
            label_lst += [i for _ in range(len(file_lst))]
            length += len(file_lst)

        self.trainDB = trainDB
        self.class_label = [i for i in range(len(class_lst))]
        self.total_lst = total_lst
        self.label_lst = label_lst
        self.length = length
        self.n_classes = len(class_lst)
    
    def __getitem__(self, index: int):
        if self.train:
            class1, class2 = choice(self.class_label, size = 2, replace = False)
            anchor, positive = choice(self.trainDB[class1], size = 2, replace = False)
            anchor, positive = torch.as_tensor(np.load(anchor), dtype = torch.float32), torch.as_tensor(np.load(positive), dtype = torch.float32)
            negative = torch.as_tensor(np.load(choice(self.trainDB[class2], size = 1)[0]), dtype = torch.float32)
            class1 = torch.as_tensor(class1, dtype = torch.int8)
            # class1 is the label of anchor and positive
            return anchor, positive, negative, class1, class2

        else:
            arr = np.load(self.total_lst[index])
            label = self.label_lst[index]
            return arr, label

    def __len__(self):
        return self.length

    def train(self):
        self.train = True
    
    def eval(self):
        self.train = False