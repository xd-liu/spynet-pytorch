from os.path import exists, join, splitext
import numpy
from PIL import Image
from torch.utils.data import Dataset
import os
import torch


class Mdata(Dataset):
    # Disparity annotations are transformed into flow format;
    # Sparse annotations possess an extra dimension as the valid mask;
    def __init__(self,
                 data_root,
                 data_list):
        self.data_root = data_root
        self.data_list = self.read_lists(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_first_name = os.path.join(self.data_root, self.data_list[index][0])
        img_second_name = os.path.join(self.data_root, self.data_list[index][1])
        tenFirst = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenSecond = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        return tenFirst.view(3, 720, 1280), tenSecond.view(3, 720, 1280)

    def read_lists(self, data_list):
        assert exists(data_list)
        samples = [line.strip().split(' ') for line in open(data_list, 'r')]
        return samples
