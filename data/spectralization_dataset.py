"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform, get_params
# from data.image_folder import make_dataset
from os import listdir
from os.path import splitext
from pathlib import Path
import os
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class SpectralizationDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        in_channel = [550]
        out_channel = list(range(420, 721, 10))
        parser.add_argument('--in_channel', type=list, default=in_channel, help='Wavelength in_channel')
        parser.add_argument('--out_channel', type=list, default=out_channel, help='Wavelength in_channel')
        parser.set_defaults(input_nc=len(in_channel), output_nc=len(out_channel), preprocess='crop')
        return parser

    """A template dataset class for you to implement custom datasets."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        self.root_dir = Path(os.path.join(opt.dataroot, opt.phase))
        self.in_channel = opt.in_channel
        self.out_channel = opt.out_channel
        self.img_dir = os.path.join(self.root_dir, "模糊图像")
        self.label_dir = os.path.join(self.root_dir, "光谱图像")
        self.ids = [splitext(file)[0] for file in listdir(self.img_dir) if not file.startswith('.')]
        self.transform = get_transform(self.opt, convert=False)

        # 加载a光谱幅照度修正参数
        a_full = loadmat(os.path.join(self.root_dir, 'a_326.mat'))['a']
        sample_dir = os.path.join(self.label_dir, listdir(self.label_dir)[0])
        wavelength_full = sorted([int(i[0:-2])
                                  for i in [splitext(file)[0]
                                            for file in listdir(sample_dir) if not file.startswith('.')]])
        self.a = np.asarray([a_full[wavelength_full.index(i)] for i in self.out_channel], dtype=np.float64)
        self.a = torch.from_numpy(self.a)

        # max_radiance
        self.max_radiance = np.asarray(loadmat(os.path.join(self.root_dir, 'a_326.mat'))['maxval'], dtype=np.float64)
        self.max_radiance = torch.from_numpy(self.max_radiance)

        # 后缀名
        ext_full = list(set(i
                            for i in [splitext(file)[1]
                                      for file in listdir(sample_dir) if not file.startswith('.')]))
        self.ext = ext_full[0]

        # 给ids按编号大小排序
        self.ids = [int(i) for i in self.ids]
        self.ids.sort()
        self.ids = [str(i) for i in self.ids]

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img_list, is_label):
        for i in range(len(pil_img_list)):
            img = pil_img_list[i]
            img = self.transform(img)
            img = torch.as_tensor(np.asarray(img) / 65535.0)
            img = torch.unsqueeze(img, 0)
            if is_label:
                img = torch.div(torch.mul(img, self.a[i]), self.max_radiance)
            if i is 0:
                data = img
            else:
                data = torch.cat((data, img), 0)
        data = data.to(torch.float32)
        return data

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the L channel of an image
            B (tensor) - - the ab channels of the same image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        name = self.ids[idx]
        img_files = [os.path.join(self.img_dir, name, str(i) + 'nm' + self.ext)
                     for i in self.in_channel]
        label_files = [os.path.join(self.label_dir, name, str(i) + 'nm' + self.ext)
                       for i in self.out_channel]

        img = []
        label = []
        for i in range(len(img_files)):
            img.append(self.load(img_files[i]))
        for i in range(len(label_files)):
            label.append(self.load(label_files[i]))

        img = self.preprocess(img, is_label=False)
        label = self.preprocess(label, is_label=True)

        A_paths = os.path.join(self.img_dir, name)
        B_paths = os.path.join(self.label_dir, name)
        return {'A': img,
                'B': label,
                'A_paths': A_paths,
                'B_paths': B_paths,
                'max_radiance': self.max_radiance}

    def print_info(self):
        img_and_label = self[0]
        img = img_and_label['image']
        label = img_and_label['label']
        print('-----------------------------------------------------------------')
        print('模糊图像Channel,Height,Width: {}'.format(img.shape))
        print('模糊图像dtype: {}'.format(img.dtype))
        print('模糊图像值范围: {} ～ {}'.format(img.min(), img.max()))
        print('真实光谱图像Channel,Height,Width: {}'.format(label.shape))
        print('真实光谱图像dtype: {}'.format(label.dtype))
        print('真实光谱图像值范围: {} ～ {}'.format(label.min(), label.max()))
        print('-----------------------------------------------------------------')

    def get_max_radiance(self):
        return np.asarray(self.max_radiance)
