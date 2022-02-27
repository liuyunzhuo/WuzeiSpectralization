import torch
from torch.utils.tensorboard import SummaryWriter

from utils.data_loading import WuZeiDataset
from utils.stretch_cnhw_image import stretch_cnhw_image
from torch.utils.data import DataLoader, random_split
from models import UNet
import os
import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from evaluate import evaluate


class PSFUNet(nn.Module):
    def __init__(self, n_channels, n_classes, psfs):
        super(PSFUNet, self).__init__()
        self.bg = BLurGenerator(psfs=psfs)
        self.unet = UNet(n_channels=n_channels,
                         n_classes=n_classes,
                         bilinear=False)

    def forward(self, x):
        return self.unet(self.bg(x))


class BLurGenerator(nn.Module):
    """spectral_cube ==> blur"""

    def __init__(self, psfs):
        super().__init__()
        self.convlist = nn.ModuleList()
        for i in range(len(psfs)):
            psf = torch.from_numpy(psfs[i][0])
            psf = psf.to(dtype=torch.float32)
            conv = nn.Conv2d(1, 1, psf.size(1), padding=psf.size(1) // 2)
            psf = torch.unsqueeze(psf, 0)
            psf = torch.unsqueeze(psf, 0)
            conv.weight.data = psf
            self.convlist.append(conv)

    def forward(self, x):
        for i in range(len(self.convlist)):
            conv = self.convlist[i]
            spec = x[0, i, :, :]
            spec = torch.unsqueeze(spec, 0)
            spec = torch.unsqueeze(spec, 0)
            if i == 0:
                y = conv(spec)
            else:
                y = torch.add(y, conv(spec))
        return y
