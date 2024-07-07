import os
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from .brats import BratsDataset
from .ixi import IxiDataset


def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform


def create_dataloader(opt, mode):
    assert opt.dataset in ['brats', 'ixi'], 'Invalid name of dataset!'
    shuffle = (mode == 'train')
    if opt.dataset == 'brats':
        dataset = BratsDataset(os.path.join(opt.dataroot, 'BraTS2020'), get_transforms(), mode=mode, pad=(opt.model == 'pix2pix'))
    elif opt.dataset == 'ixi':
        dataset = IxiDataset(os.path.join(opt.dataroot, 'IXI'), get_transforms(), mode=mode)

    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle), len(dataset)