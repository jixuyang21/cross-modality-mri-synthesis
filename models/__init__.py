from .unet import Unet
from .pix2pix import Pix2pix
from .pgan import pGAN
from .cgan import cGAN
from .cyclegan import CycleGAN


def create_model(opt):
    if opt.model == 'unet':
        return Unet(opt)
    elif opt.model == 'pix2pix':
        return Pix2pix(opt)
    elif opt.model == 'pgan':
        return pGAN(opt)
    elif opt.model == 'cgan':
        return cGAN(opt)
    elif opt.model == 'cyclegan':
        return CycleGAN(opt)