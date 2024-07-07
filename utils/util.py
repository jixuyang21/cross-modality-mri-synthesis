import os
import logging
import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import save_image

from skimage.metrics import peak_signal_noise_ratio as compare_psnr, \
                            structural_similarity as compare_ssim


def make_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def make_epoch_dirs(root):
    epochs = list(range(1, 101))
    sub_dirs = list(map(lambda x: os.path.join(root, f"epoch_{x}"), epochs))
    make_dirs(*sub_dirs)


def get_logger(opt, mode):
    logger = logging.getLogger(f'{mode}')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    file_handler = logging.FileHandler(os.path.join(opt.logs_path, f'{mode}.txt'), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_train_batch_loss(opt, epoch, batch_idx, train_loader, loss):
    if opt.model == 'unet':
        return "[epoch %d/%d] [batch %d/%d] [loss: %f]" % \
                (epoch + 1, opt.num_epochs, batch_idx + 1, len(train_loader), loss[0].item())
    else:
        return "[epoch %d/%d] [batch %d/%d] [loss G: %f] [loss D: %f]" % \
                (epoch + 1, opt.num_epochs, batch_idx + 1, len(train_loader), loss[0].item(), loss[1].item())


class MetricsEvaluatorRecoder:
    def __init__(self, opt):
        self.opt = opt
        self.compare_psnr = compare_psnr
        self.compare_ssim = compare_ssim
        self.compare_mae = nn.L1Loss()
        self.not_cyc = (opt.model in ['unet', 'pix2pix', 'pgan'])
        

    def initialize(self):    
        if self.not_cyc:
            self.psnrs, self.ssims, self.maes = [], [], []
        else:    
            self.psnrs = [[], []]
            self.ssims = [[], []]
            self.maes = [[], []]

    
    def evaluate_two_batches(self, real_imgs, fake_imgs):
        psnrs = list(map(lambda x, y: self.compare_psnr(x.numpy(), y.cpu().detach().numpy(), data_range=1), real_imgs, fake_imgs))
        ssims = list(map(lambda x, y: self.compare_ssim(x.numpy(), y.cpu().detach().numpy(), data_range=1, channel_axis=0), real_imgs, fake_imgs))
        maes = list(map(lambda x, y: self.compare_mae(x, y.cpu().detach()).item(), real_imgs, fake_imgs))
        return psnrs, ssims, maes


    def evaluate_metrics(self, real_imgs, fake_imgs):
        if self.not_cyc:
            self.psnr, self.ssim, self.mae = self.evaluate_two_batches(real_imgs[0], fake_imgs)
        else:
            (psnr1, ssim1, mae1), (psnr2, ssim2, mae2) = list(map(lambda x, y: self.evaluate_two_batches(x, y), real_imgs, fake_imgs))
            self.psnr = (psnr1, psnr2)
            self.ssim = (ssim1, ssim2)
            self.mae = (mae1, mae2)


    def record(self):
        if self.not_cyc:
            self.psnrs.extend(self.psnr)
            self.ssims.extend(self.ssim)
            self.maes.extend(self.mae)
        else:
            self.psnrs = tuple(map(lambda x, y: x + y, self.psnrs, self.psnr))
            self.ssims = tuple(map(lambda x, y: x + y, self.ssims, self.ssim))
            self.maes = tuple(map(lambda x, y: x + y, self.maes, self.mae))
    

    def compute_mean_std(self):
        if self.not_cyc:
            self.means = tuple(map(lambda x: np.array(x).mean(), [self.psnrs, self.ssims, self.maes]))
            self.stds = tuple(map(lambda x: np.array(x).std(), [self.psnrs, self.ssims, self.maes]))
        else:
            self.psnr_means = tuple(map(lambda x: np.array(x).mean(), self.psnrs))
            self.psnr_stds = tuple(map(lambda x: np.array(x).std(), self.psnrs))
            self.ssim_means = tuple(map(lambda x: np.array(x).mean(), self.ssims))
            self.ssim_stds = tuple(map(lambda x: np.array(x).std(), self.ssims))
            self.mae_means = tuple(map(lambda x: np.array(x).mean(), self.maes))
            self.mae_stds = tuple(map(lambda x: np.array(x).std(), self.maes))
    

    def concatenate_imgs(self, real_imgs, fake_imgs, device):
        if self.not_cyc:
            return torch.cat((real_imgs[0].to(device), fake_imgs, real_imgs[1].to(device)), dim=-2)
        else:
            return torch.cat((real_imgs[0].to(device), fake_imgs[0], real_imgs[1].to(device), fake_imgs[1]), dim=-2)
    
    
    def evaluate_model(self, opt, model, dataloader, logger, epoch, dataset_length):
        model.eval_mode()
        self.initialize()
        info = f'[epoch {epoch}]\n'
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                real_A = batch[opt.mod_in]
                real_B = batch[opt.mod_out]

                fake_imgs, loss = model.forward(real_A, real_B)
                val_loss += loss.item()
                self.evaluate_metrics((real_B, real_A), fake_imgs)
                self.record()

                imgs = self.concatenate_imgs((real_A, real_B), fake_imgs, model.device)

                save_image(imgs, os.path.join(opt.results_path, f'epoch_{epoch}', f"{i*self.opt.batch_size+1}-{i*self.opt.batch_size+real_A.size(0)}.png"), nrow=real_A.size(0)//2, normalize=True)
                info += self.get_batch_metrics(i * self.opt.batch_size, dataset_length)
        
        self.compute_mean_std()
        logger.info(info + self.get_final_metrics() + '-' * 10)
        
        return val_loss
            


    def draw_metrics_curve(self, writer, epoch):
        if self.not_cyc:
            writer.add_scalars(f'[val_metrics] [{self.opt.mod_in} -> {self.opt.mod_out}]', {'psnr': self.means[0], 'ssim': self.means[1], 'mae': self.means[2]}, epoch)
        else:
            writer.add_scalars(f'[val_metrics] [{self.opt.mod_in} -> {self.opt.mod_out}]', {'psnr': self.psnr_means[0], 'ssim': self.ssim_means[0], 'mae': self.mae_means[0]}, epoch)
            writer.add_scalars(f'[val_metrics] [{self.opt.mod_out} -> {self.opt.mod_in}]', {'psnr': self.psnr_means[1], 'ssim': self.ssim_means[1], 'mae': self.mae_means[1]}, epoch)
            writer.add_scalars(f'[val_metrics] [mean]', {'psnr': (self.psnr_means[0] + self.psnr_means[1])/2,
                                                            'ssim': (self.ssim_means[0] + self.ssim_means[1])/2,
                                                            'mae': (self.mae_means[0] + self.mae_means[1])/2}, epoch)


    def get_batch_metrics(self, pre_data_nums, dataset_length):
        info = ""
        if self.not_cyc:
            for i, (p, s, m) in enumerate(zip(self.psnr, self.ssim, self.mae)):
                info += "[data %d/%d] [psnr %f] [ssim %f] [mae %f]\n" % \
                (pre_data_nums + i + 1, dataset_length, p, s, m)
        else:
            for i, (p1, p2, s1, s2, m1, m2) in enumerate(zip(self.psnr[0], self.psnr[1], self.ssim[0], self.ssim[1], self.mae[0], self.mae[1])):
                info += "[%s -> %s] [data %d/%d] [psnr %f] [ssim %f] [mae %f]\n" % \
                        (self.opt.mod_in, self.opt.mod_out, pre_data_nums + i + 1, dataset_length, p1, s1, m1) + \
                        "[%s -> %s] [data %d/%d] [psnr %f] [ssim %f] [mae %f]\n" % \
                        (self.opt.mod_out, self.opt.mod_in, pre_data_nums + i + 1, dataset_length, p2, s2, m2)
        return info


    def get_final_metrics(self):
        if self.not_cyc:
            return "[psnr %f ± %f] [ssim %f ± %f] [mae %f ± %f]\n" % (self.means[0], self.stds[0], self.means[1], self.stds[1], self.means[2], self.stds[2])
        else:
            return "[%s -> %s] [psnr %f ± %f] [ssim %f ± %f] [mae %f ± %f]\n" % \
                    (self.opt.mod_in, self.opt.mod_out, self.psnr_means[0], self.psnr_stds[0], self.ssim_means[0], self.ssim_stds[0], self.mae_means[0], self.mae_stds[0]) + \
                    "[%s -> %s] [psnr %f ± %f] [ssim %f ± %f] [mae %f ± %f]\n" % \
                    (self.opt.mod_out, self.opt.mod_in, self.psnr_means[1], self.psnr_stds[1], self.ssim_means[1], self.ssim_stds[1], self.mae_means[1], self.mae_stds[1])