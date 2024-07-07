import os

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import pcGANGenerator, pcGANDiscriminator, VGG16, weights_init_normal


class pGAN:
    def __init__(self, opt):
        self.opt = opt

        self.device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')

        self.generator = pcGANGenerator(1, 1).to(self.device)
        self.discriminator = pcGANDiscriminator(2).to(self.device)
        self.vgg = VGG16().to(self.device)
        self.init_weight()
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        
        self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda epoch: 1 - max(0, epoch - 49) / 51)
        self.scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda epoch: 1 - max(0, epoch - 49) / 51)
        
        self.loss_gan = nn.MSELoss().to(self.device)
        self.loss_l1 = nn.L1Loss().to(self.device)


    def init_weight(self):
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)


    def forward(self, real_A, real_B):
        self.real_A = real_A.to(self.device, dtype=torch.float)
        self.real_B = real_B.to(self.device, dtype=torch.float)
        
        self.fake_B = self.generator(self.real_A)
        self.compute_loss_G()

        return self.fake_B, self.loss_G
    
    
    def compute_loss_G(self):
        vgg_real_B = self.vgg(self.real_B.expand(self.real_B.size(0), 3, self.real_B.size(2), self.real_B.size(3)))[0]
        vgg_fake_B = self.vgg(self.fake_B.expand(self.fake_B.size(0), 3, self.fake_B.size(2), self.fake_B.size(3)))[0]

        pred_fake_B = self.discriminator(torch.cat((self.fake_B, self.real_A), dim=1))
        valid = torch.ones_like(pred_fake_B, requires_grad=False).to(self.device)

        loss_G_vgg = self.loss_l1(vgg_fake_B, vgg_real_B)
        loss_G_adv = self.loss_gan(pred_fake_B, valid)
        loss_G_id = self.loss_l1(self.fake_B, self.real_B)
        self.loss_G = self.opt.lambda_adv * loss_G_adv + self.opt.lambda_id * loss_G_id + self.opt.lambda_vgg * loss_G_vgg
    

    def compute_loss_D(self):
        pred_real_B = self.discriminator(torch.cat((self.real_B, self.real_A), dim=1))
        pred_fake_B = self.discriminator(torch.cat((self.fake_B.detach(), self.real_A), dim=1))
        
        valid = torch.ones_like(pred_real_B, requires_grad=False).to(self.device)
        fake = torch.zeros_like(pred_fake_B, requires_grad=False).to(self.device)
        
        loss_D_real = self.loss_gan(pred_real_B, valid)
        loss_D_fake = self.loss_gan(pred_fake_B, fake)
        self.loss_D = self.opt.lambda_adv * (loss_D_real + loss_D_fake) / 2
    

    def optimize(self):
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.compute_loss_D()
        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

        return self.loss_D

    
    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

    
    def train_mode(self):
        self.generator.train()
        self.discriminator.train()
    

    def eval_mode(self):
        self.generator.eval()
        self.discriminator.eval()
    

    def save(self, epoch):
        torch.save(self.generator.state_dict(), os.path.join(self.opt.ckpts_path, f"G_epoch_{epoch}.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.opt.ckpts_path, f"D_epoch_{epoch}.pth"))


    def load(self, epoch):
        self.generator.load_state_dict(torch.load(os.path.join(self.opt.ckpts_path, f"G_epoch_{epoch}.pth")))
        self.discriminator.load_state_dict(torch.load(os.path.join(self.opt.ckpts_path, f"D_epoch_{epoch}.pth")))

