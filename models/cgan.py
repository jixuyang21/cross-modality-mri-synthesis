import os
import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import pcGANGenerator, pcGANDiscriminator, weights_init_normal


class cGAN:
    def __init__(self, opt):
        self.opt = opt

        self.device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')

        self.generator_AB = pcGANGenerator(1, 1).to(self.device)
        self.generator_BA = pcGANGenerator(1, 1).to(self.device)
        self.discriminator_A = pcGANDiscriminator(1).to(self.device)
        self.discriminator_B = pcGANDiscriminator(1).to(self.device)
        self.init_weight()
        
        self.optimizer_G = optim.Adam(itertools.chain(self.generator_AB.parameters(), self.generator_BA.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D_A = optim.Adam(self.discriminator_A.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D_B = optim.Adam(self.discriminator_B.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        
        self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda epoch: 1 - max(0, epoch - 49) / 51)
        self.scheduler_D_A = optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=lambda epoch: 1 - max(0, epoch - 49) / 51)
        self.scheduler_D_B = optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=lambda epoch: 1 - max(0, epoch - 49) / 51)

        self.loss_gan = nn.MSELoss().to(self.device)
        self.loss_l1 = nn.L1Loss().to(self.device)


    def init_weight(self):
        self.generator_AB.apply(weights_init_normal)
        self.generator_BA.apply(weights_init_normal)
        self.discriminator_A.apply(weights_init_normal)
        self.discriminator_B.apply(weights_init_normal)


    def forward(self, real_A, real_B):
        self.real_A = real_A.to(self.device, dtype=torch.float)
        self.real_B = real_B.to(self.device, dtype=torch.float)
        
        self.fake_B = self.generator_AB(self.real_A)
        self.fake_A = self.generator_BA(self.real_B)
        self.rec_A = self.generator_BA(self.fake_B)
        self.rec_B = self.generator_AB(self.fake_A)
        
        self.compute_loss_G()

        return (self.fake_B, self.fake_A), self.loss_G


    def compute_loss_G(self):
        pred_fake_B = self.discriminator_B(self.fake_B)
        pred_fake_A = self.discriminator_A(self.fake_A)
        valid = torch.ones_like(pred_fake_B, requires_grad=False).to(self.device)

        loss_G_B_adv = self.loss_gan(pred_fake_B, valid)
        loss_G_B_id = self.loss_l1(self.rec_B, self.real_B)
        loss_G_A_adv = self.loss_gan(pred_fake_A, valid)
        loss_G_A_id = self.loss_l1(self.rec_A, self.real_A)
        
        self.loss_G = self.opt.lambda_adv * (loss_G_A_adv + loss_G_B_adv) / 2 + self.opt.lambda_id * (loss_G_A_id + loss_G_B_id) / 2
    

    def compute_loss_D(self):
        pred_real_A = self.discriminator_A(self.real_A)
        pred_fake_A = self.discriminator_A(self.fake_A.detach())

        pred_real_B = self.discriminator_B(self.real_B)
        pred_fake_B = self.discriminator_B(self.fake_B.detach())
        
        valid = torch.ones_like(pred_real_B, requires_grad=False).to(self.device)
        fake = torch.zeros_like(pred_fake_B, requires_grad=False).to(self.device)

        loss_D_A_real = self.loss_gan(pred_real_A, valid)
        loss_D_A_fake = self.loss_gan(pred_fake_A, fake)
        self.loss_D_A = self.opt.lambda_adv * (loss_D_A_real + loss_D_A_fake) / 2
        
        loss_D_B_real = self.loss_gan(pred_real_B, valid)
        loss_D_B_fake = self.loss_gan(pred_fake_B, fake)
        self.loss_D_B = self.opt.lambda_adv * (loss_D_B_real + loss_D_B_fake) / 2


    def optimize(self):
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.compute_loss_D()
        self.optimizer_D_A.zero_grad()
        self.loss_D_A.backward()
        self.optimizer_D_A.step()
        
        self.optimizer_D_B.zero_grad()
        self.loss_D_B.backward()
        self.optimizer_D_B.step()
        
        return (self.loss_D_A + self.loss_D_B) / 2

    
    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D_A.step()
        self.scheduler_D_B.step()

    
    def train_mode(self):
        self.generator_AB.train()
        self.generator_BA.train()
        self.discriminator_A.train()
        self.discriminator_B.train()


    def eval_mode(self):
        self.generator_AB.eval()
        self.generator_BA.eval()
        self.discriminator_A.eval()
        self.discriminator_B.eval()
    

    def save(self, epoch):
        torch.save(self.generator_AB.state_dict(), os.path.join(self.opt.ckpts_path, f"G_AB_epoch_{epoch}.pth"))
        torch.save(self.generator_BA.state_dict(), os.path.join(self.opt.ckpts_path, f"G_BA_epoch_{epoch}.pth"))
        torch.save(self.discriminator_A.state_dict(), os.path.join(self.opt.ckpts_path, f"D_A_epoch_{epoch}.pth"))
        torch.save(self.discriminator_B.state_dict(), os.path.join(self.opt.ckpts_path, f"D_B_epoch_{epoch}.pth"))


    def load(self, epoch):
        self.generator_AB.load_state_dict(torch.load(os.path.join(self.opt.ckpts_path, f"G_AB_epoch_{epoch}.pth")))
        self.generator_BA.load_state_dict(torch.load(os.path.join(self.opt.ckpts_path, f"G_BA_epoch_{epoch}.pth")))
        self.discriminator_A.load_state_dict(torch.load(os.path.join(self.opt.ckpts_path, f"D_A_epoch_{epoch}.pth")))
        self.discriminator_B.load_state_dict(torch.load(os.path.join(self.opt.ckpts_path, f"D_B_epoch_{epoch}.pth")))

