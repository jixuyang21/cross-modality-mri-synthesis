import os

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import UnetGenerator, weights_init_normal


class Unet:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')

        self.generator = UnetGenerator().to(self.device)
        self.init_weight()
        self.optimizer = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 - max(0, epoch - 49) / 51)

        self.loss_fn = nn.MSELoss().to(self.device)

    
    def init_weight(self):
        self.generator.apply(weights_init_normal)


    def forward(self, real_A, real_B):
        self.real_A = real_A.to(self.device, dtype=torch.float)
        self.real_B = real_B.to(self.device, dtype=torch.float)
        
        self.fake_B = self.generator(self.real_A)
        self.compute_loss()
        
        return self.fake_B, self.loss
    
    
    def compute_loss(self):
        self.loss = self.loss_fn(self.fake_B, self.real_B)
    

    def optimize(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        return
    
    
    def update_lr(self):
        self.scheduler.step()

    
    def train_mode(self):
        self.generator.train()


    def eval_mode(self):
        self.generator.eval()
    

    def save(self, epoch):
        torch.save(self.generator.state_dict(), os.path.join(self.opt.ckpts_path, f"G_epoch_{epoch}.pth"))


    def load(self, epoch):
        self.generator.load_state_dict(torch.load(os.path.join(self.opt.ckpts_path, f"G_epoch_{epoch}.pth")))
