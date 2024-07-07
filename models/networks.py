import random

import torch
import torch.nn as nn
from torchvision import models


def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)



###############################################
############          Unet         ############
###############################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.conv(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x_conv = self.conv(x)
        x_down = self.down(x_conv)
        return (x_conv, x_down)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2)
        self.conv = DoubleConv(in_channels, out_channels)
        

    def forward(self, x, x_skip):
        x = self.up(x)
        x = torch.cat((x, x_skip), dim=1)
        return self.conv(x)
    

class UnetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = UnetDown(in_channels, 16)
        self.down2 = UnetDown(16, 32)
        self.down3 = UnetDown(32, 64)
        self.down4 = UnetDown(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.up1 = UnetUp(256, 128)
        self.up2 = UnetUp(128, 64)
        self.up3 = UnetUp(64, 32)
        self.up4 = UnetUp(32, 16)
        self.final = nn.Sequential(
            nn.Conv2d(16, out_channels, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x1, d1 = self.down1(x)
        x2, d2 = self.down2(d1)
        x3, d3 = self.down3(d2)
        x4, d4 = self.down4(d3)
        x5 = self.bottleneck(d4)
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        return self.final(u4)


##################################################
############          Pix2pix         ############
##################################################

class UnetDown4Pix(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp4Pix(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Pix2pixGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down1 = UnetDown4Pix(in_channels, 64, normalize=False)
        self.down2 = UnetDown4Pix(64, 128)
        self.down3 = UnetDown4Pix(128, 256)
        self.down4 = UnetDown4Pix(256, 512, dropout=0.5)
        self.down5 = UnetDown4Pix(512, 512, dropout=0.5)
        self.down6 = UnetDown4Pix(512, 512, dropout=0.5)
        self.down7 = UnetDown4Pix(512, 512, dropout=0.5)
        self.down8 = UnetDown4Pix(512, 512, normalize=False, dropout=0.5)

        self.up1 = UnetUp4Pix(512, 512, dropout=0.5)
        self.up2 = UnetUp4Pix(1024, 512, dropout=0.5)
        self.up3 = UnetUp4Pix(1024, 512, dropout=0.5)
        self.up4 = UnetUp4Pix(1024, 512, dropout=0.5)
        self.up5 = UnetUp4Pix(1024, 256)
        self.up6 = UnetUp4Pix(512, 128)
        self.up7 = UnetUp4Pix(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Pix2pixDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        layers = []
        
        for i in range(4):
            layers.append(nn.Conv2d(in_channels, 64 * 2**i, 4, stride=2, padding=1))
            if i != 1:
                layers.append(nn.BatchNorm2d(64 * 2**i))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = 64 * 2**i
        
        layers.extend([
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
        ])

        self.model = nn.Sequential(*layers)


    def forward(self, img_A, img_B):
        return self.model(torch.cat((img_A, img_B), 1))
    


#########################################################
#############          pGAN & cGAN         ##############
#########################################################

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))
        
        layers.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(in_channels),
        ])
        
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        out = x + self.model(x)
        return out



class pcGANGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=6, dropout=0.0):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ]

        for i in range(1, 3):
            layers.extend([
                nn.Conv2d(64 * 2**(i - 1), 64 * 2**i, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64 * 2**i),
                nn.ReLU(True),
            ])
       
        for i in range(num_blocks):
            layers.append(ResnetBlock(256, dropout))

        for i in range(2):
            layers.extend([
                nn.ConvTranspose2d(64 * 2**(2 - i), int(64 * 2**(1 - i)), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(int(64 * 2**(1 - i))),
                nn.ReLU(True),
            ])

        layers.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)



class pcGANDiscriminator(nn.Module):
    def __init__(self, in_channels, num_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        
        channels = 64
        for i in range(num_layers):
            prev_channels = channels
            channels = min(64 * 2**(i + 1), 512)
            layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=4, stride=1 if i == num_layers else 2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, True),
            ])

        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)


    def forward(self, input):
        return self.model(input)
      


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg16().features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.stage1(X)
        h_relu2 = self.stage2(h_relu1)       
        return h_relu2


######################################################
#############          CycleGAN         ##############
######################################################

class CycleGenerator(nn.Module):
    def __init__(self, in_channels, num_residual_blocks):
        super().__init__()
        out_channels = 64
        layers = [
            nn.ReflectionPad2d(5),
            nn.Conv2d(in_channels, 64, 7),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        for _ in range(2):
            prev_channels = out_channels
            out_channels *= 2
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

        for _ in range(num_residual_blocks):
            layers.append(ResnetBlock(out_channels, dropout=0.0))

        for _ in range(2):
            prev_channels = out_channels
            out_channels //= 2
            layers.extend([
                nn.Upsample(scale_factor=2),
                nn.Conv2d(prev_channels, out_channels, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

        layers.extend([
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(out_channels, in_channels, 7),
            nn.Tanh()
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class CycleDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        layers = []
        layers.extend([
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        channels = 64
        for _ in range(3):
            prev_channels = channels
            channels *= 2
            layers.extend([
                nn.Conv2d(prev_channels, channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        layers.extend([
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        ])
        
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

