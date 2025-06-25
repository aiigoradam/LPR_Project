# models/pix2pix.py

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32):
        super().__init__()
        self.in_conv = DoubleConv(in_ch, base)

        self.d1 = self.down(base, base * 2)
        self.d2 = self.down(base * 2, base * 4)
        self.d3 = self.down(base * 4, base * 8)
        self.d4 = self.down(base * 8, base * 16)
        self.d5 = self.down(base * 16, base * 16)

        self.u1 = self.up(base * 16, base * 16, drop=True)
        self.u2 = self.up(base * 32, base * 8, drop=True)
        self.u3 = self.up(base * 16, base * 4)
        self.u4 = self.up(base * 8, base * 2)
        self.u5 = self.up(base * 4, base)

        self.out_conv = nn.Sequential(nn.Conv2d(base * 2, out_ch, 3, padding=1), nn.Tanh())

    @staticmethod
    def down(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True)
        )

    @staticmethod
    def up(in_ch, out_ch, drop=False):
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if drop:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.in_conv(x)
        d1 = self.d1(x0)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        u1 = self.u1(d5)
        u1 = torch.cat([u1, d4], 1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d3], 1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d2], 1)

        u4 = self.u4(u3)
        u4 = torch.cat([u4, d1], 1)

        u5 = self.u5(u4)
        u5 = torch.cat([u5, x0], 1)

        return self.out_conv(u5)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv2d(in_ch * 2, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.l2 = self.block(base, base * 2, stride=2)
        self.l3 = self.block(base * 2, base * 4, stride=2)
        self.l4 = self.block(base * 4, base * 8, stride=1)
        self.out = nn.Conv2d(base * 8, 1, 4, 1, 1)

    @staticmethod
    def block(in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, y):
        z = torch.cat([x, y], 1)  # 6 channels
        z = self.l1(z)
        z = self.l2(z)
        z = self.l3(z)
        z = self.l4(z)
        return self.out(z)
