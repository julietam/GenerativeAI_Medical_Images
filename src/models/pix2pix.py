import torch
import torch.nn as nn
from typing import Optional


class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(out_c)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True)]
        if dropout:
            layers += [nn.Dropout(0.5)]
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        x = self.block(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x


class UNetGenerator(nn.Module):
    """Pix2Pix U-Net generator (256x256)."""
    def __init__(self, in_channels=1, out_channels=1, base=64):
        super().__init__()
        # Down 8x
        self.d1 = UNetDown(in_channels, base, norm=False)
        self.d2 = UNetDown(base, base*2)
        self.d3 = UNetDown(base*2, base*4)
        self.d4 = UNetDown(base*4, base*8)
        self.d5 = UNetDown(base*8, base*8)
        self.d6 = UNetDown(base*8, base*8)
        self.d7 = UNetDown(base*8, base*8)
        self.d8 = UNetDown(base*8, base*8, norm=False)
        # Up 8x
        self.u1 = UNetUp(base*8, base*8, dropout=True)
        self.u2 = UNetUp(base*16, base*8, dropout=True)
        self.u3 = UNetUp(base*16, base*8, dropout=True)
        self.u4 = UNetUp(base*16, base*8)
        self.u5 = UNetUp(base*16, base*4)
        self.u6 = UNetUp(base*8, base*2)
        self.u7 = UNetUp(base*4, base)
        self.outc = nn.Sequential(nn.ConvTranspose2d(base*2, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8, d7); u2 = self.u2(u1, d6); u3 = self.u3(u2, d5); u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3); u6 = self.u6(u5, d2); u7 = self.u7(u6, d1)
        return self.outc(u7)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""
    def __init__(self, in_channels=2, base=64):
        super().__init__()
        def C(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not norm)]
            if norm:
                layers += [nn.BatchNorm2d(out_c)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return layers
        self.net = nn.Sequential(
            *C(in_channels, base, norm=False),
            *C(base, base*2),
            *C(base*2, base*4),
            nn.Conv2d(base*4, base*8, 4, 1, 1, bias=False), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*8, 1, 4, 1, 1)
        )

    def forward(self, a, b):
        # Input is concatenation of condition and target
        x = torch.cat([a, b], dim=1)
        return self.net(x)
