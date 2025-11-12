import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """CycleGAN style ResNet generator (9 blocks for 256x256)."""
    def __init__(self, in_c=1, out_c=1, base=64, n_blocks=9):
        super().__init__()
        layers = [nn.Conv2d(in_c, base, 7, 1, 3, bias=False), nn.BatchNorm2d(base), nn.ReLU(True)]
        # Downsample
        c = base
        for _ in range(2):
            layers += [nn.Conv2d(c, c*2, 3, 2, 1, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(True)]
            c *= 2
        # Resnet blocks
        for _ in range(n_blocks):
            layers += [ResnetBlock(c)]
        # Upsample
        for _ in range(2):
            layers += [nn.ConvTranspose2d(c, c//2, 3, 2, 1, output_padding=1, bias=False), nn.BatchNorm2d(c//2), nn.ReLU(True)]
            c //= 2
        layers += [nn.Conv2d(c, out_c, 7, 1, 3), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NLayerDiscriminator(nn.Module):
    """CycleGAN discriminator (PatchGAN)."""
    def __init__(self, in_c=1, base=64, n_layers=3):
        super().__init__()
        kw = 4; padw = 1
        sequence = [nn.Conv2d(in_c, base, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(base * nf_mult_prev, base * nf_mult, kw, 2, padw, bias=False), nn.BatchNorm2d(base * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(base * nf_mult, base * nf_mult, kw, 1, padw, bias=False), nn.BatchNorm2d(base * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(base * nf_mult, 1, kw, 1, padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
