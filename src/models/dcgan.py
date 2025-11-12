import torch
import torch.nn as nn


def weights_init_normal(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)


class DCGANGenerator(nn.Module):
    """Generator for DCGAN supporting square images.

    Args:
        z_dim: latent vector size
        img_channels: 1 for grayscale, 3 for RGB
        img_size: final image side (e.g., 64, 128)
        base_channels: width multiplier
    """
    def __init__(self, z_dim: int = 100, img_channels: int = 1, img_size: int = 128, base_channels: int = 64):
        super().__init__()
        assert img_size in {64, 128, 256}, "img_size must be one of {64, 128, 256}"
        num_ups = {64: 4, 128: 5, 256: 6}[img_size]

        layers = []
        c = base_channels * (2 ** (num_ups - 1))
        layers += [nn.ConvTranspose2d(z_dim, c, 4, 1, 0, bias=False), nn.BatchNorm2d(c), nn.ReLU(True)]
        for i in range(num_ups - 2, 0, -1):
            layers += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ConvTranspose2d(c, img_channels, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(weights_init_normal)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class DCGANDiscriminator(nn.Module):
    def __init__(self, img_channels: int = 1, img_size: int = 128, base_channels: int = 64):
        super().__init__()
        assert img_size in {64, 128, 256}
        num_downs = {64: 4, 128: 5, 256: 6}[img_size]

        layers = []
        c = base_channels
        layers += [nn.Conv2d(img_channels, c, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(1, num_downs - 1):
            layers += [nn.Conv2d(c, min(c * 2, 512), 4, 2, 1, bias=False), nn.BatchNorm2d(min(c * 2, 512)), nn.LeakyReLU(0.2, inplace=True)]
            c = min(c * 2, 512)
        layers += [nn.Conv2d(c, 1, 4, 1, 0, bias=False)]
        self.net = nn.Sequential(*layers)
        self.apply(weights_init_normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)
