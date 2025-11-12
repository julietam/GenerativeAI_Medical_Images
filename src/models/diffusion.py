import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_c, out_c),
            ConvBlock(out_c, out_c),
            nn.Conv2d(out_c, out_c, 4, 2, 1),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_c, out_c),
            ConvBlock(out_c, out_c),
            nn.ConvTranspose2d(out_c, out_c, 4, 2, 1),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    """Small UNet for 64x64 diffusion examples (grayscale by default)."""
    def __init__(self, channels=1, base=64):
        super().__init__()
        self.inp = ConvBlock(channels, base)
        self.d1 = Down(base, base*2)
        self.d2 = Down(base*2, base*4)
        self.mid = ConvBlock(base*4, base*4)
        self.u2 = Up(base*4, base*2)
        self.u1 = Up(base*2, base)
        self.out = nn.Conv2d(base, channels, 1)

    def forward(self, x, t=None):
        x1 = self.inp(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        m = self.mid(x3)
        u2 = self.u2(m)
        u1 = self.u1(u2)
        return self.out(u1)


class DDPM(nn.Module):
    """Minimal DDPM wrapper with cosine beta schedule."""
    def __init__(self, model: nn.Module, img_size: int = 64, channels: int = 1, timesteps: int = 1000):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.channels = channels
        self.T = timesteps
        self.register_buffer('betas', self.cosine_beta_schedule(self.T))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_alphas_cumprod[t][:, None, None, None] * x0 + \
               self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t)
        loss = F.mse_loss(pred, noise)
        return loss, xt

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        beta_t = self.betas[t]
        sqrt_one_minus_at = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphat = (1.0 / torch.sqrt(self.alphas[t]))
        model_mean = sqrt_recip_alphat * (x - beta_t / sqrt_one_minus_at * self.model(x, torch.tensor([t], device=x.device)))
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(n, self.channels, self.img_size, self.img_size, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t)
        return x
