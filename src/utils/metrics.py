import torch
import torch.nn.functional as F
from typing import List, Tuple


def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (PSNR) for tensors in [0, max_val].
    Args:
        x, y: (N,C,H,W)
    """
    mse = F.mse_loss(x, y, reduction='none').flatten(1).mean(dim=1)
    psnr_val = 20 * torch.log10(torch.tensor(max_val, device=x.device)) - 10 * torch.log10(mse + 1e-8)
    return psnr_val.mean()


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, C1: float, C2: float, win_size: int = 11) -> torch.Tensor:
    # Very small and fast SSIM approximation using uniform window
    pad = win_size // 2
    weight = torch.ones(1, 1, win_size, win_size, device=x.device) / (win_size * win_size)
    mu_x = F.conv2d(x, weight, padding=pad, groups=x.size(1))
    mu_y = F.conv2d(y, weight, padding=pad, groups=y.size(1))
    sigma_x = F.conv2d(x * x, weight, padding=pad, groups=x.size(1)) - mu_x * mu_x
    sigma_y = F.conv2d(y * y, weight, padding=pad, groups=y.size(1)) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, weight, padding=pad, groups=x.size(1)) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-8)
    return ssim_map.mean(dim=[2, 3]).mean(dim=1)


def ssim(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Structural Similarity (SSIM) index. Simple, fast approximation.
    Args:
        x, y: (N,C,H,W) in [0, max_val]
    """
    K1, K2 = 0.01, 0.03
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2
    return _ssim_per_channel(x, y, C1, C2).mean()


def fid(real_acts: torch.Tensor, fake_acts: torch.Tensor) -> float:
    """Compute FID given Inception activations (N, D).
    Note: This expects you to provide features; use torchmetrics or pytorch-fid for extraction.
    """
    import numpy as np
    from numpy.linalg import sqrtm

    mu1, mu2 = real_acts.mean(0).cpu().numpy(), fake_acts.mean(0).cpu().numpy()
    sigma1 = np.cov(real_acts.cpu().numpy(), rowvar=False)
    sigma2 = np.cov(fake_acts.cpu().numpy(), rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2)).real
    fid_val = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid_val)


def inception_score(logits: torch.Tensor, splits: int = 1) -> Tuple[float, float]:
    """Inception Score given class probabilities/logits (N, K).
    Provide probabilities (softmaxed) for stable results.
    Returns (mean, std).
    """
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=1)
    N = probs.size(0)
    split_size = N // splits
    scores = []
    for k in range(splits):
        p = probs[k * split_size : (k + 1) * split_size]
        py = p.mean(dim=0, keepdim=True)
        kl = p * (torch.log(p + 1e-8) - torch.log(py + 1e-8))
        scores.append(torch.exp(kl.sum(dim=1).mean()))
    scores = torch.stack(scores)
    return float(scores.mean()), float(scores.std(unbiased=False))
