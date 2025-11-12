import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Dict, List


def show_images(images: torch.Tensor, nrow: int = 8, figsize=(8, 8), title: Optional[str] = None):
    """Show a grid of images in [-1,1] or [0,1]."""
    imgs = images.detach().cpu()
    if imgs.min() < 0:  # assume [-1,1]
        imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1)
    N, C, H, W = imgs.shape
    nrow = min(nrow, N)
    ncol = int(np.ceil(N / nrow))
    fig, axes = plt.subplots(ncol, nrow, figsize=figsize)
    axes = np.array(axes).reshape(ncol, nrow)
    for i in range(ncol * nrow):
        r, c = divmod(i, nrow)
        ax = axes[r, c]
        ax.axis('off')
        if i < N:
            img = imgs[i]
            if C == 1:
                ax.imshow(img[0], cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_curves(history: Dict[str, List[float]], figsize=(8, 4), title: Optional[str] = None):
    plt.figure(figsize=figsize)
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
