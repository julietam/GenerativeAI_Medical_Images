import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from src.models.pix2pix import UNetGenerator, PatchGANDiscriminator

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    HAS_TM = True
except Exception:
    HAS_TM = False
    from src.utils.metrics import ssim as ssim_fn, psnr as psnr_fn


class PairedFolderDataset(Dataset):
    def __init__(self, root: str, img_size: int = 256):
        root = Path(root)
        self.t1 = sorted((root / "T1").glob("*.png"))
        self.t2 = sorted((root / "T2").glob("*.png"))
        assert len(self.t1) == len(self.t2) and len(self.t1) > 0, "T1/T2 must have same count and be non-empty"
        self.tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.t1)

    def __getitem__(self, idx):
        a = self.tf(Image.open(self.t1[idx]).convert('L'))
        b = self.tf(Image.open(self.t2[idx]).convert('L'))
        return a, b


def train(cfg_path: str):
    import yaml
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    root = cfg['data']['root']
    img_size = int(cfg['data'].get('img_size', 256))
    batch_size = int(cfg['train'].get('batch_size', 4))
    epochs = int(cfg['train'].get('epochs', 1))
    lr = float(cfg['train'].get('lr', 2e-4))
    l1_lambda = float(cfg['train'].get('l1_lambda', 100.0))
    out_dir = Path(cfg['train'].get('out_dir', 'ckpts/pix2pix'))
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PairedFolderDataset(root, img_size=img_size)
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    G = UNetGenerator(1, 1).to(device)
    D = PatchGANDiscriminator(2).to(device)
    opt_g = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    L1 = nn.L1Loss()
    BCE = nn.BCEWithLogitsLoss()

    if HAS_TM:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    best_val = float('inf')
    for epoch in range(epochs):
        G.train(); D.train()
        for a, b in train_dl:
            a, b = a.to(device), b.to(device)
            # D
            opt_d.zero_grad()
            pred_real = D(a, b)
            z = G(a)
            pred_fake = D(a, z.detach())
            loss_d = BCE(pred_real, torch.ones_like(pred_real)) + BCE(pred_fake, torch.zeros_like(pred_fake))
            loss_d.backward(); opt_d.step()
            # G
            opt_g.zero_grad()
            pred_fake = D(a, z)
            loss_g = BCE(pred_fake, torch.ones_like(pred_fake)) + l1_lambda * L1(z, b)
            loss_g.backward(); opt_g.step()
        # validation
        G.eval()
        val_l1 = 0.0; n = 0
        ssim_vals = []; psnr_vals = []
        with torch.no_grad():
            for a, b in val_dl:
                a, b = a.to(device), b.to(device)
                z = G(a)
                val_l1 += L1(z, b).item() * a.size(0)
                n += a.size(0)
                z01, b01 = (z+1)/2, (b+1)/2
                if HAS_TM:
                    ssim_vals.append(ssim_metric(z01, b01).item())
                    psnr_vals.append(psnr_metric(z01, b01).item())
                else:
                    ssim_vals.append(ssim_fn(z01, b01).item())
                    psnr_vals.append(psnr_fn(z01, b01).item())
        val_l1 /= max(1, n)
        print(f"Epoch {epoch}: val_L1={val_l1:.4f} SSIM={np.mean(ssim_vals):.3f} PSNR={np.mean(psnr_vals):.2f}")
        # save best
        if val_l1 < best_val:
            best_val = val_l1
            torch.save(G.state_dict(), out_dir / 'G_best.pt')
            torch.save(D.state_dict(), out_dir / 'D_best.pt')
    # final
    torch.save(G.state_dict(), out_dir / 'G_last.pt')
    torch.save(D.state_dict(), out_dir / 'D_last.pt')


if __name__ == '__main__':
    import numpy as np
    parser = argparse.ArgumentParser(description='Train Pix2Pix from folders (T1/T2)')
    parser.add_argument('--config', type=str, default='configs/pix2pix.yaml')
    args = parser.parse_args()
    train(args.config)
