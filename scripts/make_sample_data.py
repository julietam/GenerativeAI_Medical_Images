import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path('data')
PAIR_ROOT = ROOT / 'sample_mri_pairs'
UNPAIRED_MRI = ROOT / 'sample_unpaired_mri'
UNPAIRED_CT = ROOT / 'sample_unpaired_ct'

IMG_SIZE = 256
N = 16
rng = np.random.default_rng(42)

def save_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(path)

# Create paired T1/T2
for i in range(N):
    # T1: random blobs
    img = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    for _ in range(rng.integers(3, 7)):
        cx, cy = rng.integers(32, IMG_SIZE-32, size=2)
        r = int(rng.integers(8, 28))
        yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
        val = float(rng.uniform(120, 220))
        img[mask] = np.maximum(img[mask], val)
    img = img + rng.normal(0, 5, img.shape)
    t1 = np.clip(img, 0, 255)
    # T2: smoothed + inverted-ish contrast
    pil_t1 = Image.fromarray(t1.astype(np.uint8))
    t2_pil = pil_t1.filter(ImageFilter.GaussianBlur(radius=1.5))
    t2 = np.array(t2_pil, dtype=np.float32)
    t2 = 255 - (t2 * 0.9 + 10)  # simple transform
    save_png(t1, PAIR_ROOT / 'T1' / f'subj_{i:03d}.png')
    save_png(t2, PAIR_ROOT / 'T2' / f'subj_{i:03d}.png')

# Unpaired MRI: reuse T1
for p in (PAIR_ROOT / 'T1').glob('*.png'):
    dest = UNPAIRED_MRI / p.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.open(p).save(dest)

# Unpaired CT: generate different texture (stripes + noise)
for i in range(N):
    base = np.tile(np.linspace(80, 200, IMG_SIZE, dtype=np.float32), (IMG_SIZE,1))
    base += rng.normal(0, 8, base.shape)
    # add rectangles
    for _ in range(rng.integers(2, 5)):
        x0, y0 = rng.integers(0, IMG_SIZE-40, size=2)
        w, h = rng.integers(20, 60, size=2)
        base[y0:y0+h, x0:x0+w] += float(rng.uniform(20, 60))
    save_png(base, UNPAIRED_CT / f'ct_{i:03d}.png')

print('Sample datasets created under:')
print(f'- {PAIR_ROOT}')
print(f'- {UNPAIRED_MRI}')
print(f'- {UNPAIRED_CT}')
