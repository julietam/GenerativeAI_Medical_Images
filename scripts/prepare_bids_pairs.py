import argparse
import os
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from skimage.transform import resize
from PIL import Image


def _normalize_resize(sl: np.ndarray, img_size: int) -> np.ndarray:
    mn, mx = float(sl.min()), float(sl.max())
    if mx > mn:
        sl = (sl - mn) / (mx - mn)
    else:
        sl = np.zeros_like(sl, dtype=np.float32)
    sl = resize(sl, (img_size, img_size), preserve_range=True, anti_aliasing=True)
    return np.clip(sl, 0.0, 1.0)


def to_png_slice(vol_path: Path, img_size: int = 256, z_policy: str | int = "center") -> np.ndarray:
    img = nib.load(str(vol_path))
    data = img.get_fdata().astype(np.float32)
    # choose axial slice
    if data.ndim == 4:
        data = data[..., 0]
    if isinstance(z_policy, int):
        z = max(0, min(int(z_policy), data.shape[2] - 1))
    elif z_policy == "center":
        z = data.shape[2] // 2
    else:
        z = max(0, min(int(z_policy), data.shape[2] - 1))
    sl = data[:, :, z]
    return _normalize_resize(sl, img_size)


def save_png(arr01: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr01 * 255).astype(np.uint8)).save(out_path)


def find_subject_pairs(ds_root: Path, max_pairs: int = 3) -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []
    # Look for BIDS-style: sub-*/(ses-*)/anat/*_T1w.nii* and *_T2w.nii*
    subs = sorted([p for p in ds_root.glob('sub-*') if p.is_dir()])
    for sub in subs:
        ses_dirs = [sub]
        ses_dirs += sorted([p for p in sub.glob('ses-*') if p.is_dir()])
        for ses in ses_dirs:
            anat = ses / 'anat'
            if not anat.exists():
                continue
            t1s = sorted(anat.glob('*_T1w.nii*'))
            t2s = sorted(anat.glob('*_T2w.nii*'))
            if t1s and t2s:
                sid = ses.relative_to(ds_root).as_posix().replace('/', '_')
                pairs.append((t1s[0], t2s[0], sid))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def main():
    ap = argparse.ArgumentParser(description='Prepare paired PNGs from BIDS (T1w/T2w) for Pix2Pix')
    ap.add_argument('--ds_root', type=str, required=True, help='OpenNeuro/BIDS dataset root (e.g., data/openneuro/ds005533)')
    ap.add_argument('--out_root', type=str, default='data/paired_mri')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--max_pairs', type=int, default=3)
    ap.add_argument('--slice_offsets', type=str, default='0,-1,1', help='Comma-separated axial offsets relative to center (e.g., 0,-1,1)')
    args = ap.parse_args()

    ds_root = Path(args.ds_root)
    out_root = Path(args.out_root)
    offsets = [int(x) for x in args.slice_offsets.split(',') if x.strip() != '']

    pairs = find_subject_pairs(ds_root, max_pairs=args.max_pairs)
    if not pairs:
        print('No T1w/T2w pairs found.')
        return

    for t1p, t2p, sid in pairs:
        # Load volumes once to determine center and bounds
        img1 = nib.load(str(t1p)); vol1 = img1.get_fdata().astype(np.float32)
        img2 = nib.load(str(t2p)); vol2 = img2.get_fdata().astype(np.float32)
        if vol1.ndim == 4: vol1 = vol1[..., 0]
        if vol2.ndim == 4: vol2 = vol2[..., 0]
        Z = min(vol1.shape[2], vol2.shape[2])
        center = Z // 2
        for off in offsets:
            zi = max(0, min(center + off, Z - 1))
            sl1 = _normalize_resize(vol1[:, :, zi], args.img_size)
            sl2 = _normalize_resize(vol2[:, :, zi], args.img_size)
            save_png(sl1, out_root / 'T1' / f'{sid}_z{zi:03d}.png')
            save_png(sl2, out_root / 'T2' / f'{sid}_z{zi:03d}.png')
        print(f'Wrote {sid} ({len(offsets)} slices) -> {out_root}')


if __name__ == '__main__':
    main()
