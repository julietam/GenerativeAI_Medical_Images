import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import nibabel as nib
from skimage.transform import resize


def _normalize_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _save_png(img01: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img01 = np.clip(img01, 0.0, 1.0)
    Image.fromarray((img01 * 255).astype(np.uint8)).save(out_path)


def prepare_ixi_pairs(ixi_root: str, out_root: str, img_size: int = 256, take_slices: int = 1) -> int:
    """Create paired T1/T2 PNG slices from IXI NIfTI files.
    Expects files with names containing subject IDs and modality tokens like 'T1' and 'T2'.
    The function matches T1/T2 by subject id prefix before the first '-' (e.g., IXI###).

    Returns number of pairs written.
    """
    ixi = Path(ixi_root)
    out = Path(out_root)
    t1_files = sorted([p for p in ixi.rglob('*.nii*') if 'T1' in p.name.upper()])
    t2_files = sorted([p for p in ixi.rglob('*.nii*') if 'T2' in p.name.upper()])

    def subj_id(p: Path) -> str:
        base = p.stem.split('.')[0]
        return base.split('-')[0]

    t1_map: Dict[str, Path] = {subj_id(p): p for p in t1_files}
    t2_map: Dict[str, Path] = {subj_id(p): p for p in t2_files}
    common_ids = sorted(set(t1_map) & set(t2_map))

    written = 0
    for sid in common_ids:
        try:
            t1_img = nib.load(str(t1_map[sid])).get_fdata()
            t2_img = nib.load(str(t2_map[sid])).get_fdata()
            # ensure same shape
            if t1_img.shape != t2_img.shape:
                min_shape = tuple(min(a, b) for a, b in zip(t1_img.shape, t2_img.shape))
                t1_img = t1_img[: min_shape[0], : min_shape[1], : min_shape[2]]
                t2_img = t2_img[: min_shape[0], : min_shape[1], : min_shape[2]]
            zc = t1_img.shape[2] // 2
            slices = [zc]
            if take_slices > 1:
                step = max(1, t1_img.shape[2] // (take_slices + 1))
                slices = list(range(step, step * (take_slices + 1), step))[: take_slices]
            for k, z in enumerate(slices):
                t1 = _normalize_minmax(t1_img[:, :, z])
                t2 = _normalize_minmax(t2_img[:, :, z])
                t1r = resize(t1, (img_size, img_size), preserve_range=True, anti_aliasing=True)
                t2r = resize(t2, (img_size, img_size), preserve_range=True, anti_aliasing=True)
                _save_png(t1r, out / 'T1' / f'{sid}_z{z:03d}_{k}.png')
                _save_png(t2r, out / 'T2' / f'{sid}_z{z:03d}_{k}.png')
                written += 1
        except Exception:
            continue
    return written


def prepare_ct_slices(ct_root: str, out_root: str, img_size: int = 256, window: Tuple[int, int] = (-1000, 1000), take_slices: int = 3) -> int:
    """Convert CT NIfTI volumes under ct_root into PNG axial slices under out_root.
    Applies HU windowing and min-max normalization per volume.

    Returns number of slices written.
    """
    root = Path(ct_root)
    out = Path(out_root)
    nii_files = sorted(root.rglob('*.nii*'))
    written = 0
    for p in nii_files:
        try:
            vol = nib.load(str(p)).get_fdata().astype(np.float32)
            # HU window
            lo, hi = window
            vol = np.clip(vol, lo, hi)
            vol = (vol - lo) / (hi - lo + 1e-6)
            zs = [vol.shape[2] // 2]
            if take_slices > 1:
                step = max(1, vol.shape[2] // (take_slices + 1))
                zs = list(range(step, step * (take_slices + 1), step))[: take_slices]
            sid = p.stem.split('.')[0]
            for k, z in enumerate(zs):
                sl = resize(vol[:, :, z], (img_size, img_size), preserve_range=True, anti_aliasing=True)
                _save_png(sl, out / f'{sid}_z{z:03d}_{k}.png')
                written += 1
        except Exception:
            continue
    return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preparation utilities')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('ixi_pairs', help='Prepare IXI T1/T2 paired PNG slices')
    p1.add_argument('--ixi_root', type=str, required=True)
    p1.add_argument('--out_root', type=str, default='data/paired_mri')
    p1.add_argument('--img_size', type=int, default=256)
    p1.add_argument('--take_slices', type=int, default=1)

    p2 = sub.add_parser('ct_slices', help='Prepare chest CT PNG slices from NIfTI volumes')
    p2.add_argument('--ct_root', type=str, required=True)
    p2.add_argument('--out_root', type=str, default='data/unpaired_ct')
    p2.add_argument('--img_size', type=int, default=256)
    p2.add_argument('--window', type=int, nargs=2, default=(-1000, 1000))
    p2.add_argument('--take_slices', type=int, default=3)

    args = parser.parse_args()
    if args.cmd == 'ixi_pairs':
        n = prepare_ixi_pairs(args.ixi_root, args.out_root, img_size=args.img_size, take_slices=args.take_slices)
        print(f'Wrote {n} IXI T1/T2 pairs to {args.out_root}')
    elif args.cmd == 'ct_slices':
        n = prepare_ct_slices(args.ct_root, args.out_root, img_size=args.img_size, window=tuple(args.window), take_slices=args.take_slices)
        print(f'Wrote {n} CT slices to {args.out_root}')
