import argparse
import os
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from skimage.transform import resize
from PIL import Image


def to_png_slice(vol_path: Path, img_size: int = 256, z_policy: str = "center") -> np.ndarray:
    img = nib.load(str(vol_path))
    data = img.get_fdata().astype(np.float32)
    # choose axial slice
    if data.ndim == 4:
        data = data[..., 0]
    z = data.shape[2] // 2 if z_policy == "center" else max(0, int(z_policy))
    sl = data[:, :, z]
    # min-max normalize per volume slice
    mn, mx = float(sl.min()), float(sl.max())
    if mx > mn:
        sl = (sl - mn) / (mx - mn)
    else:
        sl = np.zeros_like(sl, dtype=np.float32)
    sl = resize(sl, (img_size, img_size), preserve_range=True, anti_aliasing=True)
    sl = np.clip(sl, 0.0, 1.0)
    return sl


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
    args = ap.parse_args()

    ds_root = Path(args.ds_root)
    out_root = Path(args.out_root)

    pairs = find_subject_pairs(ds_root, max_pairs=args.max_pairs)
    if not pairs:
        print('No T1w/T2w pairs found.')
        return

    for t1p, t2p, sid in pairs:
        t1 = to_png_slice(t1p, img_size=args.img_size)
        t2 = to_png_slice(t2p, img_size=args.img_size)
        save_png(t1, out_root / 'T1' / f'{sid}.png')
        save_png(t2, out_root / 'T2' / f'{sid}.png')
        print(f'Wrote {sid} -> {out_root}')


if __name__ == '__main__':
    main()
