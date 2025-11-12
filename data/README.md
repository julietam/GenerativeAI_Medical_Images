# Open datasets and preparation

This repo expects simple 2D PNG layouts for the notebooks:

- Pix2Pix (T1→T2): data/paired_mri/
  - T1/*.png
  - T2/*.png (same count and aligned by filename prefix)
- CycleGAN (MRI↔CT):
  - data/unpaired_mri/*.png
  - data/unpaired_ct/*.png

You can generate these PNGs from open 3D datasets using the included utilities.

Brain MRI (IXI) — T1/T2 pairs
1) Download IXI (open) NIfTI volumes (T1 and T2) from the IXI dataset site into a folder, preserving filenames that contain the subject ID and modality (e.g., IXI###-...-T1.nii.gz, IXI###-...-T2.nii.gz).
2) Create paired PNGs (center or a few axial slices), resized to 256x256:

   Windows PowerShell
   python -m src.data.datasets ixi_pairs --ixi_root "C:\\path\\to\\IXI" --out_root data/paired_mri --img_size 256 --take_slices 1

   - This writes PNGs to data/paired_mri/T1 and data/paired_mri/T2.
   - Increase --take_slices to extract multiple slices per subject.

Chest CT (e.g., MosMedData) — Unpaired CT PNGs
1) Obtain chest CT NIfTI volumes (e.g., MosMedData CT). Place them in a folder (any depth).
2) Extract PNG axial slices (center or a few slices) with windowing to [-1000, 1000] HU:

   Windows PowerShell
   python -m src.data.datasets ct_slices --ct_root "C:\\path\\to\\CT_NIfTI" --out_root data/unpaired_ct --img_size 256 --take_slices 3

   - This writes data/unpaired_ct/*.png, suitable as domain B for CycleGAN.

How notebooks consume data
- 03_pix2pix_mri.ipynb expects data/paired_mri/T1 and data/paired_mri/T2.
- 04_cyclegan_ct_mri.ipynb expects data/unpaired_mri and data/unpaired_ct. You can point data/unpaired_mri to the T1 or T2 PNGs from IXI (e.g., data/paired_mri/T1).

Tips
- If you have DICOM or other formats, convert to NIfTI first (e.g., dcm2niix). Then use the scripts above.
- For more sophisticated pipelines, check MONAI’s tutorials for IXI/MSD downloads and transforms; the current notebooks favor a lightweight 2D PNG workflow for teaching.
