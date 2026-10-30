"""Construction of voxel-wise atrophy maps (paper Section 3.1).

Input images are assumed to be gray matter maps already in MNI space,
smoothed and resampled (in the paper: multi-atlas segmentation, deformable
registration to MNI space, 8 mm FWHM smoothing, 2x2x2 mm voxels).

Each patient image is z-scored voxel-wise against a cognitively normal (CN)
reference cohort. Low-variance voxels are masked, z-scores are clipped, and
the sign is inverted so that higher values correspond to greater atrophy.
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd

from .data import load_images
from .utils import revert_mask


def make_atrophy_maps(cn_csv, patient_csv, output_dir, path_col="path",
                      id_col="participant_id", std_threshold=10.0, clip=10.0,
                      output_col="atrophy_map_path"):
    """Create CN-referenced atrophy maps for every patient.

    Args:
        cn_csv: CSV listing the CN reference images (column ``path_col``).
        patient_csv: CSV listing patient images (columns ``path_col`` and ``id_col``).
            Any other columns (e.g. SNPs) are kept in the output CSV.
        output_dir: where atrophy maps and the updated CSV are written.
        std_threshold: voxels whose CN standard deviation is below this value
            are masked (set to 0). The default (10) assumes intensity-scaled
            gray matter maps; adjust it to the intensity range of your images.
        clip: z-scores are clipped to [-clip, clip].
        output_col: name of the new CSV column with atrophy map paths.

    Returns:
        Path to the output CSV, which can be passed directly to ``train_ggae``.
    """
    os.makedirs(output_dir, exist_ok=True)
    maps_dir = os.path.join(output_dir, "atrophy_maps")
    os.makedirs(maps_dir, exist_ok=True)

    # --- CN reference statistics ---
    cn_df = pd.read_csv(cn_csv)
    x_cn, mask, shape, affine = load_images(cn_df[path_col])
    x_cn[x_cn < 0] = 0
    mean = x_cn.mean(axis=0)
    std = x_cn.std(axis=0)
    low_std = std < std_threshold
    std[low_std] = 1.0
    mean[low_std] = 0.0
    print(f"CN reference: {len(cn_df)} images, {mask.sum()} voxels, "
          f"{low_std.sum()} low-variance voxels masked")
    del x_cn

    # --- Patient z-scores ---
    pt_df = pd.read_csv(patient_csv)
    x_pt, _, pt_shape, _ = load_images(pt_df[path_col], mask=mask)
    if pt_shape != shape:
        raise ValueError(f"Patient images have shape {pt_shape}, CN images {shape}.")
    x_pt[x_pt < 0] = 0
    z = (x_pt - mean) / std
    z[:, low_std] = 0.0
    z = np.clip(z, -clip, clip)
    atrophy = -z  # higher = more atrophy

    out_paths = []
    for subj_id, row in zip(pt_df[id_col], atrophy):
        volume = revert_mask(row, mask, shape).astype(np.float32)
        out_path = os.path.join(maps_dir, f"{subj_id}_atrophy.nii.gz")
        nib.save(nib.Nifti1Image(volume, affine), out_path)
        out_paths.append(out_path)

    pt_df[output_col] = out_paths
    out_csv = os.path.join(output_dir, "participants_with_atrophy_maps.csv")
    pt_df.to_csv(out_csv, index=False)
    print(f"Saved {len(out_paths)} atrophy maps; CSV written to {out_csv}")
    return out_csv
