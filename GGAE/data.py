"""Data loading for GG-AE.

Inputs are given as a CSV file with one row per participant:
  - a column with paths to NIfTI images (e.g. ``atrophy_map_path``)
  - one column per SNP, named ``rs...`` or ``SNP...``, coded as minor allele
    counts (0/1/2). Missing genotypes may be left empty (NaN).
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def snp_columns(df):
    """Return the SNP column names (columns starting with 'rs' or 'SNP')."""
    cols = [c for c in df.columns if c.startswith("rs") or c.startswith("SNP")]
    if not cols:
        raise ValueError("No SNP columns found (expected names starting with 'rs' or 'SNP').")
    return cols


def load_images(paths, mask=None):
    """Load NIfTI images into a (n_subjects, n_voxels) float32 array.

    Args:
        paths: list of image paths. All images must share the same grid.
        mask: optional boolean voxel mask over the flattened image. If None,
            all voxels that are non-zero in at least one image are kept.

    Returns:
        data: (n_subjects, n_masked_voxels) array
        mask: boolean mask over flattened voxels
        shape: 3D image shape
        affine: affine of the first image
    """
    paths = list(paths)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) not found, e.g. {missing[:3]}. "
            "Remove these participants from the CSV before running."
        )

    first = nib.load(paths[0])
    shape = first.shape[:3]
    affine = first.affine
    data = np.empty((len(paths), int(np.prod(shape))), dtype=np.float32)
    for i, p in enumerate(paths):
        img = nib.load(p)
        if img.shape[:3] != shape:
            raise ValueError(f"Image {p} has shape {img.shape}, expected {shape}.")
        data[i] = np.nan_to_num(img.get_fdata(caching="unchanged")).ravel()

    if mask is None:
        mask = (data != 0).any(axis=0)
    return data[:, mask], mask, shape, affine


class GeneticDataset(Dataset):
    """SNP data only (stage 1: genetic VAE)."""

    def __init__(self, csv_path, id_col=None):
        df = pd.read_csv(csv_path)
        self.snp_names = snp_columns(df)
        self.ids = df[id_col].values if id_col else np.arange(len(df))
        self.data = df[self.snp_names].values.astype(np.float32)

    @property
    def n_snps(self):
        return self.data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class GGAEDataset(Dataset):
    """Paired atrophy maps and SNPs (stage 2: GG-AE).

    Each item is ``(x_img, x_snp)`` for the same participant.
    Negative values in the atrophy maps are set to 0, so the input is
    non-negative (higher = more atrophy).
    """

    def __init__(self, csv_path, path_col="atrophy_map_path", id_col=None):
        df = pd.read_csv(csv_path)
        self.ids = df[id_col].values if id_col else np.arange(len(df))
        self.snp_names = snp_columns(df)
        self.snps = df[self.snp_names].values.astype(np.float32)

        img, self.mask, self.shape, self.affine = load_images(df[path_col])
        img[img < 0] = 0
        self.img = img

    @property
    def n_voxels(self):
        return self.img.shape[1]

    @property
    def n_snps(self):
        return self.snps.shape[1]

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.img[idx]), torch.from_numpy(self.snps[idx])
