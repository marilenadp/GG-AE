"""Utilities: seeding, early stopping, NIfTI export."""
import os
import random

import nibabel as nib
import numpy as np
import torch


def set_seed(seed):
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Stop when the monitored loss has not improved for ``patience`` epochs.

    An improvement means a decrease larger than ``min_delta`` (relative).
    A NaN loss stops training immediately.
    """

    def __init__(self, patience=200, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0

    def step(self, loss):
        if np.isnan(loss):
            return True
        if self.best is None or loss < self.best - abs(self.best * self.min_delta):
            self.best = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def revert_mask(values, mask, shape):
    """Place masked voxel values back into a 3D volume."""
    volume = np.zeros(int(np.prod(shape)))
    volume[mask] = values
    return volume.reshape(shape)


def save_components_as_nifti(W, mask, shape, affine, output_dir, names=None):
    """Save each decoder column (spatial component) as a NIfTI image.

    Args:
        W: (n_voxels, n_components) non-negative decoder matrix.
        mask, shape: voxel mask and image shape used to load the data.
        affine: affine of the input images.
        names: optional list of file name stems, one per component.

    Each component is scaled to a maximum absolute value of 1.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(W.shape[1]):
        volume = revert_mask(W[:, i], mask, shape)
        peak = np.abs(volume).max()
        if peak > 0:
            volume = volume / peak
        name = names[i] if names else f"component_{i}"
        path = os.path.join(output_dir, f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), path)
        paths.append(path)
    return paths
