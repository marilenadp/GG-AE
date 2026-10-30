# GG-AE

**GG-AE: Genetically-Guided Autoencoder for Disentangling Structural Networks of Brain Atrophy**
Marilena De Pian, Junhao Wen, Christos Davatzikos — MICCAI 2026
[Paper](https://papers.miccai.org/miccai-2026/paper/5853_paper.pdf)

GG-AE is a self-supervised framework that uses disease-associated genetic variation as a biological anchor for learning brain atrophy patterns. A variational autoencoder first learns a compact genetic representation from SNPs. An imaging autoencoder with a non-negative, part-based (NMF-like) decoder then decomposes voxel-wise atrophy maps into a **genetically aligned** subspace and a **residual** subspace that captures non-genetic variation.

## License

Copyright (c) 2026 University of Pennsylvania. All rights reserved. See https://www.cbica.upenn.edu/sbia/software/license.html

## Installation

We recommend a conda environment:

```bash
conda create --name ggae python=3.10
conda activate ggae
git clone https://github.com/marilenadp/GG-AE.git
cd GG-AE
pip install -e .
```

A GPU is recommended for stage 2 (voxel-wise data).

## Input structure

GG-AE takes CSV files with one row per participant.

**Images.** Gray matter maps registered to a common template (e.g. MNI space), smoothed and resampled, all on the same voxel grid. In the paper, maps were obtained with multi-atlas segmentation, registered with deformable registration, smoothed with an 8 mm FWHM Gaussian kernel, and resampled to 2×2×2 mm. Preprocessing is not part of this package.

**Genetics.** One column per SNP, with names starting with `rs` or `SNP`, coded as minor allele counts (0, 1, 2). Missing genotypes can be left empty; they are excluded from the reconstruction loss.

Example patient CSV:

```
participant_id    path                      rs429358    rs3851179    rs6656401 ...
subject-1         /data/subject-1_GM.nii.gz  1           0            2
subject-2         /data/subject-2_GM.nii.gz  0           1            0
subject-3         /data/subject-3_GM.nii.gz  2           NaN          1
```

Example CN reference CSV:

```
path
/data/cn-1_GM.nii.gz
/data/cn-2_GM.nii.gz
```

## Usage

The pipeline has three steps. A complete script is in [`example/run_example.py`](example/run_example.py).

### 1. Atrophy maps

```python
from GGAE import make_atrophy_maps

participants_csv = make_atrophy_maps(
    cn_csv="CN_participants.csv",
    patient_csv="patients.csv",
    output_dir="results/atrophy_maps",
)
```

Each patient map is z-scored voxel-wise against the CN reference cohort. Voxels with low variance in the CN cohort (`std_threshold`) are masked, z-scores are clipped to ±10, and the sign is inverted so that higher values mean more atrophy. The returned CSV adds an `atrophy_map_path` column and can be passed directly to step 3.

`std_threshold` (default 10) depends on the intensity scale of your gray matter maps; adjust it if your images use a different range.

### 2. Stage 1: genetic VAE

```python
from GGAE import train_genetic_vae

genetic_models = train_genetic_vae(
    csv_path="genetics.csv",
    output_dir="results/genetic_vae",
    n_latent=3,       # K
    n_runs=5,
    id_col="participant_id",
)
```

The genetic VAE can be trained on all participants with genotype data, including those without imaging. Each run saves the model (`genetic_vae.pt`) and the participants' genetic latents with their dominant pattern (`genetic_latents.csv`).

**Choosing K.** Without ground truth, train several runs for each candidate K (we used K = 1–5, 5 runs each) and choose the largest K that gives a consistent stratification of participants (`dominant_pattern`) across runs.

### 3. Stage 2: GG-AE

```python
from GGAE import train_ggae

results = train_ggae(
    csv_path=participants_csv,
    genetic_model_path=genetic_models[0],
    output_dir="results/ggae",
    n_genetic=3,      # K, must match the genetic VAE
    n_residual=1,     # L
    id_col="participant_id",
)
```

The genetic VAE is frozen, and the imaging autoencoder is trained with reconstruction, alignment, orthogonality and magnitude losses.

## Outputs

In the stage 2 output directory:

| File | Content |
|---|---|
| `components/component_<i>_gen.nii.gz` | Genetically aligned spatial patterns (columns of W) |
| `components/component_<i>_res.nii.gz` | Residual spatial patterns |
| `imaging_latents.csv` | Per-participant loadings `h_gen_*` and `h_res_*` |
| `W.npy` | Decoder matrix (masked voxels × components) |
| `ggae_model.pt` | Trained imaging autoencoder |

The per-participant `h_gen` loadings can be tested for association with SNPs, and the `h_res` loadings should show no significant genetic association.

## Hyperparameters

Default values are those used for the ADNI experiments in the paper.

| Argument | Default | Description |
|---|---|---|
| `n_genetic` (K) | 3 | Genetically aligned latent dimensions; must equal the genetic VAE's `n_latent` |
| `n_residual` (L) | 1 | Residual latent dimensions |
| `lambda_recon` | 10 | Weight of the voxel-wise MSE. We used 10 for 8 mm smoothed inputs; for lightly smoothed inputs (2 mm), 20 worked better |
| `lambda_W` | 0.1 | Soft orthonormality of the decoder W |
| `lambda_align` | 0.1 | InfoNCE alignment between h_gen and z_gen |
| `temperature` | 0.1 | InfoNCE temperature |
| `lambda_h` | 1 | Decorrelation between h_gen and h_res |
| `lambda_l2` | 0.1 | L2 penalty on h_gen and h_res |
| `batch_size` | 2400 | InfoNCE uses in-batch negatives; the paper's results used the full dataset as one batch |
| `lr` | 1e-2 | Learning rate of the imaging autoencoder |
| `patience` | 100 | Early stopping patience (epochs) |

For the genetic VAE: learning rate 1e-3, β annealed linearly from 0 to 1 over the first 1000 epochs (`kl_anneal_epochs`), early stopping patience 200.

Pass `seed` to `train_genetic_vae` and `train_ggae` for reproducible runs. Results may vary slightly between runs and hardware.

## Acknowledgments

The NNDSVD initialization is adapted from [pyOPNMF](https://github.com/anbai106/pyOPNMF) by Junhao Wen. This work was partially supported by NIH grants RF1MH134236, R01AG054409, and U01AG068057.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{depian2026ggae,
  title     = {{GG-AE}: Genetically-Guided Autoencoder for Disentangling Structural Networks of Brain Atrophy},
  author    = {De Pian, Marilena and Wen, Junhao and Davatzikos, Christos},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026}
}
```

## Contact

Marilena De Pian — depianm@pennmedicine.upenn.edu
