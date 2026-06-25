# GG-AE: Genetically-Guided Autoencoder for Brain Atrophy Pattern Decomposition

This repository contains the implementation of **GG-AE**, a biologically guided self-supervised framework for imaging-genetics analysis. GG-AE is designed to decompose imaging-derived brain atrophy patterns into genetically aligned and residual components, enabling more interpretable analysis of heterogeneous structural variation within the Alzheimer’s disease spectrum.

> Code release is currently in preparation. Additional training scripts, configuration files, and documentation will be added soon.

## Overview

GG-AE integrates genetic and imaging information in an asymmetric, biologically motivated framework. Imaging phenotypes are represented using voxel-wise interpretable NMF-derived atrophy patterns, while genetic variation is encoded into a compact latent representation. The model encourages separation of imaging variation into:

* **Genetically aligned components**, which capture atrophy variation associated with the genetic representation.
* **Residual imaging components**, which capture remaining imaging variation not explained by the genetic guidance.

The framework is self-supervised and does not rely on diagnostic labels for representation learning.

## Key Features

* Imaging-genetics representation learning
* NMF-based imaging representation for voxel-wise interpretability
* Genetic latent modeling using a VAE
* Biologically guided asymmetric fusion from genotype to phenotype
* Decomposition of atrophy patterns into genetically aligned and residual components
* Self-supervised training without diagnostic labels

## Repository Status

This repository is under active preparation. Planned contents include:

* Model implementation
* Training and evaluation scripts
* Configuration files
* Data preprocessing instructions
* Example usage
* Documentation for reproducing experiments

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{tbd,
  title     = {TBD},
  author    = {TBD},
  booktitle = {TBD},
  year      = {2026}
}
```

## License

TBD.
