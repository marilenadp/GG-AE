"""GG-AE: Genetically-Guided Autoencoder for Disentangling Structural Networks of Brain Atrophy."""
from .preprocessing import make_atrophy_maps
from .train import train_genetic_vae, train_ggae

__version__ = "0.1.0"
__all__ = ["make_atrophy_maps", "train_genetic_vae", "train_ggae"]
