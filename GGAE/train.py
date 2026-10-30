"""Training for GG-AE.

Stage 1  train_genetic_vae : learn the genetic latent z_gen from SNPs.
Stage 2  train_ggae        : with the genetic VAE frozen, learn the imaging
                             autoencoder whose latent splits into h_gen
                             (aligned to z_gen) and h_res (residual).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from .data import GeneticDataset, GGAEDataset
from .init import nndsvd
from .losses import (decoder_orthonormality, info_nce, latent_l2,
                     latent_orthogonality, vae_loss)
from .models import GeneticVAE, ImagingAE, load_genetic_vae
from .utils import EarlyStopping, get_device, save_components_as_nifti, set_seed


# =============================================================================
# Stage 1: genetic VAE
# =============================================================================

def train_genetic_vae(csv_path, output_dir, n_latent=3, n_runs=1, id_col=None,
                      max_epochs=20000, batch_size=2400, lr=1e-3,
                      kl_anneal_epochs=1000, patience=200, seed=None,
                      device=None, verbose=100):
    """Train the genetic VAE (stage 1).

    Args:
        csv_path: CSV with SNP columns (``rs...`` or ``SNP...``, coded 0/1/2).
            This can include more participants than the imaging data.
        output_dir: results are written to ``output_dir/run_<i>/``.
        n_latent: K, number of genetic latent dimensions.
        n_runs: number of independent runs. Use several runs (e.g. 5) to check
            the stability of the subject stratification when choosing K.
        id_col: optional participant ID column, used in the saved latents.
        kl_anneal_epochs: beta is increased linearly from 0 to 1 over this many epochs.
        patience: early stopping patience (epochs).
        seed: base random seed; run i uses seed + i.

    Returns:
        List of paths to the trained models (one per run).
    """
    device = get_device(device)
    dataset = GeneticDataset(csv_path, id_col=id_col)
    print(f"Genetic VAE: {len(dataset)} participants, {dataset.n_snps} SNPs, K={n_latent}")

    model_paths = []
    for run in range(n_runs):
        set_seed(None if seed is None else seed + run)
        run_dir = os.path.join(output_dir, f"run_{run + 1}")
        os.makedirs(run_dir, exist_ok=True)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model = GeneticVAE(dataset.n_snps, n_latent).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(1, max_epochs + 1):
            beta = min(1.0, epoch / kl_anneal_epochs)
            model.train()
            epoch_loss = 0.0
            for x in loader:
                x = x.to(device)
                z, q_z, p_x = model(x)
                loss = vae_loss(x, z, q_z, p_x, beta=beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)

            if verbose and epoch % verbose == 0:
                print(f"[run {run + 1}] epoch {epoch:5d} | VAE loss {epoch_loss:.6f} | beta {beta:.3f}")
            if early_stopping.step(epoch_loss):
                print(f"[run {run + 1}] early stopping at epoch {epoch}")
                break

        model_path = os.path.join(run_dir, "genetic_vae.pt")
        torch.save(model.state_dict(), model_path)
        model_paths.append(model_path)
        _save_genetic_latents(model, dataset, run_dir, device)
        print(f"[run {run + 1}] saved model to {model_path}")

    return model_paths


@torch.no_grad()
def _save_genetic_latents(model, dataset, run_dir, device):
    """Save posterior means and the dominant genetic pattern per participant."""
    model.eval()
    x = torch.from_numpy(dataset.data).to(device)
    mu = model.encode(x).mean
    emb = model.embed(mu).cpu().numpy()
    df = pd.DataFrame(emb, columns=[f"z_gen_{k}" for k in range(emb.shape[1])])
    df.insert(0, "participant_id", dataset.ids)
    df["dominant_pattern"] = emb.argmax(axis=1)
    df.to_csv(os.path.join(run_dir, "genetic_latents.csv"), index=False)


# =============================================================================
# Stage 2: genetically guided imaging autoencoder
# =============================================================================

def train_ggae(csv_path, genetic_model_path, output_dir, n_genetic=3, n_residual=1,
               path_col="atrophy_map_path", id_col=None,
               lambda_recon=10.0, lambda_W=0.1, lambda_align=0.1, temperature=0.1,
               lambda_h=1.0, lambda_l2=0.1,
               max_epochs=40000, min_epochs=100, batch_size=2400, lr=1e-2,
               patience=100, checkpoint_every=10, seed=None, device=None, verbose=100):
    """Train GG-AE (stage 2) with a frozen, pretrained genetic VAE.

    Args:
        csv_path: CSV with atrophy map paths (``path_col``) and the same SNP
            columns used to train the genetic VAE.
        genetic_model_path: path to a model saved by ``train_genetic_vae``.
        output_dir: results directory.
        n_genetic: K, size of h_gen. Must equal the genetic VAE latent size.
        n_residual: L, size of h_res.
        lambda_recon: weight of the voxel-wise MSE. 10 was used for 8 mm
            smoothed inputs; for lightly smoothed inputs (2 mm) 20 worked better.
        lambda_W: soft orthonormality of the decoder W.
        lambda_align, temperature: InfoNCE alignment weight and temperature.
        lambda_h: decorrelation between h_gen and h_res.
        lambda_l2: L2 magnitude penalty, applied to both h_gen and h_res.
        patience, min_epochs: early stopping on the total loss.

    Returns:
        dict with the trained model, decoder W, latents and output paths.
    """
    set_seed(seed)
    device = get_device(device)
    os.makedirs(output_dir, exist_ok=True)

    # --- data ---
    dataset = GGAEDataset(csv_path, path_col=path_col, id_col=id_col)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"GG-AE: {len(dataset)} participants, {dataset.n_voxels} voxels, "
          f"{dataset.n_snps} SNPs, K={n_genetic}, L={n_residual}")
    if len(dataset) > batch_size:
        print("Note: InfoNCE uses in-batch negatives; results in the paper used a single full batch.")

    # --- frozen genetic encoder ---
    gen_model = load_genetic_vae(genetic_model_path, dataset.n_snps, n_genetic, device)
    gen_model.eval()
    for p in gen_model.parameters():
        p.requires_grad_(False)

    # --- imaging autoencoder, NNDSVD-initialized decoder ---
    W_init, _ = nndsvd(dataset.img.T, n_genetic + n_residual)
    img_model = ImagingAE(dataset.n_voxels, n_genetic, n_residual, W_init).to(device)
    optimizer = torch.optim.Adam(img_model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20000, gamma=0.1)
    early_stopping = EarlyStopping(patience=patience)
    ckpt_path = os.path.join(output_dir, "ggae_model.pt")

    for epoch in range(1, max_epochs + 1):
        img_model.train()
        totals = dict(total=0.0, recon=0.0, W_ortho=0.0, align=0.0, h_ortho=0.0, l2=0.0)

        for x_img, x_snp in loader:
            x_img, x_snp = x_img.to(device), x_snp.to(device)
            img_model.project_nonnegative()

            with torch.no_grad():
                z_gen = gen_model.embed(gen_model.encode(x_snp).rsample())

            h_gen, h_res, x_hat = img_model(x_img)
            terms = dict(
                recon=lambda_recon * torch.mean((x_hat - x_img) ** 2),
                W_ortho=lambda_W * decoder_orthonormality(img_model.W),
                align=lambda_align * info_nce(h_gen, z_gen, temperature),
                h_ortho=lambda_h * latent_orthogonality(h_gen, h_res),
                l2=lambda_l2 * (latent_l2(h_gen) + latent_l2(h_res)),
            )
            loss = sum(terms.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals["total"] += loss.item()
            for k, v in terms.items():
                totals[k] += v.item()

        scheduler.step()
        totals = {k: v / len(loader) for k, v in totals.items()}

        if verbose and epoch % verbose == 0:
            print(f"epoch {epoch:5d} | " + " | ".join(f"{k} {v:.5f}" for k, v in totals.items()))
        if epoch % checkpoint_every == 0:
            torch.save(img_model.state_dict(), ckpt_path)
        if early_stopping.step(totals["total"]) and epoch >= min_epochs:
            print(f"Early stopping at epoch {epoch}")
            break

    img_model.project_nonnegative()
    torch.save(img_model.state_dict(), ckpt_path)
    return _save_ggae_outputs(img_model, dataset, output_dir, device, ckpt_path)


@torch.no_grad()
def _save_ggae_outputs(img_model, dataset, output_dir, device, ckpt_path):
    """Save spatial components (NIfTI), decoder W and subject latents."""
    img_model.eval()
    K, L = img_model.n_genetic, img_model.n_residual
    W = img_model.W.detach().cpu().numpy()  # (n_voxels, K+L)

    names = [f"component_{k}_gen" for k in range(K)] + [f"component_{K + l}_res" for l in range(L)]
    nifti_paths = save_components_as_nifti(
        W, dataset.mask, dataset.shape, dataset.affine,
        os.path.join(output_dir, "components"), names)
    np.save(os.path.join(output_dir, "W.npy"), W)

    x = torch.from_numpy(dataset.img).to(device)
    h_gen, h_res = img_model.encode(x)
    latents = np.concatenate([h_gen.cpu().numpy(), h_res.cpu().numpy()], axis=1)
    df = pd.DataFrame(latents, columns=[f"h_gen_{k}" for k in range(K)] +
                                       [f"h_res_{l}" for l in range(L)])
    df.insert(0, "participant_id", dataset.ids)
    latents_path = os.path.join(output_dir, "imaging_latents.csv")
    df.to_csv(latents_path, index=False)
    print(f"Saved components to {os.path.dirname(nifti_paths[0])} and latents to {latents_path}")

    return dict(model=img_model, W=W, latents=df, model_path=ckpt_path,
                component_paths=nifti_paths, latents_path=latents_path)
