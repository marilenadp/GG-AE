"""Loss functions (paper Eqs. 1-5)."""
import torch
import torch.distributions as D
import torch.nn.functional as F


def vae_loss(x, z, q_z, p_x, beta=1.0):
    """Negative ELBO (Eq. 1), averaged over subjects.

    Missing genotypes (NaN in ``x``) are excluded from the reconstruction term.
    The KL term is a single-sample Monte Carlo estimate against N(0, I).
    """
    observed = ~torch.isnan(x)
    log_px = p_x.log_prob(torch.nan_to_num(x, nan=0.0))
    recon = -torch.masked_select(log_px, observed).sum() / x.shape[0]
    prior = D.Normal(torch.zeros_like(z), torch.ones_like(z))
    kl = (q_z.log_prob(z).sum() - prior.log_prob(z).sum()) / x.shape[0]
    return recon + beta * kl


def decoder_orthonormality(W):
    """Soft orthonormality penalty ||W^T W - I||_F^2 on the Frobenius-normalized decoder (Eq. 2)."""
    Wn = W / W.norm()
    gram = Wn.T @ Wn
    eye = torch.eye(gram.shape[0], device=W.device)
    return (gram - eye).pow(2).sum()


def info_nce(h_gen, z_gen, temperature=0.1):
    """InfoNCE alignment between imaging and genetic latents (Eq. 3).

    Positive pairs are the same subject; all other subjects in the batch are negatives.
    """
    h = F.normalize(h_gen, dim=-1)
    z = F.normalize(z_gen, dim=-1)
    logits = h @ z.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def latent_orthogonality(h_gen, h_res):
    """Decorrelation between h_gen and h_res (Eq. 4).

    Implemented as the squared Frobenius norm of the cross-covariance
    between the (batch-centered) genetic and residual latents.
    """
    h_gen = h_gen - h_gen.mean(dim=0, keepdim=True)
    h_res = h_res - h_res.mean(dim=0, keepdim=True)
    cross_cov = h_gen.T @ h_res / (h_gen.shape[0] - 1)
    return cross_cov.pow(2).sum()


def latent_l2(h):
    """Magnitude regularization on latent activations."""
    return h.pow(2).mean()
