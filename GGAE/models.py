"""GG-AE models.

GeneticVAE (paper Section 2.1)
    Encodes SNP minor allele counts into a K-dimensional latent z_gen and
    reconstructs them with a binomial likelihood (2 trials per SNP).

ImagingAE (paper Section 2.2)
    Linear encoder + batch norm + sigmoid gives the imaging latent
    h = [h_gen, h_res] (first K units genetically aligned, last L residual).
    A bias-free linear decoder W (n_voxels x (K+L)) is kept non-negative,
    so each column of W is an additive spatial pattern, analogous to NMF.
"""
import torch
import torch.distributions as D
import torch.nn as nn


def _kaiming_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")


class GeneticVAE(nn.Module):
    def __init__(self, n_snps, n_latent):
        super().__init__()
        if n_snps < 4:
            raise ValueError("GeneticVAE needs at least 4 SNPs.")
        self.n_latent = n_latent
        self.encoder = nn.Sequential(
            nn.Linear(n_snps, n_snps // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_snps // 2, n_snps // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_snps // 4, 2 * n_latent),
        )
        self.decoder = nn.Linear(n_latent, n_snps)
        self.apply(_kaiming_init)

    def encode(self, x):
        """Return the approximate posterior q(z | x) as a diagonal Normal.

        Missing genotypes (NaN) are set to 0 at the encoder input; they are
        excluded from the reconstruction loss.
        """
        mu, log_scale = self.encoder(torch.nan_to_num(x, nan=0.0)).chunk(2, dim=-1)
        return D.Normal(mu, log_scale.exp())

    def decode(self, z):
        """Return p(x | z) as a Binomial(2, logits) distribution."""
        logits = self.decoder(torch.sigmoid(z))
        return D.Binomial(total_count=2, logits=logits)

    def embed(self, z):
        """Genetic embedding used for alignment with the imaging latent."""
        return torch.sigmoid(z)

    def forward(self, x):
        q_z = self.encode(x)
        z = q_z.rsample()
        return z, q_z, self.decode(z)


class ImagingAE(nn.Module):
    def __init__(self, n_voxels, n_genetic, n_residual, W_init):
        """
        Args:
            n_voxels: number of input voxels.
            n_genetic: K, size of the genetically aligned latent h_gen.
            n_residual: L, size of the residual latent h_res.
            W_init: (n_voxels, K+L) non-negative initial decoder (e.g. NNDSVD).
        """
        super().__init__()
        self.n_genetic = n_genetic
        self.n_residual = n_residual
        n_components = n_genetic + n_residual
        self.encoder = nn.Linear(n_voxels, n_components, bias=False)
        self.norm = nn.BatchNorm1d(n_components)
        self.decoder = nn.Linear(n_components, n_voxels, bias=False)
        self.decoder.weight = nn.Parameter(torch.as_tensor(W_init, dtype=torch.float32))

    @property
    def W(self):
        """Decoder matrix, (n_voxels, K+L)."""
        return self.decoder.weight

    @torch.no_grad()
    def project_nonnegative(self, eps=1e-16):
        """Clamp the decoder to be non-negative (projection step)."""
        self.decoder.weight.clamp_(min=eps)

    def encode(self, x):
        """Return (h_gen, h_res)."""
        h = torch.sigmoid(self.norm(self.encoder(x)))
        return h[:, :self.n_genetic], h[:, self.n_genetic:]

    def decode(self, h_gen, h_res):
        return self.decoder(torch.cat([h_gen, h_res], dim=1))

    def forward(self, x):
        h_gen, h_res = self.encode(x)
        return h_gen, h_res, self.decode(h_gen, h_res)


# --- Loading checkpoints saved by the original research code ---------------

_LEGACY_IMAGING_KEYS = {"encoder3.": "encoder.", "decoder1.": "decoder."}


def load_genetic_vae(path, n_snps, n_latent, device="cpu"):
    """Load a GeneticVAE state dict (also accepts the original research checkpoints)."""
    state = torch.load(path, map_location=device)
    if "gen_model" in state:
        state = state["gen_model"]
    state = {k: v for k, v in state.items() if not k.startswith("norm.")}
    model = GeneticVAE(n_snps, n_latent)
    model.load_state_dict(state)
    return model.to(device)


def convert_legacy_imaging_state(state):
    """Rename keys of an imaging AE checkpoint from the original research code."""
    out = {}
    for k, v in state.items():
        for old, new in _LEGACY_IMAGING_KEYS.items():
            if k.startswith(old):
                k = new + k[len(old):]
        out[k] = v
    return out
