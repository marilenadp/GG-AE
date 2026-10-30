"""Example: full GG-AE pipeline. Replace the paths with your own data."""
from GGAE import make_atrophy_maps, train_genetic_vae, train_ggae

# 1. Atrophy maps: z-score patient GM maps against a CN reference cohort.
participants_csv = make_atrophy_maps(
    cn_csv="CN_participants.csv",          # column: path
    patient_csv="patients.csv",            # columns: participant_id, path, rs... SNP columns
    output_dir="results/atrophy_maps",
)

# 2. Stage 1: genetic VAE. Several runs help check stability when choosing K.
genetic_models = train_genetic_vae(
    csv_path="genetics_all_participants.csv",  # SNP columns; may include more participants
    output_dir="results/genetic_vae",
    n_latent=3,
    n_runs=5,
    id_col="participant_id",
    seed=0,
)

# 3. Stage 2: GG-AE with the genetic VAE frozen.
results = train_ggae(
    csv_path=participants_csv,
    genetic_model_path=genetic_models[0],
    output_dir="results/ggae",
    n_genetic=3,
    n_residual=1,
    id_col="participant_id",
    seed=0,
)
print(results["latents"].head())
