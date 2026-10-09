#!/usr/bin/env bash
set -euo pipefail

# conda activate hril
# HYDRA_FULL_ERROR=1 bash run_scripts/MM_IMDb_run.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ================================== mmimdb ==================================
for seed in {42..46}; do
    python main_mmimdb.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        model.model.encoder.embed_dim=768 \
        model.model.encoder.n_layers=1 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.temperature=0.2 \
        model.model.loss_kwargs.w_synergy=5 \
        model.model.loss_kwargs.target=0.3 \
        model.model.loss_kwargs.w_align=5 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        mode=train \
        exp_name=mmimdb/seed_${seed}
done
