#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# conda activate hril
# HYDRA_FULL_ERROR=1 bash run_scripts/trifeatures_run.sh

# ================================== trifeatures_non_syn ==================================
for seed in {42..46}; do
    python main_trifeatures.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_true \
        data.data_module.biased=false \
        model.model.encoder.embed_dim=512 \
        model.model.encoder.n_layers=1 \
        optim.lr=0.0003 \
        optim.weight_decay=0.0001 \
        model.model.loss_kwargs.temperature=0.1 \
        model.model.loss_kwargs.w_synergy=3 \
        model.model.loss_kwargs.target=0.5 \
        model.model.loss_kwargs.w_align=20 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        model.model.loss_kwargs.proj_seed=${seed} \
        mode=train \
        exp_name=trifeatures_non_syn/seed_${seed}
done

# ================================== trifeatures_syn ==================================
for seed in {42..46}; do
    python main_trifeatures.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_true \
        data.data_module.biased=true \
        model.model.encoder.embed_dim=512 \
        model.model.encoder.n_layers=1 \
        optim.lr=0.0003 \
        optim.weight_decay=0.0001 \
        model.model.loss_kwargs.temperature=0.1 \
        model.model.loss_kwargs.w_synergy=3 \
        model.model.loss_kwargs.target=0.5 \
        model.model.loss_kwargs.w_align=20 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        model.model.loss_kwargs.proj_seed=${seed} \
        mode=train \
        exp_name=trifeatures_syn/seed_${seed}
done
