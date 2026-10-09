#!/usr/bin/env bash
set -euo pipefail

# conda activate hril
# HYDRA_FULL_ERROR=1 bash run_scripts/tri-mod_run.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ================================== tri_humor ==================================
for seed in {42..46}; do
    python main_multibench_tri-mod.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        data.data_module.dataset=humor \
        model.model.encoder.embed_dim=40 \
        model.model.encoder.n_layers=2 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.temperature=0.1 \
        model.model.loss_kwargs.w_synergy=3 \
        model.model.loss_kwargs.target=0.5 \
        model.model.loss_kwargs.w_align=10 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        mode=train \
        exp_name=tri_humor/seed_${seed}
done

# ================================== tri_mosi ==================================
for seed in {42..46}; do
    python main_multibench_tri-mod.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        data.data_module.dataset=mosi \
        model.model.encoder.embed_dim=40 \
        model.model.encoder.n_layers=2 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.temperature=0.1 \
        model.model.loss_kwargs.w_synergy=3 \
        model.model.loss_kwargs.target=0.7 \
        model.model.loss_kwargs.w_align=10 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=32 \
        model.model.loss_kwargs.proj_dim=24 \
        mode=train \
        exp_name=tri_mosi/seed_${seed}
done

# ================================== tri_mosei ==================================
for seed in {42..46}; do
    python main_multibench_tri-mod.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        data.data_module.dataset=mosei \
        model.model.encoder.embed_dim=40 \
        model.model.encoder.n_layers=2 \
        optim.lr=0.0003 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.temperature=0.1 \
        model.model.loss_kwargs.w_synergy=3 \
        model.model.loss_kwargs.target=0.5 \
        model.model.loss_kwargs.w_align=10 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        mode=train \
        exp_name=tri_mosei/seed_${seed}
done
