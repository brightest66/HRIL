#!/usr/bin/env bash
set -euo pipefail

# conda activate hril
# HYDRA_FULL_ERROR=1 bash run_scripts/multibench_run.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ================================== bi_mosi ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        data.data_module.dataset=mosi \
        model.model.encoder.embed_dim=40 \
        model.model.encoder.n_layers=1 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.temperature=0.1 \
        model.model.loss_kwargs.w_synergy=1 \
        model.model.loss_kwargs.target=0.3 \
        model.model.loss_kwargs.w_align=5 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        mode=train \
        exp_name=bi_mosi/seed_${seed}
done

# ================================== bi_humor ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        data.data_module.dataset=humor \
        model.model.encoder.embed_dim=40 \
        model.model.encoder.n_layers=1 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.temperature=0.1 \
        model.model.loss_kwargs.w_synergy=3 \
        model.model.loss_kwargs.target=0.3 \
        model.model.loss_kwargs.w_align=10 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        mode=train \
        exp_name=bi_humor/seed_${seed}
done

# ================================== bi_sarcasm ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        data.data_module.dataset=sarcasm \
        model.model.encoder.embed_dim=40 \
        model.model.encoder.n_layers=1 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.temperature=0.1 \
        model.model.loss_kwargs.w_synergy=5 \
        model.model.loss_kwargs.target=0.7 \
        model.model.loss_kwargs.w_align=5 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        mode=train \
        exp_name=bi_sarcasm/seed_${seed}
done

# ================================== bi_mimic ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        data.data_module.dataset=mimic \
        model.model.encoder.embed_dim=512 \
        model.model.encoder.n_layers=1 \
        optim.lr=0.0003 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.temperature=0.2 \
        model.model.loss_kwargs.w_synergy=1 \
        model.model.loss_kwargs.target=0.5 \
        model.model.loss_kwargs.w_align=5 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        mode=train \
        exp_name=bi_mimic/seed_${seed}
done

# ================================== bi_visionandtouch ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=hril \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        data.data_module.dataset=visionandtouch-bin \
        model.model.encoder.embed_dim=512 \
        model.model.encoder.n_layers=1 \
        optim.lr=0.0001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.temperature=0.1 \
        model.model.loss_kwargs.w_synergy=5 \
        model.model.loss_kwargs.target=0.3 \
        model.model.loss_kwargs.w_align=5 \
        model.model.loss_kwargs.cov_target=0.5 \
        model.model.loss_kwargs.r=24 \
        model.model.loss_kwargs.proj_dim=24 \
        mode=train \
        exp_name=bi_visionandtouch/seed_${seed}
done
