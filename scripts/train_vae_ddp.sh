# !/bin/bash
# trian.sh

torchrun \
    --nproc_per_node=8 \
    --standalone \
    ../src/train_autoencoder.py \
    trainer=ddp \
    logger=wandb \
    name="vae_latent@12_kl@0.00001" \


