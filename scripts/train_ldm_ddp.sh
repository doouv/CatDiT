#!/bin/bash
# train.sh

torchrun \
  --nproc_per_node=2\
  --standalone \
  ../src/train_diffusion.py \
   name="DiT-B__vae_latent@12_kl@0.0001_joint"

## ++<config_path>.<key>=<value> for overwrite
## ex) ++diffusion_module.denoiser.d_x=8
