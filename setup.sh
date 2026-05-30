#!/bin/bash
# CatDiT environment setup. Run on a CUDA-capable machine (prepend sudo to apt if non-root).
set -e

apt update && apt install -y git wget vim tar

conda install -y -n base -c conda-forge mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
MAMBA="$(conda info --base)/bin/mamba"   # full path: works after activate
conda create -y -n catdit python=3.10 -c defaults
conda activate catdit

# fairchem first; the next step pins torch back to 2.3.1
pip install fairchem-core==1.10.0

$MAMBA install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -c defaults

$MAMBA install -y -c conda-forge libstdcxx-ng=14.2.0
pip install torch-geometric==2.6.1 torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

pip install lightning==2.4.0 hydra-core==1.3.2 hydra-colorlog

$MAMBA install -y ase matminer==0.9.2 openbabel==3.1.1 pandas seaborn joblib scikit-learn \
    yaml importlib_resources -c conda-forge


pip install pyxtal==0.6.7 e3nn==0.5.1 sevenn==0.12.1 wandb rootutils rich==14.0.0 \
    pathos p-tqdm svgwrite cairosvg reportlab torchdiffeq huggingface_hub

# pyxtal needs pkg_resources (removed in setuptools>=81); requests needs charset-normalizer
pip install "setuptools==75.8.0" "charset-normalizer==3.4.1"

# keep numpy < 2 (must be last)
pip install "numpy<2"

python -c "import torch, torchvision; print('torch', torch.__version__, '/ tv', torchvision.__version__)"
python -c "from pymatgen.util import coord_cython; print('pymatgen ok')"
