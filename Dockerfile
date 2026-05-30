FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget vim tar && rm -rf /var/lib/apt/lists/*

# fairchem first (needed for OC20/OC22 LMDB loading during training).
# it may pull torch 2.4; the next step pins torch back to 2.3.1 to match the
# base image and the PyG (pt23) wheels.
RUN pip install --no-cache-dir fairchem-core==1.10.0

RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# conda-only deps (openbabel has no usable pip wheel; pin libstdcxx runtime)
RUN conda install -y -c conda-forge libstdcxx-ng=14.2.0 openbabel=3.1.1 && \
    conda clean -ya

# PyG extensions matching torch 2.3.1+cu121
RUN pip install --no-cache-dir torch-geometric==2.6.1 && \
    pip install --no-cache-dir torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

RUN pip install --no-cache-dir \
    lightning==2.4.0 hydra-core==1.3.2 hydra-colorlog omegaconf==2.3.0 \
    e3nn==0.5.1 sevenn==0.12.1 pymatgen ase pyxtal==0.6.7 matminer==0.9.2 \
    wandb rootutils rich==14.0.0 pathos p-tqdm torchdiffeq huggingface_hub \
    pandas seaborn joblib scikit-learn svgwrite cairosvg reportlab \
    PyYAML importlib_resources "setuptools==75.8.0" "charset-normalizer==3.4.1" \
    "numpy<2"

COPY . /workspace

CMD ["bash"]

# Usage:
#   docker build -t catdit .
#   docker run --gpus all -v /path/to/data:/workspace/data -it catdit
