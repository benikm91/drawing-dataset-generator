# Vitruvion's preprocessing, run unchanged, as the reference our reimplementation is checked against.
# Their image is nvcr.io/nvidia/pytorch:21.03-py3 (Python 3.8, torch 1.9, numpy 1.20); this is the
# CPU equivalent of it, with only what filter_sequences_from_source and tokenize_sequences import.
FROM python:3.8-slim

RUN pip install --no-cache-dir \
        "numpy==1.20.3" "torch==1.9.0" "hydra-core==1.1.0" "omegaconf==2.1.0" \
        "tqdm==4.61.0" "zstandard" "lz4" "scipy==1.6.3" "matplotlib==3.4.2" "pillow==8.2.0" "scikit-learn==0.24.2" "typing_extensions" "pytorch-lightning==1.4.0" "torchmetrics==0.4.1" "threadpoolctl==2.1.0"

# Vitruvion at 1b91fff, and SketchGraphs at 1f27f5f ("img2cad-compatible version") for the
# sketchgraphs.data module Vitruvion's .gitignore left out of its repository.
COPY vitruvion /opt/vitruvion
COPY SketchGraphs/sketchgraphs/data /opt/vitruvion/sketchgraphs/data
WORKDIR /opt/vitruvion
ENV PYTHONPATH=/opt/vitruvion
