# DDIM with Classifier-Free Guidance on CIFAR-10

Denoising Diffusion Implicit Models (DDIM) with Classifier-Free Guidance for conditional image generation — U-Net built from scratch in PyTorch.

## Features
- U-Net with sinusoidal time embeddings and class conditioning
- DDIM non-Markovian accelerated sampling (50 steps vs 1000)
- Classifier-Free Guidance for controllable generation quality
- FID evaluation against CIFAR-10 test set

## Setup
```bash
pip install torch torchvision matplotlib tqdm torch-fidelity
```

## Usage
```bash
jupyter notebook report.ipynb
```
