<div align="center">
 
# DDIM with Classifier-Free Guidance on CIFAR-10
 
**COMP8221: Advanced Machine Learning — Assignment 1**
 
A from-scratch PyTorch implementation of [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) with [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) for class-conditional image generation on CIFAR-10.
 
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
 
</div>
 
---
 
## Overview
 
Standard DDPM generates high-quality images but requires all 1000 reverse steps at inference — painfully slow. This project combines two improvements that solve that:
 
| Technique | What it does | Practical impact |
|---|---|---|
| **DDIM** ([Song et al., 2021](https://arxiv.org/abs/2010.02502)) | Non-Markovian reverse process that allows timestep-skipping | **50 steps ≈ 1000 steps** — ~16× wall-clock speedup |
| **Classifier-Free Guidance** ([Ho & Salimans, 2021](https://arxiv.org/abs/2207.12598)) | Single model for conditional + unconditional generation via label dropout | Sharp, class-consistent samples without a separate classifier |
 
### Results at a Glance
 
| Metric | Value |
|---|---|
| **FID** | 13.72 |
| **Inception Score** | 9.64 ± 0.30 |
| **Training** | 400 epochs (~156k steps) on a single GPU |
| **Sampling** | 50 DDIM steps, ~0.68 s for 10 images |
 
> **Note:** Published DDIM reports FID 4.67 on CIFAR-10 with 800k+ training steps. Our gap is expected given the compute budget — the assignment does not require state-of-the-art performance.
 
---
 
## Architecture
 
A **67M-parameter U-Net** built entirely from scratch using standard `torch.nn` layers:
 
```
Input (3×32×32)
  ↓ Conv 3→128
Encoder:    128 → 256 → 512 channels (ResBlocks + Self-Attention at 16×16, 8×8)
Bottleneck: ResBlock → Attention → ResBlock at 512 ch, 8×8
Decoder:    Mirror of encoder with skip connections
  ↓ GroupNorm → SiLU → Conv 128→3
Output (3×32×32) — predicted noise ε̂
```
 
**Key components:**
- **Sinusoidal time embeddings** → 2-layer MLP, injected via FiLM (scale-shift) in every ResBlock
- **Learnable class embeddings** — 11 entries (10 CIFAR-10 classes + 1 null token for CFG)
- **Multi-head self-attention** at 16×16 and 8×8 resolutions
- **Cosine noise schedule** for uniform SNR progression across timesteps
- **EMA** (decay 0.9999) of model weights for stable generation
 
---
 
## Project Structure
 
```
ddim-cfg-cifar10/
├── report.ipynb               # Full notebook — implementation, training, evaluation
├── README.md
├── checkpoints/                # Saved model weights
│   ├── best_model.pt
│   ├── final_model.pt
│   └── checkpoint_epoch*.pt
├── data/                       # CIFAR-10 (auto-downloaded)
└── results/
    ├── loss_curves/
    ├── samples/
    ├── diffusion_process/
    └── fid/
        ├── generated/           # 10k generated images
        └── real/                # 10k real test images
```
 
---
 
## Getting Started
 
### Requirements
 
```
Hardware: NVIDIA GPU with ≥12 GB VRAM (trained on RTX 6000 Ada, 48 GB)
OS:       Linux (tested on Ubuntu 22.04)
Python:   3.10+
```
 
### Installation
 
```bash
git clone https://github.com/AlirezaYegane/ddim-cfg-cifar10.git
cd ddim-cfg-cifar10
pip install torch torchvision matplotlib tqdm torch-fidelity numpy
```
 
### Run
 
**Full pipeline (train + evaluate + visualise):**
```bash
jupyter notebook report.ipynb
# Run All Cells
# If a checkpoint exists, training is skipped automatically
```
 
**Headless training:**
```bash
jupyter nbconvert --to script report.ipynb
python report.py
```
 
**Resume from checkpoint:**
```python
checkpoint = torch.load('checkpoints/final_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```
 
> Training takes ~9–10 hours from scratch on an RTX 6000 Ada. If `checkpoints/final_model.pt` exists, the notebook loads it and skips directly to evaluation.
 
---
 
## Training Configuration
 
| Category | Parameter | Value |
|---|---|---|
| **Model** | Base channels | 128 |
| | Channel multipliers | (1, 2, 4) → 128, 256, 512 |
| | Attention resolutions | 16×16, 8×8 |
| | Dropout | 0.1 |
| **Diffusion** | Timesteps | 1000 |
| | Schedule | Cosine |
| | DDIM sampling steps | 50 |
| | η | 0.0 (deterministic) |
| **CFG** | Unconditional dropout | 10% |
| | Guidance scale | 3.0 |
| **Training** | Optimizer | AdamW (lr = 2e-4) |
| | Batch size | 128 |
| | Epochs | 400 |
| | EMA decay | 0.9999 |
| | Gradient clipping | max norm 1.0 |
| | LR schedule | 1k-step warmup + cosine decay |
 
---
 
## What's in the Notebook
 
The notebook is self-contained — every component is implemented, trained, and evaluated in sequence:
 
1. **Implementation** — U-Net, DDIM scheduler, CFG sampling, MSE loss — all from scratch
2. **Data pipeline** — CIFAR-10 with [-1, 1] normalisation + horizontal flip
3. **Training** — 400 epochs with EMA, warmup, cosine LR, gradient clipping
4. **Quantitative evaluation** — FID and Inception Score on 10k generated samples
5. **Qualitative analysis:**
   - 10×10 class-conditional sample grid
   - Reverse diffusion visualisation (x_T → x_0)
   - DDIM step-count comparison (10, 25, 50, 100, 200, 1000 steps)
   - CFG guidance scale ablation (w = 0, 1, 3, 7.5, 15)
   - Stochasticity comparison (η = 0 → 1)
 
---
 
## References
 
```
@inproceedings{ho2020denoising,
  title     = {Denoising Diffusion Probabilistic Models},
  author    = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle = {NeurIPS},
  year      = {2020}
}
 
@inproceedings{song2021denoising,
  title     = {Denoising Diffusion Implicit Models},
  author    = {Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle = {ICLR},
  year      = {2021}
}
 
@inproceedings{ho2021classifierfree,
  title     = {Classifier-Free Diffusion Guidance},
  author    = {Ho, Jonathan and Salimans, Tim},
  booktitle = {NeurIPS Workshop on Deep Generative Models},
  year      = {2021}
}
 
@inproceedings{nichol2021improved,
  title     = {Improved Denoising Diffusion Probabilistic Models},
  author    = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
  booktitle = {ICML},
  year      = {2021}
}
```
 
---
 
<div align="center">
 
**COMP8221 · Advanced Machine Learning · 2026 S1**
 
</div>