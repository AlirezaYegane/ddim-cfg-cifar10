<p align="center">
  <img src="logo/logo.png" alt="Macquarie University" width="220"/>
</p>

<h1 align="center">From-Scratch DDIM with Classifier-Free Guidance on CIFAR-10</h1>

<p align="center">
  <strong>COMP8221 — Advanced Machine Learning</strong><br/>
  <strong>Assignment 1 — Option 3: Diffusion Models</strong>
</p>

<p align="center">
  A from-scratch PyTorch implementation of Denoising Diffusion Implicit Models (DDIM)<br/>
  with Classifier-Free Guidance (CFG) for class-conditional image generation on CIFAR-10.
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-CIFAR--10-brightgreen">
  <img alt="Model" src="https://img.shields.io/badge/Model-DDIM%20%2B%20CFG-purple">
</p>

---

## Overview

This repository accompanies my COMP8221 Assignment 1 submission for **Option 3: Diffusion Models**. The project implements a **non-standard diffusion model** based on **DDIM** and extends it with **Classifier-Free Guidance (CFG)** for class-conditional image generation on CIFAR-10.

I chose this combination for a practical reason:
- **DDIM** reduces sampling cost by replacing the standard DDPM reverse chain with a **non-Markovian** update rule that supports timestep skipping.
- **CFG** improves class fidelity and conditional control without requiring a separate classifier.

Together, they form a technically meaningful diffusion variant that fits the assignment rubric well while remaining manageable to implement and analyse from scratch.

---

## Project Snapshot

| Item | Details |
|---|---|
| **Student** | Alireza Yegane |
| **Student ID** | 60957107 |
| **Unit** | COMP8221 — Advanced Machine Learning |
| **Assignment Option** | Option 3 — Diffusion Models |
| **Model** | DDIM + Classifier-Free Guidance |
| **Dataset** | CIFAR-10 (32×32 RGB, 10 classes) |
| **Framework** | PyTorch |
| **Main notebook** | `2026S1 COMP8221 Assignment 1 60957107 Alireza Yegane.ipynb` |
| **Repository** | `github.com/AlirezaYegane/ddim-cfg-cifar10` |

---

## Results at a Glance

| Metric | Value |
|---|---:|
| **FID** | **13.72** |
| **Inception Score** | **9.64 ± 0.30** |
| **Training length** | **400 epochs (~156k steps)** |
| **Sampling setup** | **50 DDIM steps, η = 0.0, guidance scale = 3.0** |
| **Runtime** | **~0.68 s for 10 images** |

> These results are not intended to match large-scale published diffusion benchmarks. The focus of the project is on **correct implementation, clear analysis, and strong rubric alignment** rather than state-of-the-art performance.

---

## Why This Project Fits Option 3

This submission is intentionally built around **DDIM**, not a basic DDPM baseline.

### What makes it a valid diffusion variant?
- The **forward noising process** follows the standard diffusion setup used during training.
- The **reverse process** is changed to DDIM's **non-Markovian sampler**, which allows fewer reverse steps.
- The model is extended with **Classifier-Free Guidance**, so the same network supports both conditional and unconditional predictions.

In other words, this is not a workshop baseline reproduced verbatim. It is a **from-scratch implementation of a diffusion variant** with a distinct sampling procedure and a meaningful conditional generation mechanism.

---

## Repository Structure

```text
.
├── 2026S1 COMP8221 Assignment 1 60957107 Alireza Yegane.ipynb
├── README.md
├── checkpoints/
│   ├── best_model.pt
│   ├── final_model.pt
│   └── checkpoint_epoch*.pt
├── data/
│   └── CIFAR-10 (auto-downloaded)
└── results/
    ├── loss_curves/
    ├── samples/
    ├── diffusion_process/
    └── fid/
        ├── generated/
        └── real/
```

---

## Method Summary

### 1) U-Net Noise Predictor
The core model is a **from-scratch U-Net** that receives:
- a noisy image `x_t`,
- a timestep `t`,
- and a class label `c`,

and predicts the added noise `ε̂`.

### 2) Time Conditioning
Timesteps are encoded using **sinusoidal embeddings** followed by a small MLP, allowing the network to adapt to different noise levels across the diffusion trajectory.

### 3) Class Conditioning for CFG
The model uses a learnable class embedding table with **11 entries**:
- 10 CIFAR-10 classes,
- plus 1 **null token** for unconditional generation.

During training, class labels are dropped with probability `p_uncond = 0.1`, so the same model learns both conditional and unconditional behaviour.

### 4) FiLM-Based Conditioning
Time and class information are fused and injected into residual blocks through **FiLM-style scale-shift modulation**, allowing the network to adapt feature processing based on both timestep and class.

### 5) Self-Attention
Self-attention is applied at **16×16** and **8×8** resolutions, where global context is useful and computational cost is still reasonable.

### 6) DDIM Scheduler
The project implements:
- the closed-form **forward process** used during training,
- and the **DDIM reverse process** used during accelerated sampling.

Both **linear** and **cosine** schedules are implemented, with the **cosine schedule** used for the final model.

---

## Training Objective

The model is trained with the standard **noise-prediction MSE objective** used in diffusion models:
1. sample a clean image `x0`,
2. sample a random timestep `t`,
3. add Gaussian noise to obtain `x_t`,
4. predict the noise with the U-Net,
5. minimise the mean squared error between predicted noise and true noise.

This objective is simple, stable, and directly compatible with DDIM sampling at inference time.

---

## Dataset and Preprocessing

### Why CIFAR-10?
I use **CIFAR-10** because it is:
- a standard low-resolution benchmark for generative image modelling,
- computationally realistic for a from-scratch course project,
- and naturally suited to conditional generation because it provides 10 balanced semantic classes.

### Preprocessing pipeline
The data pipeline includes:
- conversion to tensor,
- scaling to the `[-1, 1]` range,
- and horizontal flipping during training.

This preprocessing is appropriate for diffusion training and is kept intentionally simple so the model behaviour remains interpretable.

---

## Training Configuration

| Category | Parameter | Value |
|---|---|---:|
| **Model** | Base channels | 128 |
|  | Channel multipliers | (1, 2, 4) |
|  | Residual blocks per level | 2 |
|  | Attention resolutions | 16, 8 |
|  | Dropout | 0.1 |
| **Diffusion** | Timesteps | 1000 |
|  | Beta schedule | Cosine |
| **CFG** | Unconditional dropout | 0.1 |
|  | Guidance scale | 3.0 |
| **Training** | Batch size | 128 |
|  | Learning rate | 2e-4 |
|  | Epochs | 400 |
|  | EMA decay | 0.9999 |
| **Sampling** | DDIM steps | 50 |
|  | η | 0.0 |

The training loop also includes:
- random timestep sampling,
- gradient clipping,
- EMA model tracking,
- checkpoint saving,
- and logged training curves.

---

## What the Notebook Contains

The notebook is written as a complete technical report rather than a raw experiment log. It includes:

1. **Introduction and motivation**
2. **Setup and configuration**
3. **Model architecture**
4. **Diffusion process and DDIM scheduler**
5. **Loss function and training utilities**
6. **Dataset and preprocessing**
7. **Training procedure**
8. **Quantitative evaluation**
9. **Qualitative analysis and visualisation**
10. **Discussion and limitations**
11. **Conclusion**
12. **Reproducibility notes and appendices**

---

## Qualitative Analysis Included

The final notebook contains:

- **class-conditional sample grids**
- **reverse diffusion visualisation** from pure noise to final image
- **DDIM step-count comparisons**
- **guidance-scale ablations**
- **stochasticity comparisons** across different `η` values
- and a short discussion of **failure modes and limitations**

---

## Rubric Alignment

| Rubric Area | Evidence in This Submission |
|---|---|
| **Implementation** | From-scratch U-Net, time embedding, class embedding, FiLM conditioning, self-attention, DDIM scheduler |
| **Dataset & Preprocessing** | CIFAR-10 justification, `[-1, 1]` scaling, simple augmentation |
| **Training & Quantitative Evaluation** | Training loop, random timestep sampling, MSE objective, FID, Inception Score |
| **Analysis & Qualitative Visualisation** | Pure-noise sampling, reverse trajectory, sample grids, ablations |
| **Report & Code Quality** | Structured notebook, modular code, reproducibility notes, clear explanations |

---

## How to Run

### Environment
- Python 3.10+
- PyTorch
- torchvision
- matplotlib
- tqdm
- numpy
- torch-fidelity

### Install

```bash
git clone https://github.com/AlirezaYegane/ddim-cfg-cifar10.git
cd ddim-cfg-cifar10
pip install torch torchvision matplotlib tqdm torch-fidelity numpy
```

### Main usage

```bash
jupyter notebook "2026S1 COMP8221 Assignment 1 60957107 Alireza Yegane.ipynb"
```

Then run all cells in order.

### Headless execution

```bash
jupyter nbconvert --to script "2026S1 COMP8221 Assignment 1 60957107 Alireza Yegane.ipynb"
python "2026S1 COMP8221 Assignment 1 60957107 Alireza Yegane.py"
```

### Checkpoints
If a trained checkpoint already exists, the notebook loads it and skips expensive retraining steps where appropriate.

---

## Reproducibility Notes

To make the project easier to reproduce:
- all core hyperparameters are centralised in a single configuration dictionary,
- the random seed is fixed,
- the notebook contains the complete implementation in sequence,
- and evaluation outputs are shown directly in the notebook.

This means the notebook functions as both:
- a **technical report**, and
- a **self-contained implementation record**.

---

## Limitations

This project is intentionally scoped as a course assignment, so a few limitations remain:
- CIFAR-10 is a low-resolution benchmark, so visual fidelity is naturally capped.
- The training budget is much smaller than large published diffusion baselines.
- Results are strong for a from-scratch student implementation, but not intended as a state-of-the-art claim.

I have therefore focused on:
- correct implementation,
- clear explanation,
- meaningful evaluation,
- and honest analysis of trade-offs.

---

## Submission Files

The main submission package consists of:
- `2026S1 COMP8221 Assignment 1 60957107 Alireza Yegane.ipynb`
- PDF export of the notebook
- this `README.md`
- repository source code and checkpoints as needed

---

## References

```bibtex
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

<p align="center">
  <strong>Macquarie University</strong><br>
  COMP8221 · Advanced Machine Learning · 2026 S1
</p>
