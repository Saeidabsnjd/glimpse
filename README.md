# glimpse — Real-Time Generative Neural Rendering for Mixed Reality

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![Blender](https://img.shields.io/badge/Blender-4.x%2F5.x-lightblue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
This repository contains the software and research pipeline for real-time generative neural rendering in Mixed Reality (MR) environments. High-fidelity rendering of complex optical phenomena—specifically Subsurface Scattering (SSS) in human skin and path-traced global illumination—is computationally intractable under strict MR display constraints (>60 FPS, <15 ms latency). This project implements a single-step generative neural model conditioned on fast draft renders and G-buffer primitives (camera-space surface normals, metric depth) to reconstruct path-traced ground truth in a single forward pass.

Research context: Immersive Experience Lab (IXLAB), TU Dresden, in collaboration with University of Padova.

## Structure
```
src/
  data/                              Blender synthetic dataset generator and G-buffer extraction
  models/                            Single-step generative neural rendering architectures
  losses/                            Photometric, perceptual (LPIPS), and physical material losses
  utils/                             Camera projection mathematics, EXR I/O, and metrics
```

## Run
```bash
# Clone the repository
git clone https://github.com/Saeidabsnjd/glimpse.git
cd glimpse

# Initialize environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

# Install core dependencies (CUDA-enabled PyTorch)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
