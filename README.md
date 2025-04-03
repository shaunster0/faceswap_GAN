# FaceSwap GAN

This project implements a simplified version of the GHOST face-swapping model using PyTorch. It leverages an identity encoder (ArcFace), an attribute encoder (U-Net-based), and a generator with Adaptive Attention Denormalization (AAD) blocks to produce high-fidelity swapped face images.

<p align="center">
  <img src="assets/demo_swap.png" alt="Example Face Swap" width="600"/>
</p>

## 🚀 Features

- U-Net-based attribute encoder with optional skip connections
- Pretrained ArcFace identity encoder
- AAD-based generator (inspired by [GHOST](https://github.com/zyainfal/ghost))
- PatchGAN discriminator
- Supports LFW dataset preprocessing with face alignment
- Modular training pipeline with Weights & Biases integration
- Identity & reconstruction losses to preserve realism

## 📦 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/faceswap-gan.git
   cd faceswap-gan
