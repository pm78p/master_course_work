# Dual-Decoder UNet-Transformer Autoencoder for Blind Source Separation

[![Python](https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00.svg?style=flat&logo=Google-Colab&logoColor=white)](https://colab.research.google.com/)

A TensorFlow/Keras implementation that takes as input a grayscale image formed by averaging an MNIST digit and a Fashion-MNIST item, and separates them back into their original components using a hybrid UNet + Transformer + CBAM architecture with two parallel decoder heads.

---

## 📖 Overview

This project tackles **Blind Source Separation (BSS)** in images:  
- **Input**: single-channel (32×32) image, the pixel-wise average of one MNIST digit and one Fashion-MNIST sample  
- **Output**: two reconstructed images—one of the MNIST digit and one of the Fashion-MNIST item  

Key innovations:
1. **UNet-style encoder** with convolutional downsampling and skip-connections
2. **Transformer bottleneck** (multi-head self-attention) for long-range dependencies
3. **Dual decoders**:
   - Each branch uses upsampling + convolution  
   - Embedded lightweight transformer blocks  
   - Convolutional Block Attention Module (CBAM) for channel & spatial refinement  
   - UNet-style autoencoder sub-blocks to further polish reconstructions  

---

## ⚙️ Model Architecture

![My Project Screenshot](architecture.jpg)

```mermaid
graph TD
  A[Input (32×32×1)] -->|Conv + MaxPool ×3| B[Encoder Feature Map]
  B -->|Reshape → Transformer Blocks| C[Contextual Features]
  C -->|Reshape → Decoder Branch 1| D1[Upsample + CBAM + Autoencoder Sub-block]
  C -->|Reshape → Decoder Branch 2| D2[Upsample + CBAM + Autoencoder Sub-block]
  D1 -->|Final Conv| Output1[Reconstructed MNIST]
  D2 -->|Final Conv| Output2[Reconstructed Fashion-MNIST]
