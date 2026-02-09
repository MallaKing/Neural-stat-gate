# Neural Statistical Gating

**A lightweight, non-linear audio denoiser that uses Higher-Order Statistics (Skewness & Kurtosis) to differentiate signal from noise.**

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)

---

##  The Idea

Most simple denoisers use **Variance (Energy)** to decide what to keep.
* **The Problem:** If the noise gets loud (high variance), the filter thinks it's a signal and lets it through.
* **The Solution:** This model looks at the **Shape** of the distribution, not just the volume.

By feeding the **Kurtosis** (tail heaviness) and **Skewness** (asymmetry) into a tiny MLP, the model learns that **Gaussian Noise is still Noise**, even when it's loud. It effectively learns a "Statistical Gate" that clamps down on unstructured data while letting structured signals (speech, sine waves) pass.

---

##  How It Works

The architecture is a lightweight **Statistical Re-projection**. It doesn't generate audio from scratch; it calculates statistics, predicts a correction, and re-projects the input.

### 1. Feature Extraction
For every small window of audio (e.g., 50 samples), we calculate differentiable moments:
* **$\mu$ (Mean):** DC Offset
* **$\sigma$ (Standard Deviation):** Energy
* **Skewness:** Asymmetry (detects signal onset)
* **Kurtosis:** Tail heaviness (distinguishes Gaussian noise from structured signal)

### 2. The Gate (MLP)
A tiny neural network (3 layers) maps these 4 stats to two affine parameters: **$\gamma$ (Scale)** and **$\beta$ (Shift)**.

### 3. Re-projection
$$\hat{x} = \gamma(\mathbf{s}) \cdot \left( \frac{x - \mu}{\sigma} \right) + \beta(\mathbf{s})$$

If the model detects noise (Low Kurtosis), it pushes $\gamma \to 0$. If it detects signal (High Kurtosis), it restores the amplitude.

---

##  Features

* **Tiny Footprint:** Only ~2,500 parameters (vs 5M+ for U-Net).
* **Robust:** Handles non-stationary noise (bursts) better than standard spectral subtraction.
* **Interpretable:** We aren't learning a black box; we are learning a mapping from *Statistics* $\to$ *Gain*.
* **PyTorch Native:** Fully differentiable 4th-order moment calculation.

---
