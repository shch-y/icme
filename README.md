# ICME 2026 Low-Bit-width Large-Model Quantization Challenge

This repository contains our submission materials for the **ICME 2026 Low-Bit-width Large-Model Quantization Challenge**.

We participated in **Sub-Challenge 1: W4A4 Quantization for Inference (HiF4 / MXFP4)**, which targets post-training 4-bit weight and activation quantization for inference on **Wan-AI/Wan2.2-I2V-A14B**.

See [[Install.md]] for how to reproduce the result.
See [[Result.md]] for our final vbench metric results.


## Track

- Challenge: ICME 2026 Low-Bit-width Large-Model Quantization Challenge
- Sub-Challenge: W4A4 Quantization for Inference
- Target model: Wan-AI/Wan2.2-I2V-A14B
- Quantization format: HiF4 / MXFP4-compatible W4A4 inference
- Evaluation: OpenS2V-5M prompts with VBench metrics

## Overview

Our work focuses on practical post-training quantization strategies for video generation inference. During the competition, we mainly explored three complementary techniques:

1. **MixQ-style mixed precision quantization**
2. **SmoothQuant-style activation smoothing**
3. **Block-wise compression / block quantization**

The goal is to reduce the quantization error of linear layers under W4A4 constraints while preserving video quality measured by VBench.

## Method Summary

### 1. MixQ / Mixed Precision Outlier Handling

We observed that a small subset of activation channels contributes disproportionately to quantization error, especially in FFN layers.

Our mixed precision strategy keeps a small number of outlier columns in higher precision and quantizes the remaining columns using HiF4-style W4A4 quantization.

The outlier columns are selected by activation statistics, primarily using:

```text
score_j = max(|x[:, j]|)
```

Columns with the largest scores are routed through a high-precision branch, while the rest are handled by the quantized branch.

### 2. SmoothQuant-style Smoothing

We also implemented SmoothQuant-style per-channel smoothing to reduce activation outliers before quantization.

For each linear layer, activation and weight statistics are used to compute a smoothing scale:

```text
s = act_scale^alpha / weight_scale^(1 - alpha)
```

The scale is folded into the weight, and the corresponding input activation is divided by the same scale during inference. This reduces activation dynamic range while preserving the original linear transformation.

### 3. Block Compression

We explored block-wise quantization and compression strategies to improve numerical stability and reduce the effect of local outliers.

Several block layouts were tested during development, including:

- 32x32 block quantization
- 128x1 block quantization along the K dimension
- Per-block scaling

These experiments helped us understand the trade-off between quantization granularity, memory overhead, and inference stability.

## Evaluation

The generated videos are evaluated with VBench metrics, following the challenge requirements. We primarily focused on metrics relevant to image-to-video generation quality, such as:

- Subject consistency
- Overall consistency
- Motion smoothness
- Imaging quality
- Aesthetic quality

We find VBench-I2V can not evaluate the metric **Overall consistency**, which is a metric for VBench-T2V.

We report our results in [[Result.md]]


## Acknowledgements

This work builds on the following open-source resources:

- Wan-AI/Wan2.2
- VBench
- HiFloat4 simulation toolkit
- SmoothQuant
- OpenS2V-5M benchmark resources

## Contributors
Yidong Chen
Chengyu Shi
Jiahao Liu