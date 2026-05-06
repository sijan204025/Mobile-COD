# Mobile-COD: Boundary-Guided Lightweight Framework for Real-Time Camouflaged Object Detection on Mobile Platforms

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19677006.svg)](https://doi.org/10.5281/zenodo.19677006)

## Citation

If you use this code, data, or models in your research, please cite:

**Md Mehady Hasan, Caikou Chen, Abdul Ghafar, "Mobile-COD: Boundary-Guided Lightweight Framework for Real-Time Camouflaged Object Detection on Mobile Platforms," *The Visual Computer*, 2026.**

---

## Overview

Mobile-COD is a lightweight framework for real-time camouflaged object detection on mobile devices. Our method focuses on boundary-guided feature extraction to accurately separate camouflaged objects from their backgrounds while maintaining real-time performance on resource-constrained mobile platforms.

---

## Dependencies & Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- torchvision ≥ 0.10.0
- OpenCV ≥ 4.5.0
- NumPy ≥ 1.19.0


### Installation

```bash
git clone https://github.com/sijan204025/Mobile-COD.git
cd Mobile-COD
pip install -r requirements.txt
```  

 ## Key Algorithm Description

### Boundary-Guided Module

Our Boundary-Guided Module enhances edge-aware feature learning by incorporating multi-scale boundary supervision. This module explicitly models fine-grained boundaries between camouflaged objects and their background, addressing the core challenge of camouflaged object detection.

### Lightweight Backbone

We adopt a lightweight MobileNet-based backbone optimized for mobile deployment. Depth-wise separable convolutions reduce computational cost and model size while enabling real-time inference on mobile platforms.

### Implementation Details

| Parameter | Value |
|-----------|-------|
| Input resolution | 256 × 256 |
| Backbone | MobileNet |
| Framework | PyTorch |

## Datasets

Three benchmark datasets downloaded from Kaggle are used to evaluate Mobile-COD:

| Dataset | Source |
|---------|--------|
| CAMO | Kaggle |
| COD10K | Kaggle |
| NC4K | Kaggle |

These datasets are publicly available for research use. All input images are resized to **256 × 256** before being fed into the network. 

