# Mind the Gap: Bridging Domain Shift in Multi-Modal Beam Prediction

## Overview

Deep learning can predict the best mmWave beam from sensor data and avoid exhaustive beam sweeping. However, a model trained in one deployment scenario often loses accuracy in a new one. This repository adapts a multi-modal beam predictor to an unseen scenario **without target labels**, using two unsupervised domain adaptation (UDA) methods:

- **CUDA** (Concept-based Unsupervised Domain Adaptation): aligns the domains adversarially through a concept bottleneck whose concepts are angular sectors of the beam codebook.
- **DSAN** (Deep Subdomain Adaptation Network): aligns feature distributions within beam sectors using the local maximum mean discrepancy (LMMD).

On the target scenario, CUDA raises Top-1 accuracy from **17.73% to 38.0%** and Top-5 accuracy from **55.67% to 83.9%** compared with the source-only model.

## Method

### Multi-modal backbone

| Component | Details |
|---|---|
| RGB branch | EfficientNet-B5 (ImageNet) → CBAM → global average pooling → 2-layer BiGRU (128 per direction) |
| GPS branch | 8 kinematic features per step (relative position, distance, 2D velocity, speed, sin/cos heading) → 1-layer BiGRU (128 per direction) |
| Fusion | Learned sigmoid gates, projection to 128-d, Transformer encoder (2 layers, 4 heads, FFN 512, dropout 0.3) |
| Head | 128 → 64 → 64 beams, with LayerNorm, GELU and dropout |
| Input window | T = 5 time steps, images resized to 224×224 |

### Soft Beam Loss (SBL)

Beam powers are standardized per sample and turned into a soft target with a temperature-scaled softmax (τ_s = 0.5). The loss combines soft and hard cross-entropy (α = 0.3), so beams with power close to the best beam are not treated as completely wrong.

### Domain adaptation

- **CUDA**: three stages. (I) source training, (II) concept bottleneck training with the backbone frozen, (III) adversarial alignment on concept representations, with the alignment term clipped at τ.
- **DSAN**: LMMD over K = 16 beam sectors, soft sector assignments for target samples, feature queues of size 64, and a sigmoid schedule for the alignment weight.

Target samples are used without beam labels or beam powers.

## Dataset

We use [DeepSense 6G](https://www.deepsense6g.net/), Scenarios 1–4 (Testbed 1, 60 GHz, 64-beam codebook):

| Scenario | Location | Lighting | Role |
|---|---|---|---|
| 1 | Street 1 | Day | Source |
| 2 | Street 1 | Night | Source |
| 3 | Street 2 | Day | Source |
| 4 | Street 2 | Night | Target (80% adaptation / 20% test) |

Download the scenarios from the DeepSense 6G website and place them in a local data folder.

## Results

All target results are on the Scenario 4 test split.

### Source-only vs. adaptation

| Method | Top-1 (%) | Top-3 (%) | Top-5 (%) |
|---|---|---|---|
| Source-only (EfficientNet-B5) | 17.73 | 39.82 | 55.67 |
| DSAN | 37.29 | 64.56 | 71.48 |
| **CUDA** (τ = 0.6, λ_d = 0.1, K = 8) | **38.0** | **66.5** | **83.9** |

### Backbone comparison (no adaptation)

| Backbone | In-domain Top-1 | In-domain Top-5 | Target Top-1 | Target Top-5 |
|---|---|---|---|---|
| ResNet-50 | **47.20** | 88.20 | 13.70 | 34.61 |
| EfficientNet-B3 | 41.30 | 86.30 | 9.99 | 28.64 |
| EfficientNet-B5 | 42.40 | **89.50** | **17.73** | **55.67** |
| Swin-T | 37.10 | 85.40 | 3.60 | 10.10 |
| ConvNeXt-Small | 38.40 | 83.50 | 12.36 | 37.50 |

### Loss comparison (CUDA, τ = 0.6, λ_d = 0.3, K = 16)

| Loss | Top-1 (%) | Top-3 (%) | Top-5 (%) |
|---|---|---|---|
| Cross-entropy | 34.4 | 66.8 | 81.3 |
| Focal | 32.2 | **68.4** | 82.0 |
| **Soft Beam Loss** | **35.8** | 66.8 | **83.5** |

> **Note on model selection:** no labeled target validation set is available, so hyperparameters were selected on the target test split. The selected results should be read as an upper bound.

## Training setup

- Optimizer: Adam, learning rate 3e-5, batch size 32
- Source training: 40 epochs; adaptation: 25 epochs
- Augmentation (training only): color and geometric transforms, random image dropout, and a night-simulation transform calibrated on unlabeled Scenario 4 images
- Hardware: single GPU (Google Colab)

## Usage

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

<!-- Add the exact commands for source training, CUDA adaptation and DSAN adaptation here. -->


## Acknowledgments

This work uses the [DeepSense 6G](https://www.deepsense6g.net/) dataset. CUDA follows Xu et al., *Concept-Based Unsupervised Domain Adaptation* (arXiv:2505.05195), and DSAN follows Zhu et al., *Deep Subdomain Adaptation Network for Image Classification* (IEEE TNNLS, 2021).

## Contact

- Salma Elmenshawy: S.Elmensh01816@aast.edu
- Noureldin Elmadany: Nourelmadany@aast.edu
