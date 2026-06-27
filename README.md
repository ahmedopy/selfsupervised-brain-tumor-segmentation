# 🧠 MIM-HeMIS: Self-Supervised Brain Tumor Segmentation under Missing MRI Modalities

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Imaging-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)
![Dataset](https://img.shields.io/badge/Dataset-BraTS%202021-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

# 🧩 Overview

Brain tumor segmentation from multi-modal MRI is a fundamental task in neuro-oncology, supporting diagnosis, treatment planning, and disease monitoring. Although recent deep learning methods have demonstrated remarkable performance, they generally assume that all MRI modalities are available during inference. In real-world clinical settings, however, MRI sequences are frequently missing due to acquisition artifacts, patient conditions, scanning costs, or varying clinical protocols.

This repository presents **MIM-HeMIS**, a novel self-supervised framework designed for **robust brain tumor segmentation under arbitrary missing MRI modalities**.

The proposed framework integrates:

- **Masked Image Modeling (MIM)** for self-supervised representation learning.
- **HeMIS (Hetero-Modal Image Segmentation)** for modality-invariant feature fusion.
- A **hybrid CNN-Transformer architecture** for capturing both local and global contextual information.
- A flexible fusion strategy capable of handling **any combination of missing MRI modalities without retraining**.

Unlike conventional methods, the proposed model remains operational even when one or multiple MRI modalities are unavailable, making it highly suitable for realistic clinical deployment.

---

# ✨ Key Contributions

✅ Self-supervised pretraining using **Masked Image Modeling (MIM)** on multi-modal MRI volumes.

✅ A novel integration of **HeMIS and SSL representations** for robust heterogeneous modality fusion.

✅ Handles **arbitrary missing MRI modalities** during inference.

✅ No modality-specific retraining required.

✅ Robust segmentation performance across **all 15 possible modality combinations**.

✅ Extensive evaluation on the **BraTS 2021 benchmark dataset**.

---

# 🏗️ Framework Architecture

The proposed pipeline consists of two major stages:

```text
Multi-modal MRI Volumes
(T1, T1ce, T2, FLAIR)
            │
            ▼
──────────────────────────────
Self-Supervised Pretraining
(Masked Image Modeling)
──────────────────────────────
            │
            ▼
Transformer-based SSL Encoder
            │
            ▼
──────────────────────────────
Modality-Specific CNN Encoders
──────────────────────────────
            │
            ▼
HeMIS Abstraction Layer
(Mean + Variance Fusion)
            │
            ▼
Fusion of SSL + HeMIS Features
            │
            ▼
Segmentation Decoder
            │
            ▼
Brain Tumor Segmentation
(NCR, ED, ET)
```

---

# 🧠 Model Components

| Component | Description |
|-----------|-------------|
| Self-Supervised Learning | Masked Image Modeling (MIM) |
| SSL Backbone | Transformer Encoder |
| Modality Encoder | 3D CNN Encoder |
| Fusion Strategy | HeMIS Abstraction (Mean + Variance) |
| Decoder | U-Net-style Segmentation Decoder |
| Loss Function | Weighted Cross-Entropy + Dice Loss |
| Input Modalities | T1, T1ce, T2, FLAIR |
| Output Classes | Background, NCR, ED, ET |

---

# 📂 Repository Structure

```text
📦 selfsup-brain-tumor-segmentation
│
├── README.md
├── requirements.txt
│
├── configs/                  # Training and experiment configurations
│
├── models/
│   └── hemis_ssl_model.py    # MIM-HeMIS architecture
│
├── datasets/
│   └── brats_dataset.py      # BraTS dataset loader and preprocessing
│
├── training/
│   ├── pretrain.py           # Self-supervised MIM pretraining
│   └── finetune.py           # Segmentation fine-tuning
│
├── evaluation/
│   ├── evaluate.py           # Quantitative evaluation
│   ├── uncertainty.py        # Uncertainty analysis
│   └── visualize.py          # Qualitative visualization scripts
│
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   ├── schedulers.py
│   └── helpers.py
│
├── checkpoints/              # Saved model weights
├── results/                  # Experimental outputs and plots
├── figures/                  # Paper figures and visualizations
└── notebooks/                # Exploratory analysis notebooks
```

---

# 📊 Dataset

This work uses the **Brain Tumor Segmentation (BraTS) 2021** dataset.

> **Note:** The BraTS 2021 dataset is **not included** in this repository due to licensing restrictions.

Please obtain the dataset from the official BraTS challenge website and organize it according to the expected directory structure.

### Supported MRI Modalities

- FLAIR
- T1
- T1ce
- T2

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your_username/selfsup-brain-tumor-segmentation.git

cd selfsup-brain-tumor-segmentation
```

Create a virtual environment:

```bash
conda create -n mim-hemis python=3.10
conda activate mim-hemis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## 1. Self-Supervised Pretraining

Pretrain the Transformer encoder using Masked Image Modeling:

```bash
python training/pretrain.py \
    --data_dir /path/to/BraTS2021 \
    --epochs 300 \
    --mask_ratio 0.75
```

---

## 2. Segmentation Fine-Tuning

Fine-tune the pretrained model for tumor segmentation:

```bash
python training/finetune.py \
    --data_dir /path/to/BraTS2021 \
    --pretrained checkpoints/ssl_pretrained.pth
```

---

## 3. Model Evaluation

Evaluate the model on validation or test data:

```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/best_model.pth
```

---

# 📈 Evaluation Protocol

The framework is evaluated under:

- Complete MRI modality setting.
- Single missing modality scenarios.
- Multiple missing modality scenarios.
- All **15 possible modality combinations**.

Performance is assessed using:

| Metric |
|---------|
| Dice Similarity Coefficient (DSC) |
| Tumor Core Dice (TC) |
| Whole Tumor Dice (WT) |
| Enhancing Tumor Dice (ET) |

---

# 🔬 Experimental Findings

- Self-supervised pretraining significantly improves segmentation performance.
- HeMIS-based fusion provides strong robustness against missing modalities.
- The proposed framework consistently outperforms conventional baselines under incomplete MRI settings.
- Performance degradation remains minimal even when multiple modalities are absent.

---

# 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{yourcitation2026,
  title={MIM-HeMIS: Self-Supervised Masked Image Modeling with Heterogeneous Modality Fusion for Brain Tumor Segmentation under Missing MRI Modalities},
  author={Your Name},
  year={2026}
}
```

---
# ⭐ Acknowledgements

- BraTS 2021 Challenge
- PyTorch
- Medical Open Network for AI (MONAI)
- The Brain Tumor Segmentation research community
