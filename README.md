# MFCFA: An AI-Based Multi-task Framework for Cardiac Function Assessment

[![Paper](https://img.shields.io/badge/Paper-SPIE%20ICGIP%202023-blue)](https://doi.org/10.1117/12.3005204)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper:

> **An AI-Based Multi-task Framework for Cardiac Function Assessment through Echocardiograms**
> Aoyang Guo\*, Haimiao Mo\*, Zhijian Hu, Hongjia Wu, Juan Liang
> *SPIE ICGIP 2023* (\* equal contribution)

---

## Abstract

Accurate assessment of cardiac function is critical for diagnosing and managing cardiovascular diseases. Existing deep learning methods for ejection fraction (EF) prediction from echocardiograms largely overlook the intrinsic physiological relationship between EF, end-diastolic volume (EDV), and end-systolic volume (ESV), defined as:

$$EF = \frac{EDV - ESV}{EDV} \times 100\%$$

We propose **MFCFA**, a multi-task learning framework that jointly predicts EF, ESV, and EDV from echocardiogram videos using a shared spatiotemporal feature extractor (R2Plus1D-18). By incorporating the interdependencies among these three tasks into a weighted loss function, MFCFA achieves state-of-the-art EF prediction on the EchoNet-Dynamic benchmark.

---

## Framework

![MFCFA Framework](figures/framework.png)

The framework consists of three components:

1. **Feature Extraction**: A pretrained R2Plus1D-18 3DCNN extracts spatiotemporal features from echocardiogram video clips. The network decomposes 3D convolutions into 2D spatial and 1D temporal convolutions across five convolutional blocks, followed by spatiotemporal pooling.

2. **Multi-task Learning**: Three task-specific fully connected heads independently predict EF, ESV, and EDV from the shared feature representation.

3. **Weighted Loss**: A weighted combination of MSE losses for each subtask:

$$\mathcal{L} = w_1 \cdot \mathcal{L}_{EF} + w_2 \cdot \mathcal{L}_{ESV} + w_3 \cdot \mathcal{L}_{EDV}$$

where $w_1 = 0.9$, $w_2 = 0.09$, $w_3 = 0.01$.

---

## Results

### Comparison with State-of-the-Art (EchoNet-Dynamic Test Set)

| Method | RMSE ↓ | MAE ↓ | R² ↑ |
|--------|--------|-------|------|
| **MFCFA (Ours)** | **5.13** | **3.89** | **0.82** |
| EchoNet-Dynamic | 5.32 | 4.05 | 0.81 |
| EchoCoTr | 5.17 | 3.95 | 0.82 |
| LWDVN | 5.30 | 4.10 | 0.81 |
| R3D | 5.62 | 4.22 | 0.79 |
| MC3 | 5.97 | 4.54 | 0.77 |
| UVT | 8.38 | 5.95 | 0.52 |

### Ablation Study

| Method | T_EF | T_ESV | T_EDV | RMSE ↓ | MAE ↓ | R² ↑ |
|--------|------|-------|-------|--------|-------|------|
| **MFCFA (Ours)** | ✓ | ✓ | ✓ | **5.13** | **3.89** | **0.82** |
| M1 (single-task) | ✓ | — | — | 5.56 | 4.22 | 0.79 |
| M2 (EF + ESV) | ✓ | ✓ | — | 5.23 | 3.91 | 0.82 |
| M3 (EF + EDV) | ✓ | — | ✓ | 5.38 | 4.08 | 0.81 |

The ablation results confirm that jointly training all three tasks yields the best performance, and that ESV is a more effective auxiliary task than EDV when only one auxiliary task is used.

---

## Dataset

We use the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset (Ouyang et al., NeurIPS ML4H 2019), which contains 10,036 apical four-chamber echocardiogram videos collected at Stanford Hospital with corresponding EF, ESV, and EDV labels.

| Split | Videos |
|-------|--------|
| Train | 7,465 |
| Val | 1,289 |
| Test | 1,282 |

After downloading, configure the data path by creating `echonet.cfg` in the project root:

```
DATA_DIR = /path/to/EchoNet-Dynamic/
```

---

## Installation

```bash
git clone https://github.com/kwokxxx/mfcfa.git
cd mfcfa
pip install -r requirements.txt
```

**Requirements:** Python 3.7+, PyTorch 1.8.0, CUDA-capable GPU (trained on RTX 3090)

---

## Usage

### Training

```bash
python -m echonet video
```

With custom options:

```bash
python -m echonet video \
  --data_dir /path/to/EchoNet-Dynamic \
  --output output/mfcfa \
  --num_epochs 45 \
  --batch_size 15 \
  --model_name r2plus1d_18 \
  --pretrained
```

### Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | from `echonet.cfg` | Path to EchoNet-Dynamic dataset |
| `--output` | `output/video/` | Directory for logs and checkpoints |
| `--model_name` | `r2plus1d_18` | Backbone architecture |
| `--pretrained` | `True` | Initialize from Kinetics-400 weights |
| `--num_epochs` | `45` | Training epochs |
| `--batch_size` | `15` | Batch size |
| `--lr` | `1e-4` | Initial learning rate (decayed ×0.1 every 15 epochs) |
| `--frames` | `32` | Frames per clip |
| `--period` | `2` | Frame sampling period |

### Output

Training produces the following in the output directory:

- `log.csv` — per-epoch train/val loss and R²
- `checkpoint.pt` — latest checkpoint
- `best.pt` — best validation checkpoint
- `test_predictions.csv` — per-clip EF predictions on test set
- `test_scatter.pdf` — scatter plot of predicted vs. actual EF

---

## Project Structure

```
├── echonet/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py             # Data path configuration
│   ├── datasets/
│   │   └── echo.py           # EchoNet-Dynamic dataset loader (multi-task targets)
│   └── utils/
│       ├── __init__.py
│       └── video.py          # Training loop, multi-task loss, evaluation
├── figures/
│   └── framework.png         # Model architecture diagram
├── example.cfg               # Example data path configuration
├── requirements.txt
└── LICENSE
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{guo2024mfcfa,
  title     = {An AI-based multitask framework for cardiac function assessment through echocardiograms},
  author    = {Guo, Aoyang and Mo, Haimiao and Hu, Zhijian and Wu, Hongjia and Liang, Juan},
  booktitle = {Fifteenth International Conference on Graphics and Image Processing (ICGIP 2023)},
  volume    = {13089},
  pages     = {442--450},
  year      = {2024},
  month     = {March},
  publisher = {SPIE}
}
```

---

## Acknowledgements

This project builds on [EchoNet-Dynamic](https://github.com/echonet/dynamic) (Ouyang et al., *Nature* 2020). The dataset loader and training pipeline are adapted from their open-source codebase.
