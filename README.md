# IXP Traffic Forecasting

**Course:** Network Measurement and Data Analysis Lab — Politecnico di Milano 2025/2026  
**Authors:** Luca Bordin · Mattia Menegale · Francesco Cavalieri

---

## Overview

This repository contains the work for the **IXP Traffic Forecasting** project, developed as part of the NDA Lab course. The project uses the [IXP Traffic Dataset](https://github.com/nsg-ethz/ixp-traffic-dataset) — a two-year collection (Jan 2023–Dec 2024) of 5-minute traffic statistics from 472 IXPs worldwide, covering 87% of all publicly announced IXP port capacity.

The project is split into two parts:

| Part | Task |
|------|------|
| **Main task** | Compare time-series forecasting models (ARIMA, GRU, LSTM, TSFM, Random Forest) on **European** IXPs |
| **Advanced task** | Simulate **Federated Learning** across Jakarta's IXPs |

---

## Main Task — Forecasting Model Comparison (Europe)

### Problem

> Can we accurately predict future IXP traffic volumes using historical time-series data?

Given the past **N** 5-minute intervals of inbound traffic at a European IXP, predict the next **K** intervals. N and K are treated as sensitivity analysis parameters.

### Models Compared

| Model | Type | Notes |
|-------|------|-------|
| **ARIMA** | Statistical | Autoregressive baseline; no learned representations |
| **Random Forest** | Ensemble ML | Tree-based regressor on sliding-window features |
| **GRU** | Deep Learning | Gated Recurrent Unit — fewer parameters than LSTM |
| **LSTM** | Deep Learning | Long Short-Term Memory — standard RNN for sequences |
| **TSFM** | Foundation Model | Zero-shot / fine-tuned (e.g., TimesFM, Chronos) |

### Dataset

- **Region:** Europe (372 IXPs, 34.9% of global total, 185 with collected data)
- **Granularity:** 5-minute intervals resampled to hourly for stable training
- **Features:** timestamp, IXP ID, inbound traffic volume (bps), port capacity

### Methodology

- Sliding window approach: each sample is a fixed-length lookback window → horizon prediction
- Train/validation/test split (temporal, no shuffling)
- Per-IXP standardization to handle the wide range of traffic scales across European IXPs
- Evaluation with scale-free metrics (NMAE%, R²) to allow fair comparison across IXPs of different sizes

### Notebooks

| Notebook | Description |
|----------|-------------|
| `Project_12_Finale (2).ipynb` | **Main project notebook** — full pipeline: data loading, all four models, sensitivity analysis, comparison |

---

## Advanced Task — Federated Learning on Jakarta's IXPs

### Problem

> Can a shared GRU model trained without any IXP sharing its raw data compete with a locally trained one?

Jakarta (Indonesia) hosts one of Asia's densest IXP clusters. The goal is to simulate **Federated Learning** across 15 Jakarta IXPs, keeping 3 as held-out test clients that never participate in training.

### Setup

```
Jakarta IXP profiles (13 usable after quality filtering, target was 15)
         │
         ├─ 10 training clients  ──► FedAvg + FedProx (15 rounds, μ=0.01)
         │
         └─ 3 held-out IXPs  ──► Evaluation only
```

**Architecture:** single-layer GRU (`hidden_size=64`), identical across all baselines so performance differences reflect *data exposure*, not model capacity.

**Task:** given the past **24 hours**, predict the next **6 hours** (hourly resolution).

**Baselines compared:**
- **Local** — independent GRU per IXP, trained only on that IXP's own data
- **Federated (FedAvg)** — one global GRU trained via FL, weights aggregated each round
- **Centralized** — GRU trained on pooled (standardized) data from all training clients; practical upper bound if data sharing were allowed

### Results

| Model | Mean NMAE% ↓ | Mean R² ↑ | Skill vs. Persistence ↑ |
|-------|-------------|-----------|------------------------|
| **Local** | **11.3%** | **0.02** | **-6.2** |
| Centralized | 13.4% | -0.99 | -10.4 |
| Federated | 13.6% | -6.0 | -16.9 |

**Key finding:** with 13 highly heterogeneous IXPs spanning four orders of magnitude in traffic volume (10⁻⁵–10³ Gbit/s), the local model outperforms both federated and centralized approaches. The IXP heterogeneity is the main bottleneck — averaging very different traffic profiles into a single shared model hurts more than it helps. FL trains cleanly (NMAE% drops from 37% to 13.5% over 15 rounds) but does not break even with the local baseline on this dataset.

### Notebooks

| Notebook | Description |
|----------|-------------|
| `Federated_Learning (1).ipynb` | Main FL pipeline — data loading, FedAvg training, baselines, evaluation |
| `Federated_Learning_Preprocessed_LOOCV.ipynb` | Leave-One-Out Cross-Validation variant |
| `Federated_Learning_Processed.ipynb` | Post-processing and extended analysis |

---

## How to Run

All notebooks are designed for **Google Colab** (GPU recommended).

1. Clone the IXP dataset onto Google Drive:
   ```bash
   git clone --depth 1 https://github.com/nsg-ethz/ixp-traffic-dataset.git
   ```
2. Open a notebook in Colab, mount Drive, and set `PROJECT_DIR` to your Drive path (first cell).
3. Run all cells — the dataset is auto-downloaded and cached if not already present.

**Dependencies:** `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `requests`, `pyarrow`

---

## Repository Structure

```
PROJECT/
├── Project_12_Finale (2).ipynb               # Main task — model comparison (Europe)
├── Federated_Learning (1).ipynb              # Advanced task — FL main pipeline
├── Federated_Learning_Preprocessed_LOOCV.ipynb
├── Federated_Learning_Processed.ipynb
└── Presentazione NDA.pptx                    # Slide deck
```

---

## Tech Stack

- **Deep Learning:** PyTorch (GRU, LSTM, FedAvg, FedProx)
- **Classical ML / Stats:** scikit-learn (Random Forest), statsmodels (ARIMA)
- **Foundation Models:** TimesFM / Chronos (zero-shot inference)
- **Data:** pandas, numpy, pyarrow (Parquet)
- **Visualization:** matplotlib, seaborn
- **External APIs:** PeeringDB REST API (IXP metadata)
- **Platform:** Google Colab (T4/A100 GPU)

---

## Authors

- [Luca Bordin](mailto:luca1.bordin@mail.polimi.it)
- Mattia Menegale
- Francesco Cavalieri

**Institution:** Politecnico di Milano — MSc Computer Science & Engineering
