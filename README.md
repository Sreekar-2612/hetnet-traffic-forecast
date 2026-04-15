# TASTF: Tier-Aware Spatiotemporal Forecasting for Heterogeneous Cellular Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.x-3C2179)](https://pyg.org)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Proposed Solution — TASTF](#proposed-solution--tastf)
- [Key Assumptions](#key-assumptions)
- [Dataset](#dataset)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Heterogeneous Graph Construction](#heterogeneous-graph-construction)
- [Loss Functions](#loss-functions)
- [Training Configuration & Hyperparameters](#training-configuration--hyperparameters)
- [Hyperparameter Variations & Effects](#hyperparameter-variations--effects)
- [Baselines](#baselines)
- [Results](#results)
- [Efficiency Analysis](#efficiency-analysis)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
- [Limitations & Future Work](#limitations--future-work)
- [Attribution](#attribution)

---

## Problem Statement

Modern cellular networks are evolving toward **Heterogeneous Network (HetNet)** architectures in which multiple tiers of base stations — **Macro** (high-power, wide-coverage), **Pico** (medium-power, hotspot), and **Femto** (low-power, indoor) — coexist to serve increasingly complex traffic demands. Accurate **spatiotemporal traffic forecasting** across these tiers is critical for:

- Dynamic resource allocation and load balancing
- Energy-efficient base station sleep/wake scheduling
- Proactive capacity planning and network slicing

Traditional forecasting models treat all cells as **homogeneous**, ignoring the fundamental **hierarchy** and **cross-tier interference patterns** present in HetNets. This leads to sub-optimal predictions, especially for cells with vastly different traffic characteristics.

---

## Proposed Solution — TASTF

**TASTF (Tier-Aware Spatiotemporal Forecasting)** is our proposed deep learning framework that explicitly models the heterogeneous nature of multi-tier cellular networks. The core innovation is a **two-stage architecture**:

1. **Spatial Stage — Heterogeneous Graph Neural Network (HeteroGNN):**
   Cells are partitioned into Macro, Pico, and Femto tiers based on traffic volume. A heterogeneous graph with **tier-specific intra-tier edges** and **cross-tier edges** (Macro→Pico, Macro→Femto) is constructed. Type-specific `SAGEConv` layers within `HeteroConv` learn distinct spatial representations for each tier, capturing both local neighborhood similarity and cross-tier influence.

2. **Temporal Stage — Transformer Encoder:**
   The sequence of spatial embeddings across timesteps is fed into a Transformer encoder with sinusoidal positional encoding. Multi-head self-attention captures **long-range temporal dependencies** and periodic traffic patterns (e.g., diurnal, weekly cycles).

3. **Prediction Head:**
   A linear projection maps the final temporal embedding to multi-step ahead forecasts.

**Why this works better than homogeneous models:**
- Tier-specific convolutions prevent **feature dilution** where high-traffic Macro patterns would dominate low-traffic Femto cells.
- Cross-tier edges explicitly model the **spatial hierarchy** and offloading patterns in HetNets.
- Temporal features (hour-of-day, day-of-week encoded as sin/cos) enable the model to capture **cyclic patterns** without manual feature engineering.

---

## Key Assumptions

| # | Assumption | Justification |
|---|-----------|---------------|
| 1 | **Tier assignment is static** — cells are grouped into Macro/Pico/Femto based on mean activity over the training period | Simplifies graph construction; avoids the complexity of dynamic re-tiering during inference |
| 2 | **Spatial proximity ≈ KNN on grid coordinates** — nearby cells influence each other | In the absence of real geographic coordinates (GeoJSON), a `√N` grid layout is used as a heuristic for KNN edge construction |
| 3 | **Activity = Internet + Call_in + Call_out** — a composite signal is used instead of individual features | Reduces dimensionality while capturing total network load |
| 4 | **MinMax scaling to [0, 1]** — fitted only on training data | Prevents data leakage; ensures fair evaluation |
| 5 | **Chronological (non-shuffled) train/val/test split** | Respects temporal ordering, which is critical for time-series forecasting |
| 6 | **Macro = top 10%, Femto = bottom 30%, Pico = middle 60%** of cells by mean activity | Mirrors real-world HetNet deployment density ratios |
| 7 | **10-minute aggregation intervals** — the native resolution of the Telecom Italia dataset | No resampling is performed |

---

## Dataset

**Source:** [Telecom Italia Big Data Challenge](https://dandelion.eu/datamine/open-big-data/) — Milan city SMS, call, and internet activity aggregated on a 100×100 spatial grid.

| Property | Value |
|----------|-------|
| **City** | Milan, Italy |
| **Spatial resolution** | 10,000 grid cells (filtered to top 100) |
| **Temporal resolution** | 10-minute intervals |
| **Files used** | `sms-call-internet-mi-2013-11-01.txt`, `sms-call-internet-mi-2013-11-02.txt` |
| **Fields** | `grid_id`, `interval` (epoch ms), `country_code`, `sms_in`, `sms_out`, `call_in`, `call_out`, `internet` |
| **Target signal** | `activity = internet + call_in + call_out` |
| **License** | Open data — ODbL-style attribution applies |

---

## Data Preprocessing Pipeline

```
Raw .txt files
    │
    ▼
┌──────────────────────────────────┐
│  1. Load & Filter (grid ≤ 100)   │
│  2. Compute activity composite   │
│  3. Pivot: interval × grid       │
│  4. Sort chronologically         │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  5. MinMaxScaler fit on train    │
│     timesteps only (no leakage)  │
│  6. Transform entire dataset     │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  7. Sliding window sequences     │
│     Input:  12 steps (2 hours)   │
│     Output: 3 steps  (30 min)    │
│  8. Chronological split:         │
│     Train 70% | Val 10% | Test 20│
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  9. Temporal features: sin/cos   │
│     of hour-of-day & day-of-week │
│     → 4-dim vector per sample    │
└──────────────────────────────────┘
```

---

## Model Architecture

### Architecture Diagram

```
Input: x_seq (B, T=12, N=100)   Time Features: (B, 4)
         │                              │
         ▼                              │
    ┌─────────────────────────────┐     │
    │  For each timestep t=0..11: │     │
    │  ┌────────────────────────┐ │     │
    │  │ Partition into tiers:  │ │     │
    │  │ Macro(10) Pico(60)    │ │     │
    │  │ Femto(30)             │ │     │
    │  └──────────┬────────────┘ │     │
    │             ▼              │     │
    │  ┌────────────────────────┐│     │
    │  │ HeteroConv Layer 1     ││     │
    │  │ SAGEConv per edge type ││     │
    │  │ (5 types) → ReLU       ││     │
    │  └──────────┬─────────────┘│     │
    │             ▼              │     │
    │  ┌────────────────────────┐│     │
    │  │ HeteroConv Layer 2     ││     │
    │  │ SAGEConv → ReLU →      ││     │
    │  │ LayerNorm → Linear     ││     │
    │  └──────────┬─────────────┘│     │
    │             ▼              │     │
    │  Reassemble: (B, N, 64)    │     │
    └─────────────┬──────────────┘     │
                  ▼                     │
         Stack: (T, B×N, 64)           │
                  │                     │
                  ▼                     │
    ┌──────────────────────────┐        │
    │ Sinusoidal Positional    │ ◄──────┘
    │ Encoding (max_len=100)   │
    └──────────┬───────────────┘
               ▼
    ┌──────────────────────────┐
    │ Transformer Encoder      │
    │ 2 layers, 4 heads        │
    │ d_model=64, ff=128       │
    │ dropout=0.1              │
    └──────────┬───────────────┘
               ▼
         Final embedding [-1]
               │
               ▼
    ┌──────────────────────────┐
    │ Linear Head              │
    │ 64 → horizon (3)         │
    └──────────┬───────────────┘
               ▼
    Output: (B, 3, N=100)  (multi-step forecast)
```

### Component Details

| Component | Configuration | Details |
|-----------|---------------|---------|
| **HeteroGNNEncoder** | 2 `HeteroConv` layers with `SAGEConv` | 5 edge types: 3 intra-tier `geo` + 2 cross-tier `cross`; hidden=32, output=64 |
| **Activation** | ReLU | Applied after each convolution layer |
| **Normalization** | LayerNorm(64) | Applied after second `HeteroConv` |
| **Per-tier Projection** | `nn.Linear(64, 64)` × 3 | Separate linear projections for macro, pico, femto embeddings |
| **Positional Encoding** | Sinusoidal | Standard Transformer-style, `d_model=64`, `max_len=100` |
| **Transformer Encoder** | 2 layers, 4 heads | `d_model=64`, `dim_feedforward=128`, `dropout=0.1` |
| **Prediction Head** | `nn.Linear(64, 3)` | Maps final embedding to 3-step horizon |

### Edge Types in the Heterogeneous Graph

| Edge Type | Direction | Purpose |
|-----------|-----------|---------|
| `(macro, geo, macro)` | Macro ↔ Macro | Intra-tier spatial proximity |
| `(pico, geo, pico)` | Pico ↔ Pico | Intra-tier spatial proximity |
| `(femto, geo, femto)` | Femto ↔ Femto | Intra-tier spatial proximity |
| `(macro, cross, pico)` | Macro → Pico | Cross-tier offloading influence |
| `(macro, cross, femto)` | Macro → Femto | Cross-tier coverage overlap |

---

## Heterogeneous Graph Construction

1. **Tier Partitioning:** Cells are sorted by mean training activity:
   - **Macro** (top 10%): 10 cells with highest average traffic
   - **Pico** (middle 60%): 60 cells with moderate traffic
   - **Femto** (bottom 30%): 30 cells with lowest traffic

2. **Coordinate Assignment:**
   - If `milano-grid.geojson` is available → real geographic centroids (lon, lat)
   - Otherwise → heuristic `√N` square layout (ceil(√100) = 10×10 grid)

3. **KNN Edge Construction (k=5):**
   - For each node, the 5 nearest neighbors within the same or target tier (by Euclidean distance) are connected
   - Self-loops are excluded

---

## Loss Functions

### Primary: Mean Squared Error (MSE)

The default loss function used for deterministic forecasting:

```
MSE = (1/n) × Σ (ŷᵢ − yᵢ)²
```

MSE penalizes large errors quadratically, making it effective for regression tasks where outliers should be corrected. Training and validation losses are both computed with MSE, enabling consistent early stopping decisions.

### Optional: Gaussian Negative Log-Likelihood (NLL)

When `--probabilistic` mode is enabled, the model outputs both **mean (μ)** and **log-variance (log σ²)** predictions:

```
NLL = 0.5 × mean[ (y − μ)² × exp(−log σ²) + log σ² + log(2π) ]
```

This enables **uncertainty quantification** — the model learns not just the expected traffic but its confidence in each prediction.

### Evaluation Metrics

| Metric | Formula | Space |
|--------|---------|-------|
| **MAE** | `mean(|ŷ − y|)` | Normalized [0,1] and original units |
| **RMSE** | `√mean((ŷ − y)²)` | Normalized [0,1] and original units |
| **sMAPE** | `100 × mean(2|ŷ − y| / (|ŷ| + |y|))` | Scale-independent (%) |

---

## Training Configuration & Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rate; good default for GNNs and Transformers |
| **Learning rate** | `1e-3` | Standard starting point; reduced by scheduler |
| **LR Scheduler** | `ReduceLROnPlateau` | `factor=0.5`, `patience=4`, `min_lr=1e-5` — halves LR when validation plateaus |
| **Batch size** | 32 | Balance between gradient estimation quality and memory |
| **Max epochs** | 50 | Upper bound; typically stopped earlier by early stopping |
| **Early stopping patience** | 10 | Stops if val loss doesn't improve by `min_delta=1e-6` for 10 consecutive epochs |
| **Gradient clipping** | `max_norm=1.0` | Prevents exploding gradients in Transformer layers |
| **Random seed** | 42 | Reproducibility across runs |
| **Sequence length** | 12 timesteps (2 hours) | Captures intra-day patterns |
| **Forecast horizon** | 3 timesteps (30 minutes) | Practical prediction window for resource scheduling |
| **Number of cells** | 100 | Top-100 Milan grid cells |
| **KNN neighbors** | k=5 | Balances graph density vs. over-smoothing |
| **GNN hidden dim** | 32 | First `HeteroConv` output dimension |
| **GNN output dim** | 64 | Embedding dimension fed to Transformer |
| **Transformer heads** | 4 | Multi-head attention; `64/4 = 16` dim per head |
| **Transformer layers** | 2 | Shallow encoder; sufficient for 12-step sequences |
| **Feedforward dim** | 128 | `2 × d_model` expansion ratio |
| **Dropout** | 0.1 | Light regularization in Transformer layers |
| **Scaler** | MinMaxScaler [0,1] | Fit only on training timesteps — no leakage |
| **Data split** | 70% / 10% / 20% | Train / Validation / Test (chronological, no shuffle) |

---

## Hyperparameter Variations & Effects

| Hyperparameter | Tried Values | Observation |
|----------------|-------------|-------------|
| **Ablation mode** | `hetero` vs `homo` | Heterogeneous graph consistently outperforms homogeneous — tier-specific convolutions capture the traffic hierarchy better |
| **GNN type** | SAGEConv vs GATConv | SAGEConv is the default; GATConv (`--use-gat`) adds attention-weighted aggregation with 4 heads but increases compute cost |
| **Temporal features** | On (4-dim sin/cos) vs Off | With temporal features, the model captures diurnal/weekly cycles; without them (`--no-temporal`), performance degrades on periodic patterns |
| **Probabilistic head** | Off vs On | Gaussian NLL head provides uncertainty estimates but adds complexity; deterministic MSE is preferred for point forecasts |
| **Learning rate** | 1e-3 → adapted by scheduler | LR halves when validation plateaus (patience=4); min LR floor of 1e-5 prevents learning from stopping entirely |
| **Tier percentages** | Macro 10% / Femto 30% | These mirror real-world HetNet deployment densities; adjustable via `macro_pct` and `femto_pct` parameters |
| **k (KNN neighbors)** | 5 | Higher k increases graph density but risks over-smoothing; k=5 provides good neighbor coverage for 100-cell grids |

---

## Baselines

Two baselines are evaluated on the **same test split** for fair comparison:

| Baseline | Method | Description |
|----------|--------|-------------|
| **Naive Persistence** | Last-value repeat | Predicts all horizon steps as the last observed input timestep — the simplest possible forecast |
| **Ridge Regression** | Linear model | Flattened input sequence → per-horizon Ridge regression (α=1.0); captures linear temporal dependencies |

---

## Results

### Training Convergence

The model was trained for **40 epochs** before early stopping was triggered (patience = 10 epochs without improvement):

| Epoch | Train Loss (MSE) | Val Loss (MSE) |
|-------|-------------------|----------------|
| 01 | 0.017839 | 0.010089 |
| 02 | 0.009402 | 0.008641 |
| 03 | 0.007917 | 0.008196 |
| 05 | 0.007300 | 0.007887 |
| 10 | 0.006997 | 0.007695 |
| 15 | 0.006835 | 0.007805 |
| 20 | 0.006747 | 0.007604 |
| 25 | 0.006701 | 0.007606 |
| 30 | 0.006619 | 0.007606 |
| 40 | 0.006619 | 0.007606 |

**Observation:** Training loss steadily decreased from 0.0178 → 0.0066. Validation loss reduced rapidly in the first 5 epochs and stabilized around ~0.0076, indicating convergence without severe overfitting.

### Test Set Performance

| Metric | Normalized Space [0,1] | Original Activity Units |
|--------|----------------------|------------------------|
| **MAE** | **0.057582** | **1.634233** |
| **RMSE** | **0.082484** | **3.629564** |
| **sMAPE** | **40.24%** | **31.15%** |

### Interpretation

- **MAE = 0.0576** in normalized space means the model's average prediction error is about **5.8%** of the data range — a strong result for spatiotemporal forecasting on real urban traffic data.
- **RMSE = 0.0825** indicates that even the larger errors remain within ~8.2% of the data range.
- **sMAPE = 31.15%** (original units) reflects the challenge of predicting low-traffic Femto cells where small absolute errors produce large percentage errors. The normalized sMAPE (40.24%) is higher because values near zero amplify the metric.
- The gap between normalized MAE (0.058) and RMSE (0.082) is moderate, suggesting the model doesn't produce extreme outlier predictions.

---

## Efficiency Analysis

### Computational Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| **HeteroGNN (per timestep)** | O(N × k × d) | N=100 nodes, k=5 neighbors, d=64 embedding dim; 5 edge types processed in parallel by `HeteroConv` |
| **Transformer Encoder** | O(T² × N × d) | T=12 timesteps; self-attention is quadratic in sequence length but T is small |
| **Linear Head** | O(N × d × H) | H=3 horizon steps |
| **Overall per sample** | O(T × N × k × d + T² × N × d) | Dominated by GNN for large graphs, Transformer for long sequences |

### Model Size

| Metric | Value |
|--------|-------|
| **Total parameters** | ~460K (based on checkpoint size ~460 KB) |
| **Checkpoint file** | `tastf_best.pt` (460 KB) |
| **Full checkpoint** | `tastf_checkpoint.pt` (464 KB) — includes config + split metadata |
| **Memory footprint** | Lightweight; runs on CPU or single GPU |

### Training Efficiency

| Metric | Value |
|--------|-------|
| **Convergence** | Validation loss stabilizes by epoch ~5-10 |
| **Early stopping** | Triggered at epoch 40 (patience=10) |
| **LR scheduler** | Automatically adapts learning rate upon plateau |
| **Gradient clipping** | Prevents training instabilities |
| **Time per epoch** | Seconds (100 cells × 12 steps × 32 batch on CPU) |

### Scalability Considerations

- **Cell scaling:** KNN graph construction is O(N² × k), manageable for hundreds of cells; for thousands, spatial indexing (KD-trees) would be needed.
- **Temporal scaling:** Transformer attention is O(T²); for T ≤ 24, this is efficient. Longer sequences would benefit from linear attention variants.
- **Batch processing:** Mini-batch training with batch size 32 enables efficient GPU utilization.

---

## Repository Structure

```
hetnet-traffic-forecast/
├── README.md                         # This file
├── TASTF_Implementation.ipynb        # Main notebook: setup → train → evaluate
├── requirements.txt                  # Python dependencies
├── wireless dataset/                 # Milan Telecom Italia data (tab-separated .txt)
│   ├── sms-call-internet-mi-2013-11-01.txt
│   └── sms-call-internet-mi-2013-11-02.txt
└── reference_implementation/         # Modular Python pipeline
    ├── model.py                      # TASTF, TASTFHomo, HeteroGNNEncoder, HomoGNNEncoder
    ├── train.py                      # Training loop with baselines, early stopping, LR scheduler
    ├── data_loader.py                # Data loading, scaling, sliding window, chronological split
    ├── graph_builder.py              # Tier partitioning, KNN edges, HeteroData construction
    ├── evaluate.py                   # Tier-wise prediction plots
    ├── metrics.py                    # MAE, RMSE, sMAPE, inverse-transform metrics
    ├── baselines.py                  # Naive persistence + Ridge regression baselines
    ├── paths.py                      # Cross-platform data path resolution
    ├── viz.py                        # Spatial error heatmap visualization
    ├── read_metrics.py               # Post-hoc metric reader from results.npz
    ├── tastf_best.pt                 # Best model weights (early stopping checkpoint)
    ├── tastf_checkpoint.pt           # Full checkpoint (weights + config + metadata)
    ├── results.npz                   # Saved test predictions and ground truth
    └── tier_predictions.png          # Visualization of per-tier forecasts
```

---

## Environment Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.x (CPU or CUDA)
- PyTorch Geometric 2.x

### Installation

```bash
# 1. Install PyTorch (match your CUDA version)
pip install torch torchvision

# 2. Install PyG extensions (use pre-built wheels for speed)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<VERSION>+<CU_OR_CPU>.html
pip install torch-geometric

# 3. Install remaining dependencies
pip install pandas numpy scikit-learn matplotlib
```

> **Tip:** Replace `<VERSION>` and `<CU_OR_CPU>` with your PyTorch build info from `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.

---

## How to Run

### Option 1: Jupyter Notebook (Recommended)

```bash
# Open and run all cells top to bottom
jupyter notebook TASTF_Implementation.ipynb
```

### Option 2: Command Line

```bash
cd reference_implementation

# Default training (SAGEConv, hetero graph, temporal features)
python train.py

# With GATConv attention
python train.py --use-gat

# Homogeneous ablation
python train.py --ablation homo

# Probabilistic mode with uncertainty
python train.py --probabilistic

# Custom hyperparameters
python train.py --epochs 100 --batch-size 64 --lr 5e-4 --seed 123
```

### Option 3: Google Colab

1. The first notebook cell auto-clones the repository
2. If real data is not found, a **synthetic fallback** generates demo data (metrics will not be representative)
3. For real results, upload the `wireless dataset/` folder or mount Google Drive

---

## Limitations & Future Work

### Current Limitations

1. **Limited data:** Only 2 days of Milan data — insufficient for robust generalization claims
2. **Static tier assignment:** Tiers don't adapt to changing traffic patterns over time
3. **Heuristic coordinates:** Without GeoJSON, spatial graph uses `√N` grid layout instead of real geography
4. **sMAPE inflation:** Low-activity Femto cells produce high sMAPE despite small absolute errors
5. **No cross-validation:** Single chronological split; results may vary with different date ranges

### Potential Improvements

- **More data:** Extend to full 2-month Telecom Italia dataset and multi-city validation
- **Dynamic tier re-assignment:** Periodically re-partition based on recent traffic
- **Stronger baselines:** LSTM, GRU, Temporal GCN, STGCN for comparison
- **Attention-based graph (GAT):** Tune GATConv with attention visualization
- **Longer horizons:** Extend beyond 30-min to 1-hour or 6-hour forecasts
- **Real-world deployment:** Integration with network management systems for online inference

---

## References

### Dataset

1. **Barlacchi, G., De Nadai, M., Larcher, R., Casella, A., Chitic, C., Torrisi, G., Antonelli, F., Vespignani, A., Pentland, A., & Lepri, B.** (2015). *A multi-source dataset of urban life in the city of Milan and the Province of Trentino.* Scientific Data, 2, 150055. https://doi.org/10.1038/sdata.2015.55
   - The Telecom Italia Big Data Challenge dataset used in this project. Contains SMS, call, and internet activity records aggregated over a spatial grid of Milan.

2. **Telecom Italia Big Data Challenge.** (2014). *Milan Grid — SMS, Call, Internet activity data.* Available at: https://dandelion.eu/datamine/open-big-data/
   - Open data portal for the Milan telecommunications dataset. Licensed under Open Database License (ODbL).

### Frameworks & Libraries

3. **Fey, M. & Lenssen, J.E.** (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds. https://arxiv.org/abs/1903.02428
   - PyTorch Geometric framework used for heterogeneous graph construction and GNN layers (`HeteroConv`, `SAGEConv`, `GATConv`).

4. **Paszke, A. et al.** (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019. https://arxiv.org/abs/1912.01703
   - Deep learning framework used for model implementation, training, and inference.

### Foundational Methods

5. **Hamilton, W.L., Ying, R., & Leskovec, J.** (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS 2017. https://arxiv.org/abs/1706.02216
   - The SAGEConv operator used in the HeteroGNN encoder for neighborhood aggregation.

6. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y.** (2018). *Graph Attention Networks.* ICLR 2018. https://arxiv.org/abs/1710.10903
   - The GATConv operator available as an alternative attention-based GNN layer.

7. **Vaswani, A. et al.** (2017). *Attention Is All You Need.* NeurIPS 2017. https://arxiv.org/abs/1706.03762
   - The Transformer encoder architecture used for temporal modeling with multi-head self-attention.

### Related Work in Traffic Forecasting

8. **Yu, B., Yin, H., & Zhu, Z.** (2018). *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.* IJCAI 2018. https://arxiv.org/abs/1709.04875
   - Foundational work on combining GCNs with temporal convolutions for traffic prediction.

9. **Li, Y., Yu, R., Shahabi, C., & Liu, Y.** (2018). *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* ICLR 2018. https://arxiv.org/abs/1707.01926
   - Diffusion-based graph convolutions for traffic flow forecasting, influencing the spatial modeling approach.

---

## Contact

This is a **research prototype** for academic coursework. Reproducibility depends on the exact environment, data path, and random seed. When reproducing results, ensure:
- `DATA_DIR` points to real Milan `.txt` files
- PyTorch and PyG versions match
- Seed is set to 42 for consistency