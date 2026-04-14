# TASTF: Tier-Aware Spatiotemporal Forecasting for HetNets

## 🚀 Project Overview
**TASTF** is an advanced spatiotemporal forecasting framework designed specifically for Heterogeneous Cellular Networks (HetNets). Unlike traditional models that treat all base stations (BS) as identical nodes, TASTF acknowledges the physical diversity of network tiers (**Macro, Pico, and Femto**) to improve traffic prediction accuracy.

### Key Novelty
The core innovation is the construction of a **Heterogeneous Graph** where node types and edge types are tier-specific. This allows the model to learn distinct spatial propagation patterns for different BS classes (e.g., high-power Macro BS vs. low-power Femto BS).

---

## 🏗️ Model Architecture
The TASTF pipeline consists of three primary sequential components:

1.  **HeteroGNNEncoder**: 
    -   Utilizes **PyTorch Geometric (PyG)** `HeteroConv`.
    -   Applies tier-specific `SAGEConv` layers.
    -   Models both **intra-tier** (Geo-KNN) and **cross-tier** (Macro influence) spatial dependencies.
2.  **TransformerEncoder**:
    -   Stacks 12 timesteps (2 hours of 10-min intervals) of GNN embeddings into a sequence.
    -   Uses temporal self-attention to capture long-range dependencies.
3.  **Linear Prediction Head**:
    -   Projects the final temporal context into a 3-step forecast horizon (30 minutes).

---

## 📊 Results & Key Metrics
The model was trained and validated on the **Telecom Italia Milan Dataset** (November 2013).

### Performance Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **MAE** | 0.1049 | Average absolute deviation in normalized load. |
| **RMSE** | 0.1508 | Standard deviation of prediction residuals. |
| **MAPE** | 1486.99% | High due to sparse zero-traffic intervals in grid cells. |

### Visualization
The model successfully tracks the diurnal patterns and bursty traffic behaviors across all tiers.
- **Macro BS**: High amplitude, periodic traffic patterns.
- **Pico/Femto BS**: More localized, bursty traffic signatures.

*(Refer to `reference_implementation/tier_predictions.png` for the full visual report)*

---

## 📂 Project Structure
```text
project/
├── TASTF_Implementation.ipynb   # 🏆 Consolidated interactive notebook (MAIN ENTRY POINT)
├── TASTF_Implementation_Plan.docx # Original design document
├── wireless dataset/            # Raw Milan traffic data (.txt files)
└── reference_implementation/    # Modular Python source code
    ├── data_loader.py           # Multi-file dataset loader
    ├── graph_builder.py         # Tier assignment & Graph logic
    ├── model.py                 # PyTorch Neural Network definitions
    ├── train.py                 # Training script with Early Stopping
    ├── evaluate.py              # Metrics & Visualization logic
    ├── tastf_best.pt            # Trained model weights
    ├── results.npz              # Serialized prediction results
    └── tier_predictions.png     # Visual prediction report
```

---

## ✅ What Has Been Done
- [x] **Analysis**: Thoroughly analyzed the TASTF implementation plan and research gaps.
- [x] **Data Engineering**: Implemented a robust 8-column data loader for Telecom Italia telemetry.
- [x] **Graph Construction**: Developed logic to automatically assign BS tiers based on activity thresholds.
- [x] **Model Implementation**: Built the integrated HeteroGNN-Transformer pipeline.
- [x] **Training Suite**: Established a full lifecycle script with gradient clipping and validation-based early stopping.
- [x] **Consolidation**: Migrated the entire workflow into a single, user-friendly Jupyter Notebook.
- [x] **Visualization**: Generated comparative plots for all three BS tiers.

---

## ⏳ What is Left (Future Directions)
1.  **Non-Stationarity Module**: Integrating a module to handle shifting traffic distributions (planned for Solution 3).
2.  **DRL Integration**: Utilizing TASTF forecasts as inputs for Proactive Resource Management via Deep Reinforcement Learning (planned for Solution 4).
3.  **Dynamic Graph Weighting**: Implementing attention-based edge weights in the GNN for better spatial focus.
4.  **Multi-Dataset Validation**: Testing on other city datasets (e.g., Call Detail Records from Beijing).

---

## 🛠️ Setup & Running
1.  **Install dependencies**:
    ```bash
    pip install torch-geometric torch-scatter torch-sparse pandas numpy scikit-learn matplotlib
    ```
2.  **Run the Notebook**:
    Open `TASTF_Implementation.ipynb` in VS Code or Jupyter and execute the cells sequentially.
