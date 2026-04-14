# TASTF: Tier-Aware Spatiotemporal Forecasting for HetNets

## Project status (as of last consolidated work)


| Area                         | Status                   | Notes                                                                                                                                                                                                                                |
| ---------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| End-to-end notebook pipeline | **Working**              | `TASTF_Implementation.ipynb` runs: data load → hetero graph → TASTF model → training with early stopping → test MAE/RMSE and plots.                                                                                                  |
| Real Milan data in repo      | **Present**              | Two-day slice: `wireless dataset/sms-call-internet-mi-2013-11-01.txt`, `sms-call-internet-mi-2013-11-02.txt`.                                                                                                                        |
| Google Colab                 | **Working with caveats** | If the repo is not cloned/uploaded, the notebook can generate **synthetic** Telecom-Italia-shaped `.txt` files under `/content/wireless dataset` so cells execute; metrics from that run are **not** representative of real traffic. |
| Reference scripts            | **Available**            | `reference_implementation/` contains modular `train.py`, `evaluate.py`, `model.py`, etc.; may have been used for earlier metrics and `tier_predictions.png`.                                                                         |
| Research-grade benchmarking  | **Not complete**         | No fixed leaderboard run, weak baselines in notebook, train/val logging has a known scaling quirk (see below).                                                                                                                       |


**Bottom line:** The project is a **functional research prototype** with a **single main notebook** as the primary entry point. Core ideas (hetero GNN + transformer on tiered cells) are implemented; **reported numbers depend on environment (local real data vs Colab synthetic)** and should be interpreted with the **Results** and **Bottlenecks** sections below.

---

## What this project is

**TASTF (Tier-Aware Spatiotemporal Forecasting)** targets **heterogeneous cellular networks (HetNets)** where base stations differ by tier (**Macro, Pico, Femto**). Instead of treating every cell as identical, the model:

1. **Partitions** grid cells into tiers by mean activity (configurable top/bottom fractions).
2. **Builds a heterogeneous graph** (`torch_geometric.HeteroData`) with intra-tier (geo-like KNN) and cross-tier (e.g. macro → pico/femto) edges.
3. Runs a **HeteroConv / SAGEConv** encoder per time step, then a **Transformer encoder** over the sequence of spatial embeddings, and a **linear head** for multi-step forecasts.

**Forecast setting (notebook defaults):**

- **Input:** 12 timesteps (2 hours at 10-minute resolution).
- **Output:** 3 steps ahead (30 minutes).
- **Cells:** up to `N_CELLS = 100` (filters Milan grid IDs ≤ 100).
- **Target signal:** `activity = internet + call_in + call_out` (per row), then pivoted by time × grid; values are **MinMax-scaled** to [0, 1] for training.

---

## Repository layout

```text
hetnet-traffic-forecast/
├── TASTF_Implementation.ipynb    # Main entry: installs PyG (optional), loads data, trains, evaluates
├── README.md                     # This file
├── wireless dataset/             # Milan Telecom Italia open data (tab-separated .txt)
│   ├── sms-call-internet-mi-2013-11-01.txt
│   └── sms-call-internet-mi-2013-11-02.txt
└── reference_implementation/     # Alternate modular pipeline
    ├── data_loader.py
    ├── graph_builder.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── read_metrics.py           # Loads results.npz and prints MAE/RMSE/MAPE
    ├── tastf_best.pt           # Saved weights (if training was run)
    ├── results.npz             # Saved predictions (if present)
    └── tier_predictions.png    # Tier plot (if generated)
```

---

## Data

**Source:** Telecom Italia Big Data Challenge — **Milan** SMS / call / internet aggregates on a grid (open data; ODbL-style attribution applies to official releases).

**Local files (this repo):**

- `hetnet-traffic-forecast/wireless dataset/*.txt` — tab-separated rows; loader uses **all** `*.txt` in the chosen directory (sorted by filename).

**Path resolution (`resolve_wireless_dataset_dir` in the notebook):**

- Tries explicit path, VS Code/Cursor notebook directory, common relative paths (`hetnet-traffic-forecast/wireless dataset`, etc.), then walks parents.
- **Colab:** if no `.txt` are found and `/content` exists (or `TASTF_USE_SYNTHETIC=1`), **synthetic** demo files are written so the pipeline runs without uploading data.

---

## Environment and setup

**Core Python stack:** PyTorch, PyTorch Geometric, pandas, NumPy, scikit-learn, matplotlib.

**Important — PyTorch Geometric extensions:**

- Installing `torch-scatter` / `torch-sparse` **from source** (plain `pip install` without wheels) can take **tens of minutes** or fail on mismatched CUDA/toolchains.
- The notebook uses **pre-built wheels** from `https://data.pyg.org/whl/` matched to your installed `torch` and CUDA/CPU build. If install fails, open the index URL in a browser and pick the folder that matches `import torch; print(torch.__version__, torch.version.cuda)`.

**Minimal manual install (after PyTorch is installed):**

```bash
pip install pandas numpy scikit-learn matplotlib
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<VERSION>+<CU_OR_CPU>.html
pip install torch-geometric
```

Replace `<VERSION>` and `<CU_OR_CPU>` with your PyTorch build (see PyG wheel index).

---

## How to run

1. Install PyTorch (and CUDA if GPU) **first**, then PyG as above.
2. Open `TASTF_Implementation.ipynb` and run cells **top to bottom**.
3. **Local (recommended for real results):** open the project folder so `wireless dataset` is discoverable, or set
  `DATA_DIR = resolve_wireless_dataset_dir(r"<full path>\hetnet-traffic-forecast\wireless dataset")`.
4. **Colab:** upload the `wireless dataset` folder or `git clone` this repository; otherwise the run may use **synthetic** data.

**Training defaults (notebook):** batch size 32, Adam `lr=1e-3`, up to 50 epochs, early stopping patience 10, gradient clip 1.0. Checkpoint: `tastf_notebook.pt`.

---

## Model architecture (summary)

- **HeteroGNNEncoder:** two `HeteroConv` stages with `SAGEConv` on edge types `(macro|pico|femto, geo/cross, ...)`, ReLU, `LayerNorm`, per-type linear projections.
- **PositionalEncoding:** sinusoidal, added to temporal sequence of node embeddings.
- **TransformerEncoder:** default 2 layers, 4 heads, `d_model=64`, feedforward 128.
- **Head:** linear to `horizon` steps; output shape aligned to `(batch, horizon, nodes)`.

Tier indices (macro/pico/femto) come from **training data** statistics via `build_hetero_graph`.

---

## Results

Metrics below are on **MinMax-normalized [0, 1]** targets unless you invert the scaler for physical units.

### Latest notebook run captured in `TASTF_Implementation.ipynb` (Google Colab-style log)

- **DATA_DIR** printed as `/content/wireless dataset` → indicates **Colab**; combined with project history, this run most likely used **synthetic** demo data when the real folder was not mounted.
- **Validation MSE** (~0.079) was **flat** after the first few epochs; **early stopping** at epoch **18** (patience 10).
- **Test (printed):** **MAE ≈ 0.2446**, **RMSE ≈ 0.2824**.

**Interpretation:** The pipeline **completed** and produced stable but **moderate** errors in normalized space. This run is **not** a substitute for evaluation on the **real** Milan files in `wireless dataset/`.

### Earlier reference implementation metrics (historical)

The previous README listed **MAE ≈ 0.1049**, **RMSE ≈ 0.1508**, and a very high **MAPE** from sparse zeros — likely from `reference_implementation` + `results.npz` / `read_metrics.py`. Those numbers **differ** from the notebook Colab run above; treat them as **another experiment/config**, not as the same run as the current notebook.

**Recommendation:** For any report, **re-run** evaluation with **explicit** real `DATA_DIR`, record **seed**, **torch/PyG versions**, and **baselines** (naive persistence, linear).

---

## Bottlenecks, limitations, and known issues

1. **Environment split (local vs Colab):** Paths and data availability differ; Colab may silently use **synthetic** data if real `.txt` files are missing.
2. **PyG install friction:** `torch-scatter` / `torch-sparse` builds from source are slow; always prefer **PyG wheel index** matching your PyTorch build.
3. **Validation plateau:** Observed val ~0.079 with little improvement — suggests **limited model capacity**, **need for tuning**, or **data/process noise**; possible **overfitting** to training noise (train loss still drifts down slightly in the log).
4. **No strong baselines in notebook:** Hard to claim SOTA or even “good” without naive/linear baselines on the same split.
5. **Grid / tier simplification:** `sqrt(N)` layout for KNN is a **heuristic**; real Milan geography may not match a square grid index ordering.
6. **Scope of bundled data:** Only **two days** of Milan files are in-repo; generalization claims need more dates and external cities.
7. **Licensing:** Use official **Telecom Italia / Big Data Challenge** attribution when redistributing or publishing on open data.

---

## What is done vs what remains

**Done**

- Notebook-first story: data loading, tier graph, TASTF model, training loop, checkpoint, test metrics, visualization cell.
- Robust data path resolution and Colab **fallback** synthetic generator.
- Modular reference code and optional artifacts (`tastf_best.pt`, `results.npz`, plots).

**Remaining / future work**

- Fix training loss **averaging** for honest train/val comparison.
- Add **baselines** and optional **inverse transform** metrics in original units.
- Tune learning rate, model size, regularization; try more days of Milan data.
- Optional: dynamic graphs, non-stationarity, multi-city validation, integration with planning/RL (as in original roadmap).

---

## Attribution

Telecom Italia Milan open data is used for research purposes; cite the **Telecom Italia Big Data Challenge** and relevant **open data** terms when publishing. This README does not replace the dataset license text from the original distributor.

---

## Contact / maintenance

Treat this repository as a **research codebase**: reproducibility depends on **exact environment**, **data path**, and whether runs use **real** or **synthetic** inputs. When in doubt, print `DATA_DIR`, list `*.txt` in that folder, and confirm `torch`/`torch_geometric` versions in the first notebook cells.