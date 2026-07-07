# Semantic Enrichment Placement in Transformer–GGNN Architectures for Financial Fraud Detection

Source code accompanying the paper *"Assessing Semantic Enrichment Placement in Transformer–GGNN Architectures for Imbalanced Financial Transaction Fraud Detection."* This repository contains the training/evaluation notebooks for the two semantic-enrichment pipelines (P3 and P4) compared in the study, run on the PaySim mobile-money transaction dataset.

## Repository structure

```
.
├── notebooks/
│   ├── pipeline3_semantic_last_ggnn_transformer_llm.ipynb   # P3: GGNN -> Transformer + LLM
│   └── pipeline4_semantic_first_transformer_llm_ggnn.ipynb  # P4: Transformer + LLM -> GGNN (proposed)
├── data/                # not tracked in git — see "Data" section below
└── README.md
```

> Note: place the two `.ipynb` files under `notebooks/` (or adjust paths below) when you initialize the repo.

## Pipelines in this repo

| File | Architecture | Role |
|---|---|---|
| `pipeline3_semantic_last_ggnn_transformer_llm.ipynb` | `GGNN → Transformer + LLM` | Semantic-last ablation (P3) |
| `pipeline4_semantic_first_transformer_llm_ggnn.ipynb` | `Transformer + LLM → GGNN` | Proposed semantic-first model (P4) |

The two GAT + Cluster baselines (P1: no SMOTE, P2: with SMOTE) referenced in the paper are separate notebooks not included in this upload — add them under `notebooks/` following the same naming convention if/when available, e.g. `pipeline1_gat_cluster_no_smote.ipynb`, `pipeline2_gat_cluster_smote.ipynb`.

## Data

Both notebooks expect the PaySim CSV (`Online Payments Fraud Analysis-Detection.csv` / `PS_20174392719_1491204439457_log.csv`) mounted from Google Drive in Colab:

```python
DATA_CSV = "/content/drive/MyDrive/Colab Notebooks/Online Payments Fraud Analysis-Detection.csv"
```

Update this path to match your own Drive/local layout. The dataset is **not** included in this repository (large file, third-party data) — download it from Kaggle ("PaySim" / "Online Payments Fraud Detection Dataset") and place it locally, or mount your own Drive folder.

## Environment

Both notebooks are written for **Google Colab** with a GPU runtime (developed/tested on A100) and pin a specific stack for `torch-geometric` compatibility:

- Python 3.10 (notebooks explicitly downgrade Colab's default Python via `apt-get`)
- PyTorch 2.3.0 + CUDA 12.1
- PyTorch Geometric (PyG) + `pyg-lib`
- `scikit-learn`, `pandas`, `numpy`

The first 1–2 cells of each notebook handle environment setup (Python downgrade, PyTorch/PyG install) — run them once per fresh Colab session before anything else.

## How to run

1. Open the notebook in Google Colab (GPU runtime recommended: Runtime → Change runtime type → GPU).
2. Run the setup cells (Python/PyTorch/PyG installation) — this may require a runtime restart when prompted.
3. Mount Google Drive and set `DATA_CSV` to your dataset path.
4. Run all remaining cells in order. Key stages in each notebook:
   - Global config (`CFG` dict: hidden dim, GGNN steps, neighbor sampling size, batch size, SMOTE ratio, etc.)
   - Data loading and feature engineering (node/edge feature construction, temporal train/val/test split)
   - Model definition (GGNN, Transformer feature-token encoder, LLM/semantic embedding module, depending on pipeline)
   - Training loop (warm-up phase + main phase, AMP-enabled)
   - Evaluation on the validation set, threshold calibration
   - Final test-set evaluation (ROC-AUC, PR-AUC, Precision, Recall, F1) and temporal fraud-checking-rate analysis

## Key configuration (shared across both pipelines for fair comparison)

| Parameter | Value |
|---|---|
| GGNN hidden dimension | 128 |
| GGNN message-passing steps | 5 |
| Neighbor sampling size | [25, 25] |
| Batch size | 8192 |
| SMOTE target positive ratio | 0.08 (applied in latent/transaction embedding space, training set only) |
| Split strategy | Temporal split by `step`: 80% train / 10% val / 10% test |

## Reference results (from the paper, same temporal split/protocol)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| P3: GGNN → Transformer + LLM (semantic-last) | 0.6687 | 0.0184 | 0.0130 | 0.0165 | 0.0145 |
| P4: Transformer + LLM → GGNN (semantic-first, proposed) | **0.8989** | **0.3209** | **0.2699** | **0.4326** | **0.3324** |

Re-running the notebooks end-to-end should reproduce results close to the table above; minor differences are expected across hardware/library versions and random seeds.

## Reproducibility notes

- Set random seeds explicitly if you need bit-for-bit reproducibility across runs (not all cells currently fix every seed).
- SMOTE and any other train-only transforms are fit on the training split only, then applied/frozen for validation and test, to avoid data leakage.
- The decision threshold is selected on the validation set (recall-constrained and/or calibrated via temperature scaling) before final test-set scoring — do not re-tune the threshold on the test set.

## Citation

If you use this code, please cite the accompanying paper:

```
[Author(s)]. "Assessing Semantic Enrichment Placement in Transformer–GGNN Architectures
for Imbalanced Financial Transaction Fraud Detection." [Target Journal], [Year].
```

## License

Add a license (e.g., MIT) appropriate for your intended use before making this repository public.
