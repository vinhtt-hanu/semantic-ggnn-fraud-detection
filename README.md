# Language Model–Enhanced Graph Neural Networks for Financial Fraud Detection

Code and results accompanying the manuscript *"Language Model–Enhanced Graph Neural Networks for Financial Fraud Detection"* (submitted to *Discover Artificial Intelligence*, Springer). All experiments use the public PaySim mobile-money dataset (Lopez-Rojas et al., 2016).

The repository has two parts with different status:

| Folder | Content | Status |
|---|---|---|
| `GPU_Evidence_30Neural_15Tabular_Runs/` | Controlled, strictly chronological five-seed study (Section 7 of the manuscript) | **Current results**; reproducible |
| `notebooks/` | Archived original notebooks of pipelines P3 and P4 | **Historical code**; not a leakage-free or matched comparison |

## 1. Controlled causal ego-event study (`GPU_Evidence_30Neural_15Tabular_Runs/`)

Six causal ego-event graph models (redefined analogues of the original semantic-first and semantic-last pipelines, not reproductions of them) and three tabular comparators, each run with five seeds (11, 22, 33, 44, 55).

**Temporal protocol**

| Interval | PaySim steps | Use |
|---|---|---|
| Observation history | < 483 | Account statistics and event history only; labels not used |
| Training | 483–594 | All 1,180 frauds + 18,000 sampled normal transactions (19,180 rows) |
| Validation | 595–668 | All 54,732 transactions (808 frauds); checkpoint and threshold selection |
| Test | 669–743 | All 68,848 transactions (846 frauds); reported once |

History events are strictly earlier than the scored transaction's step. Prediction inputs exclude labels, post-transaction balances and residuals derived from them. The frozen language model is `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

**Files**

| File | Content | Manuscript |
|---|---|---|
| `experiment.py` | Data loading, temporal query selection, causal ego-event graph construction, text templates | Sections 7.1–7.3 |
| `train.py` | Frozen LM encoding, model definitions, training, threshold selection, tabular baselines, paired statistics | Sections 7.3–7.6 |
| `independent_verify.py` | Re-scores every saved test prediction and checks the reported metrics | Section 7.7 |
| `study_protocol.json`, `training_run_config.json` | Cut-offs, row counts, encoder revision, run settings | Section 7.1, Table 7 |
| `graph_audit.json` | Graph statistics and causality checks | Section 7.2, Fig. 10 |
| `train_only_scalers.json` | Scalers fitted on the training interval only | Section 7.3 |
| `frozen_minilm_text_bank.json` | Metadata of the 1,913 discretized text profiles and the LM checkpoint | Section 7.3 |
| `results_per_seed.csv`, `results_mean_std.csv` | Per-seed and mean ± SD test metrics of the neural models | Table 7, Fig. 11 |
| `paired_differences.csv`, `paired_uncertainty.json` | Paired seed differences and 95% t intervals | Table 8 |
| `tabular_baselines_same_inputs.csv` | Random Forest, XGBoost and LightGBM on the same inputs and rows | Table 9 |
| `test_predictions/*.npz` | Test-set scores of all 30 neural runs (model × seed) | Section 7.7 |
| `independent_verification.json` | Output of `independent_verify.py` | Section 7.7 |
| `controlled_fusion_five_seeds_pr_auc.png` | Per-seed test PR-AUC plot | Fig. 11 |
| `RUN_AUDIT_REPORT.md` | Short summary of the run and its scope | — |

**Main results (test PR-AUC, mean ± sample SD over five seeds)**

| Model | PR-AUC |
|---|---|
| Semantic-first | 0.6643 ± 0.0149 |
| Semantic-last | 0.6628 ± 0.0184 |
| No GGNN | 0.6708 ± 0.0150 |
| Random Forest | 0.7506 ± 0.0070 |
| XGBoost | 0.9129 ± 0.0021 |
| LightGBM | 0.9205 ± 0.0000 |

The paired semantic-first minus semantic-last difference is +0.0015 with a 95% interval of [−0.0160, 0.0190], so no semantic-first advantage is established.

**Re-running**

The study was run on a Google Colab GPU runtime. Download the PaySim CSV (`PS_20174392719_1491204439457_log.csv`) from Kaggle, then call the functions in this order:

```python
from experiment import open_paysim, select_queries, build_causal_examples
from train import encode_frozen_lm, train_all, run_tabular_baselines

df   = open_paysim("PS_20174392719_1491204439457_log.csv")
qs   = select_queries(df)                    # cut-offs 483 / 595 / 669
data = build_causal_examples(df, qs)         # causal ego-event graphs
emb, lm_meta = encode_frozen_lm(data["texts"], "out")
train_all(data, qs, emb, "out")              # 6 models x 5 seeds (GPU required)
run_tabular_baselines(data, qs, "out")
```

To check the saved results without re-training, set the variable `E` at the top of `independent_verify.py` to the folder path and run `python independent_verify.py`.

Requirements: Python 3.10+, PyTorch with CUDA, `sentence-transformers`, `huggingface_hub`, `scikit-learn`, `xgboost`, `lightgbm`, `pandas`, `numpy`, `scipy`.

## 2. Archived original notebooks (`notebooks/`)

| Notebook | Pipeline |
|---|---|
| `GGNN -_ LLM + Transformer.ipynb` | P3: GGNN → Transformer + LM (semantic-last) |
| `Transformer + LLM -_ GGNN.ipynb` | P4: LM + Transformer → GGNN (semantic-first) |

These notebooks produced the historical P3/P4 values in Table 6 of the manuscript (P3: PR-AUC 0.0184; P4: PR-AUC 0.3209). They are kept for transparency only. They compute account statistics from the full dataset, expose graph information from later time steps, and use different language-model encoders, so their results are **not** a leakage-free or matched comparison and should not be used as evidence that one placement is better. Notebooks for P1 and P2 (GAT + Cluster) are not available.

## Data

PaySim is not redistributed here; obtain it from Kaggle under its original terms.

## Citation

If you use this code, please cite the manuscript (citation details will be added after publication).
