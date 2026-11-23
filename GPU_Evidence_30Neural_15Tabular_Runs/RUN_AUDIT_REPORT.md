# Run audit: controlled chronological semantic-fusion comparison on PaySim

## Execution summary

- Planned and completed GPU runs: 30 of 30 (six neural configurations × five seeds: 11, 22, 33, 44, 55).
- Tabular comparators: 15 of 15 (Random Forest, XGBoost, LightGBM × the same five seed labels).
- Input file: `PS_20174392719_1491204439457_log.csv` (PaySim, 6,362,620 transactions).
- Hardware: Google Colab GPU runtime.

## Temporal partitions

| Interval | Steps | Use |
|---|---|---|
| Observation history | < 483 | Account statistics and event history only; no labels used |
| Supervised training | 483–594 | All 1,180 frauds + 18,000 sampled normal transactions (19,180 rows) |
| Validation | 595–668 | All 54,732 transactions (808 frauds); checkpoint and threshold selection |
| Test | ≥ 669 | All 68,848 transactions (846 frauds); reported once |

## Graph and language-model inputs

- Graph: causal ego-event star per query (2 account roots + up to 3 prior transaction events per endpoint; at most 8 nodes). All history events are strictly earlier than the query step; same-step transactions are added to history only after every query at that step has been built.
- Historical transaction-event nodes: 206,861; predictions with at least one prior event: 55.2%.
- Frozen LM: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dimensions, 1,913 unique discretized text descriptions (no account identifiers or labels).
- Prediction inputs exclude fraud labels, post-transaction balances and residuals derived from them.

## Scope of the results

- The six neural configurations are redefined causal ego-event analogues of the original semantic-first/semantic-last pipelines. They are not reproductions of the archived P3/P4 notebooks, which used different LM encoders and full-data information.
- The fixed-size ego-event star approximates local account history; wider network effects in the full account graph are not modeled.
- Tabular comparators use the same flattened inputs and row partitions, but not an equivalent optimizer or compute budget.
- Paired 95% t intervals are computed across five seed pairs only and do not represent deployment-level uncertainty.

## Verification

`independent_verify.py` re-scores every saved per-seed test prediction file against the test labels and recomputes PR-AUC, ROC-AUC, precision and F1. Result (`independent_verification.json`): all 30 neural runs re-scored with no discrepancies. The 15 tabular rows are checked for completeness only, because tabular test scores were not saved; paired semantic-first minus semantic-last PR-AUC = +0.0015, 95% t interval [−0.0160, 0.0190].

The figure `controlled_fusion_five_seeds_pr_auc.png` is plotted directly from `results_per_seed.csv`.
