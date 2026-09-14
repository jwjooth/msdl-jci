# MSDL-JCI Complete Audit & Fix — Implementation Plan

## Root Cause Analysis (Phase 0–1 Complete)

### Confirmed Diagnosis

The model is suffering from **complete model collapse** — not a label bug, not an evaluation bug, but a training failure where the model converges to a near-constant output.

**Evidence:**
- All 377 test predictions fall in `[0.5035, 0.5262]` — std = 0.0044
- The model predicts positive for **100%** of test samples
- ROC-AUC = 0.5028 (indistinguishable from random)
- Inverted AUC = 0.4972 (confirms no label inversion — just random)
- Test positive rate = 57.56%, matching reported "accuracy" exactly

### Root Causes Identified

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | **Scaler data leakage** — `MinMaxScaler` is `fit_transform` on the ENTIRE dataset before train/val/test split | CRITICAL | [dataset_builder.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/utils/dataset_builder.py#L306-L310) |
| 2 | **No class weighting** — BCEWithLogitsLoss with no `pos_weight`, 54% positive class imbalance drives the model toward always-positive | HIGH | [walk_forward.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/evaluation/walk_forward.py#L89) |
| 3 | **No random seed in main proposed pipeline** — `run_proposed_deep_learning_pipeline()` does not set seeds | HIGH | [main.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/main.py#L19-L56) |
| 4 | **No embargo between splits** — with 5-day prediction horizon, the last 5 training samples' targets overlap with the first 5 validation samples' features | MEDIUM | [walk_forward.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/evaluation/walk_forward.py#L192-L193) |
| 5 | **Macro `bfill()` at boundary** — backfill leaks future macro data into the start of the dataset | MEDIUM | [dataset_builder.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/utils/dataset_builder.py#L241-L243) |
| 6 | **No training diagnostics** — no logging of per-epoch metrics, gradient norms, or probability distributions, making training failures invisible | MEDIUM | [walk_forward.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/evaluation/walk_forward.py#L99-L144) |
| 7 | **Trading simulation has no transaction costs** — gross returns are unrealistically reported as actual performance | MEDIUM | [trading_simulation.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/evaluation/trading_simulation.py) |
| 8 | **Single split evaluation** — no multi-fold walk-forward, no statistical testing | MEDIUM | [walk_forward.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/evaluation/walk_forward.py#L173-L261) |
| 9 | **MacroMLPBranch uses BatchNorm1d** — with small validation batch sizes, BatchNorm can behave erratically; it also means the same input gives different outputs in train vs eval mode for identical data | LOW | [fusion.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/models/fusion.py#L77-L85) |
| 10 | **No balanced accuracy, MCC, PR-AUC in metrics** — the current metric set is insufficient to detect model collapse | LOW | [metrics.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/evaluation/metrics.py) |

> [!CAUTION]
> **Issue #1 (Scaler Leakage)** is the most dangerous: the scaler is fitted on the entire dataset including test data, then those scaled values are used to create windowed features. This means the model has implicit knowledge of test-period feature ranges during training.

> [!IMPORTANT]
> **Issue #2 (No Class Weighting) + Model Collapse** is the proximate cause of the "57.56% accuracy, 100% recall" symptom. The model learns that always predicting positive minimizes unweighted BCE loss when the positive class is ~54%.

## Proposed Changes

### Phase 1: Fix Critical Data Pipeline Bugs

---

#### [MODIFY] [dataset_builder.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/utils/dataset_builder.py)

1. **Remove global scaler fit** — make `create_multisource_tensors` accept a `train_end_idx` parameter. Fit scalers only on `[:train_end_idx]` rows, then transform the rest
2. **Remove `bfill()` for macro** — only forward-fill, drop rows where macro data is genuinely unavailable at the start
3. **Add `future_close` column protection** — ensure it's never included in features

#### [MODIFY] [walk_forward.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/evaluation/walk_forward.py)

1. **Add embargo of 5 samples** between train/val and val/test to prevent label leakage
2. **Add `pos_weight` to BCEWithLogitsLoss** computed from training set only
3. **Add comprehensive per-epoch logging**: train/val loss, val AUC, val balanced accuracy, predicted positive ratio, gradient norms, gating weight means/entropy
4. **Set all random seeds** deterministically at the start of training
5. **Add threshold optimization** on validation set using MCC or balanced accuracy

---

### Phase 2: Enhance Metrics & Evaluation

#### [MODIFY] [metrics.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/evaluation/metrics.py)

Add: balanced accuracy, MCC, PR-AUC, log loss, Brier score

#### [MODIFY] [trading_simulation.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/evaluation/trading_simulation.py)

Add: transaction costs (0.15% per side), slippage (0.05% per side), gross vs net returns, Sortino ratio, exposure %, turnover

---

### Phase 3: Model Training Stabilization

#### [MODIFY] [fusion.py](file:///c:/Users/jorda/Documents/kuliah%20hohohoho/semester%207/project/MSDL-JCI/src/msdl_jci/models/fusion.py)

1. **Initialize classifier bias** to `log(pos_rate / (1 - pos_rate))` so the model starts calibrated
2. **Initialize gating logits near zero** for balanced initial weights  
3. **Replace BatchNorm1d with LayerNorm** in MacroMLPBranch (more stable with small batches)
4. **Add optional modality dropout** — randomly zero out entire branches during training
5. **Add optional auxiliary branch losses** — each branch gets its own classification head during training
6. **Add branch output normalization** before gating to prevent scale dominance

---

### Phase 4: Robust Walk-Forward Validation

#### [NEW] `src/msdl_jci/evaluation/robust_walk_forward.py`

Implement multi-fold expanding-window walk-forward with:
- Configurable fold count
- Purging/embargo (≥5 samples)
- Per-fold metrics
- Aggregate statistics (mean, std, CI)
- Statistical tests (Wilcoxon, bootstrap)

---

### Phase 5: Comprehensive Baselines

#### [NEW] `scripts/run_full_audit.py`

Master script that:
1. Builds dataset with proper train-only scaling
2. Runs all baselines: always-positive, always-negative, random, previous-day, momentum, logistic regression, random forest, gradient boosting, and all 5 neural models
3. Computes full metric suite for each
4. Saves per-phase reports to `reports/`
5. Generates comparison tables and plots

---

### Phase 6: Diagnostic Scripts

#### [NEW] `scripts/diagnose_evaluation.py`

Standalone diagnostic producing probability histograms, ROC curves, calibration plots, confusion matrices, label distributions.

---

### Phase 7: Tests

#### [NEW] `tests/unit/test_target_alignment.py`

Verify:
- Target at row i uses only `Close[i+5]` vs `Close[i]`
- No future feature leakage
- Dataset preserves chronological order
- Target/feature alignment after windowing

#### [NEW] `tests/unit/test_no_lookahead.py`

Verify:
- Scaler fitted on train only
- Macro data doesn't backfill
- News embedding date ≤ prediction date
- Walk-forward embargo respected

---

### Phase 8: Reports

All phase reports written to `reports/00_baseline_reproduction/` through `reports/10_final_report.md` as specified.

## Open Questions

> [!IMPORTANT]
> **Q1: Macro publication lag** — The BI Rate and inflation data files contain period dates, not publication dates. Should we apply a conservative 1-month lag (i.e., January data not available until February 1st), or is the current `merge_asof(direction='backward')` sufficient if the data is already dated by publication?

> [!IMPORTANT]
> **Q2: News embedding construction** — Were the IndoBERT embeddings computed using only news articles published before market close on each date? If any articles from after hours or next morning are included, this is a leakage source.

> [!NOTE]  
> **Q3: Target threshold** — Currently `y = 1 if close_{t+5} > close_t` (strictly greater). Zero returns are labeled as 0 (DOWN). This is a reasonable convention but should be documented.

## Verification Plan

### Automated Tests
```
uv run pytest -v                              # All existing + new tests
uv run pytest tests/unit/test_target_alignment.py -v
uv run pytest tests/unit/test_no_lookahead.py -v
```

### Reproduction
```
uv run python scripts/run_full_audit.py       # Full audit pipeline
uv run python scripts/diagnose_evaluation.py  # Evaluation diagnostics
```

### Expected Outcome After Fixes
- Model should **no longer predict all-positive** — probability distribution should span a meaningful range
- Balanced accuracy should be meaningfully above 50%
- MCC should be positive if the model has learned anything
- ROC-AUC should be > 0.5 if the model discriminates at all
- If the model still collapses after all fixes, the final report will honestly state that the proposed architecture does not provide predictive value for this dataset
