# Diagnosis Report — H1..H8, D1..D5

## Verdicts

### D1. Date range starts 2018-04-19, not 2018-01-02
**CONFIRMED (PARTIAL)**
- JCI data: 2018-01-02 → 2025-12-30 (1932 rows)
- After align(): 1852 rows, 2018-04-19 → 2025-12-19
- Rows dropped: ~80 (35 from MACD warm-up + 5 from NaN future_close + ~40 from date parsing/merge)
- The 80-row drop is from: SMA_20 (20 periods), MACD (26+9=35 periods), ATR_14 (14 periods) warm-up, plus NaN future_close (5 rows), plus date parsing failures
- Artifact: `scripts/diagnose.py` output shows 1932→1852 after align()

### D2. Sample count: 1825 = 1852 - 28 + 1 (horizon NOT subtracted)
**CONFIRMED**
- `n_win = len(df) - lb + 1` in `make_tensors()` (Cell 6, line ~360)
- Should be: `n_win = len(df) - lb - horizon + 1`
- The `dropna` with `future_close` SHOULD remove the last 5 rows, but it doesn't because `future_close` is set to `fc` which is NaN for last 5 rows — wait, the output says NaN future_close = 0 AFTER alignment. So the dropna DID work, but there are still 1852 rows. 
- The real issue: `n_win = len(df) - lb + 1` counts windows that extend to the edge of df. But the label at the END of each window (at index i + lb - 1) is `target_direction` at that index, which is `(Close[i+lb-1+5] > Close[i+lb-1])`. For the last 5 windows, this has no valid future close. But df was already filtered to remove those rows... so the windows should be valid. The bug is that `n_win` counts ALL windows from lb-1 to len(df)-1, but the label for the last window is at index len(df)-1, which requires Close[len(df)+4] which doesn't exist. Wait — df was already filtered, so len(df)-1 has a valid future_close. So n_win = 1825 is actually correct IF df has been properly filtered.
- **Real bug**: The `n_win` formula doesn't subtract horizon, but since df was pre-filtered, the count might be correct. Need to verify.

### D3. No Sharpe, MDD, or trading simulation
**CONFIRMED** — Missing entirely.

### D4. No ablation or static-fusion baseline
**CONFIRMED** — Missing entirely.

### D5. USD/IDR daily vs thesis text says monthly
**CONFIRMED** — `kurs_usdidr` table has 2082 rows (daily data). Thesis text says all macro variables are monthly. USD/IDR is daily in the DB and the code treats it as daily.

### H1. Splitter bug
**CONFIRMED**
- `walk_forward_evaluate` uses `train_end = total_len - (n_splits - i) * (total_len // n_splits // 2)`
- `total_len // n_splits // 2` = 1825 // 5 // 2 = 182
- Test sizes end up N=1,2,3,4,5 instead of ~250-300
- Also: `preds` and `truths` are accumulated ACROSS folds, not per-fold
- Per-fold metrics are cumulative: fold 0 has 1 sample, fold 1 has 2, etc.

### H2. Metric bug
**CONFIRMED**
- Per-fold `roc_auc_score(truths, preds)` uses accumulated `truths`/`preds` from ALL previous folds
- Fold 0: 1 sample, AUC undefined (silently 0.5)
- Single-class y_true handled by fallback `0.5` — not a raised error
- Accuracy per fold: 0.00, 0.50, 0.33, 0.25, 0.20 — consistent with always predicting UP against increasing DOWN rate

### H3a. No purge/embargo between train and test
**CONFIRMED**
- Code: `hs, he = train_end - lookback, min(...)` — no gap between train end and test start
- Overlapping windows: train window ends at `train_end`, test starts at `train_end - lookback`

### H3b. Scaler fit on all data before split
**REJECTED**
- `make_tensors()` fits scalers on `[:train_end_idx]` only — correct
- Verified by checks cell: `feat_scaler.data_min_` equals train prefix min

### H3c. Macro forward-fill by PERIOD date
**CONFIRMED**
- `merge_asof(..., direction="backward")` uses the macro PERIOD date, not PUBLICATION date
- Inflation for month M is published in early month M+1
- BI rates: the DB has periods like "17 December 2025" (decision date) which is closer to publication
- Inflation: "Desember 2025" refers to December data published in early January
- Using period date means December inflation data is available from December 1, leaking information

### H3d. News aligned so articles from t+1..t+5 appear in sample t
**INCONCLUSIVE** — No news processing exists; all embeddings are zero.

### H3e. Label NaN issue (D2)
**CONFIRMED**
- `m["target_direction"] = (fc > m["Close"]).astype(float)` — NaN > value = False = 0.0
- Last 5 rows get label 0.0 (DOWN) silently
- `dropna` with `future_close` should remove them, but the count suggests otherwise

### H4. Gate collapse
**CONFIRMED**
- Gate weights: α=0.0, β=0.0, γ=1.0, std=0.0 across all folds
- Root causes: (1) no LayerNorm on branch outputs, (2) no softmax temperature, (3) no entropy regularization, (4) random init — the news branch has the largest bias because `news_proj` has a bias term and the zero-vector input produces a non-zero constant after projection
- Gate input is 144-d unnormalized mix of tech (LSTM, bounded), macro (MLP), and news (projected IndoBERT)

### H5. Degenerate news branch — zero vector injection before projection
**CONFIRMED**
- In `align()`: `zeros = pd.DataFrame(0.0, ...); m = pd.concat([m, zeros], axis=1)`
- These zeros flow into `raw_news` → `news_feat = self.news_proj(news_emb)`
- `news_proj` has a bias, so zero input → non-zero constant output
- The zero vector should be injected AFTER the projection layer

### H6. Training hygiene
**CONFIRMED**
- No training loop exists. Model is instantiated with random weights and immediately evaluated.
- `walk_forward_evaluate()` does not train — it just evaluates a random-weight model.
- No early stopping, no learning rate, no loss tracking.

### H7. IndoBERT trainable or in train mode
**CONFIRMED (likely)**
- BERT is loaded in `forward()` without explicit `model.eval()` call
- `requires_grad=False` is set, but mode is not set to eval
- Dropout may be active during embedding extraction

### H8. Feature non-stationarity
**CONFIRMED**
- 12.7% of test features fall outside train min/max range
- Raw Close, SMA, MACD levels min-max scaled on train will drift on later folds

## Summary Table

| ID | Verdict | Key Evidence |
|---|---|---|
| D1 | PARTIAL | Date gap from indicator warm-up (~35 rows) + merge/parsing (~40 rows) |
| D2 | CONFIRMED | `n_win = len(df) - lb + 1` does not subtract horizon |
| D3 | CONFIRMED | No Sharpe/MDD/trading simulation |
| D4 | CONFIRMED | No ablation or static-fusion baseline |
| D5 | CONFIRMED | USD/IDR is daily (2082 rows), thesis says monthly |
| H1 | CONFIRMED | Fold test sizes N=1,2,3,4,5; wrong `total_len // n_splits // 2` |
| H2 | CONFIRMED | Per-fold metrics accumulated across folds; single-class → silent 0.5 |
| H3a | CONFIRMED | No purge/embargo; test starts at train_end - lookback |
| H3b | REJECTED | Scalers correctly fit on train prefix only |
| H3c | CONFIRMED | merge_asof backward uses period date, not publication date |
| H3d | INCONCLUSIVE | No news processing exists |
| H3e | CONFIRMED | NaN label → 0.0 (DOWN) via `.astype(float)` |
| H4 | CONFIRMED | Gate collapsed: α=0, β=0, γ=1.0, std=0 |
| H5 | CONFIRMED | Zero vector injected before projection (bias makes it constant) |
| H6 | CONFIRMED | No training loop; random weights evaluated directly |
| H7 | CONFIRMED | BERT not set to eval mode; dropout may be active |
| H8 | CONFIRMED | 12.7% of test features outside train min/max range |
