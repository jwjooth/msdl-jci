# Thesis Gaps and Ambiguities

## 1. Fusion Dimension Application
The thesis specifies: Adaptive MLP Fusion input is 144 (64+16+64), output is 3 weights (α, β, γ) via softmax. It does NOT say how these weights are applied to branch outputs of sizes 64 (tech), 16 (macro), and 64 (news).

**Recommended text fix**: "The adaptive MLP fusion generates weighting coefficients (α, β, γ) that are applied after projecting the macroeconomic branch output to the common 64-dimensional space: fused = α·tech_feat + β·macro_proj + γ·news_feat, where macro_proj = Linear(16 → 64)."

## 2. Macro Variable Frequency Inconsistency
Thesis text says: "Monthly macroeconomic variables (BI-7Day Reverse Repo Rate, CPI Inflation %, and USD/IDR exchange rate) are standardized using Z-score normalization." But USD/IDR is daily data in the database.

**Recommended text fix**: "BI-7Day Reverse Repo Rate and CPI Inflation are monthly variables; USD/IDR exchange rate is daily. Macroeconomic variables are forward-filled to daily frequency using the latest officially published figure as of date t, then standardized using Z-score normalization fit on the training set only."

## 3. Non-Stationary Features
The thesis uses raw Close, SMA_20, and MACD levels as input features. These are non-stationary and min-max scaled on train will drift outside [0,1] on later folds. 12.7% of test features fall outside train range.

**Recommended text fix**: "Technical features include stationary transformations: log returns, RSI_14, MACD normalized by SMA_26, ATR_14 normalized by Close, and price relative to SMA_20. Min-Max scaling is fit on the training set only."

## 4. Label Leakage via NaN
Current code: `target_direction = (future_close > Close).astype(float)` evaluates NaN comparisons to False (0.0), silently labeling the last `horizon` rows as DOWN. Should drop those rows explicitly.

**Recommended text fix**: "Targets are computed as Target_t = 1 if Close[t+horizon] > Close[t], else 0. Samples where Close[t+horizon] is unavailable (the final `horizon` observations) are dropped from the dataset."

## 5. Overlapping Labels and Embargo
With lookback=28 and horizon=5, consecutive samples share up to 32 overlapping timesteps. Without a purge/embargo gap, the training and test windows in walk-forward validation overlap in time.

**Recommended text fix**: "A gap of at least lookback + horizon (33 trading days) is inserted between the end of each training window and the start of its corresponding test window to prevent temporal leakage from overlapping features and labels."

## 6. Undefined Training Hyperparameters
The thesis does not specify: optimizer, learning rate, number of epochs, batch size, early stopping patience, or validation split strategy for training.

**Recommended text fix**: "Training uses Adam optimizer (lr=1e-3), batch size 32, max 30 epochs, and early stopping with patience 5 on a validation block carved from the end of each training window (last 10%)."

## 7. IndoBERT Model Identifier
The thesis specifies "Frozen IndoBERT-base" but does not name the specific Hugging Face checkpoint. `csebuetnlp/mubi-bert-base` is gated/private and requires authentication.

**Recommended text fix**: "We use the publicly available `indobenchmark/indobert-base-p1` checkpoint (12-layer, 768-hidden, Indonesian BERT base), loaded in evaluation mode with all parameters frozen."

## 8. Missing Evaluation Baselines
The thesis claims comparison against "relevant baseline model like LSTM model" but does not specify what exact baseline configuration is used (pure technical LSTM, no fusion, no training details).

**Recommended text fix**: "The baseline is a pure LSTM trained on technical features only (same lookback=28, same early stopping, same walk-forward protocol). No macro or news features, no fusion."

## 9. No Fold Size Guarantee
The thesis says "expanding window Walk-Forward mechanism" with 5 splits but does not specify the test fold size or total number of evaluation samples per fold.

**Recommended text fix**: "Expanding-window walk-forward validation with 5 folds. Each test fold contains approximately 260 samples (total_len // (n_splits + 2)), giving a train/test split of roughly 70/30 per fold."
