# Assumptions

## Fusion Gate Application (Thesis Gap)
The thesis specifies: Adaptive MLP Fusion input 144 = 64 + 16 + 64, output 3 weights (α, β, γ) via softmax. It does NOT specify how these weights are applied to branches of size 64/16/64.

**Our implementation**: Project macro branch 16→64 with `macro_proj`, then fused = α*tech(64) + β*macro_proj(64) + γ*news(64). Documented here as a thesis gap.

## Date Range (D1)
JCI data starts 2018-01-02, but `align()` drops ~80 early rows due to indicator warm-up (SMA_20 needs 20, MACD needs 35, ATR_14 needs 14 periods). After alignment, data starts 2018-04-19. This is justified by the indicator warm-up period.

## USD/IDR Frequency (D5)
The thesis text calls all macro variables "monthly." However, `kurs_usdidr` table has daily data (2082 rows). We keep USD/IDR as daily because the source is daily. The thesis likely means BI rate and inflation are monthly, while USD/IDR is daily.

## Zero-Vector News Injection (H5)
Thesis says: "Trading days with no news get a 64-dimensional vector of 0.0". We inject zeros at the 64-d projection output level (after news_proj), not before. When news_ids is all-zeros, StockModel returns a 64-d zero vector directly, bypassing the projection bias.

## Macro Forward-Fill (H3c)
Thesis says: "Monthly macroeconomic indicators are expanded to daily frequency by carrying forward the latest officially published figure up to date t." We use merge_asof with direction="backward" on period dates. Inflation for month M is published in early M+1, so this leaks. FIX: not yet applied — see diagnosis.md H3c.

## Sample Count Formula (D2)
`n_win = len(df) - lb - horizon + 1` (was `len(df) - lb + 1`). Fixed in commit.

## Walk-Forward Fold Sizes (H1)
Fixed: expanding window with per-fold test size ~260 (total_len // (n_splits+2)), purge gap = lookback + horizon = 33.
