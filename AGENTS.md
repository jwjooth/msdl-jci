# MSDL-JCI

Thesis repo: predict JCI direction (t+5) via **LSTM (technical) + MLP (macro) + frozen IndoBERT (news) + soft-gating fusion**. Not for live trading. Journal publication target — Ch.3 revision tracked in issue #47.

The whole project is **one notebook**: `notebooks/development.ipynb` (config → utils → model → run → viz → checks), plus read-only `scripts/diagnose/` helpers and `tests/`.

## Commands

- Setup: `uv sync --extra dev` (`uv.lock` is source of truth; Python 3.12; ruff is the only dev dep).
- Kernel (once): `uv run python -m ipykernel install --user --name python3`.
- Run: `uv run jupyter execute notebooks/development.ipynb --inplace`. First run downloads `indobenchmark/indobert-base-p1` and writes `database/news_emb_cache.npz`.
- Lint: `uv run ruff check .` — must be clean. `ruff format` NOT enforced.
- Tests: `python tests/test_stats.py` — AST-extracts `trading_stats`/`paired_stats`/`youden_threshold`/`attach_news` from notebook cell 6 and checks edge cases. No pytest needed.
- End-to-end verification: the notebook's `## Checks` cell (train-prefix scaler proof, shapes, finiteness).

## Gotchas

- `Config` cell owns all hyperparameters (thesis Table 5): lookback 28, horizon 5, seed 42, lstm 64, proj 64, MLP 3→32→16, fusion Linear(144→3)+softmax. `DB_PATH` reads `MSDL_DB` env var (default: `database/main_database.db`).
- All SQLite reads go through `read_table(name, con=None)`, which enforces `ENTITY_TABLES` schema. Pass a shared connection from `load_frames()` to avoid reopening.
- News: frozen `indobenchmark/indobert-base-p1` CLS, cached to `database/news_emb_cache.npz`. Attached past-only (strictly-before-date; Sat/Sun → Monday). Never cite `csebuetnlp/mubi-bert-base` (nonexistent) — see issue #37.
- `database/main_database.db`, `database/news_emb_cache.npz`, `database/wf_artifacts.npz` ARE tracked in git despite `.gitignore` listing `database/`. Do not delete; `.gitignore` only prevents new accidental adds.
- Missing DB → notebook runs a synthetic seed-42 fallback; results are meaningless.
- Single-layer LSTM dropout=0.2 emits a harmless `dropout/num_layers=1` UserWarning every run — by design.
- `attach_news` uses vectorized searchsorted + cumsum (O(N+M)); acceptable at current scale.
- `news_embeddings` downloads IndoBERT on first cache miss; fails offline without pre-download. Set `TRANSFORMERS_OFFLINE=1` in CI.
- `trading_stats` buy-hold uses daily compounding (same basis as strategy); both Sharpe ratios are directly comparable.
- `classical_baseline` scales tech+macro together and news separately on train prefix — now comparable to deep path (news_proj frozen Linear 768→64).
- `paired_stats` uses `scipy.stats.t.ppf(0.975, df=n-1)` (12.706 fallback for n=1); works any fold count. CI covers zero on identical folds (tested).

## Do not "fix" — leak-free methodology

- Order: align → train-only scaling (70% prefix, `train_end_idx`) → window → t+5 labels (`future_close > Close`). Scalers must never see the test prefix.
- Macro alignment is ffill-only with leading NaNs dropped. Never `bfill` (injects future macro values).
- Result is an honest weak signal (pooled AUC ≈ 0.50 post-rescrape, no ablation/trading edge, news gate γ ≈ 0.09). The checks cell passes — do not "improve" metrics by loosening the above.
- No tag `v1.0-bab4` present locally; verify via `git tag` upstream before quoting Bab-4 numbers.

## Closed issues (#67–#77 all resolved)

- #67 seed Python `random`; #68 shared SQLite connection; #69 `train_split_ratio` in Config; #70 single head w/ zero-masking; #71 `n_splits`/`max_epochs`/`patience`/`lr` in Config; #72 dynamic `_EMB_DIM`; #73 corpus audit (detik-primary early, not exclusive); #74 full IndoBERT spec in summary cell; #75 per-fold Youden-J threshold reporting; #76 §4.x EMH doc guidance; #77 LFS/docx repo-hygiene doc guidance.

## Issue guidance (docx/thesis)

- #39: §4.x EMH discussion (1.5 pp max, Fama 1970, "fail to reject" not "prove"); fallback = practical-implications paragraph.
- #43: thesis close-out checklist (docx add/remove/change) — remaining doc work; closes #36–#39.
- #44: LFS deferred post-defense; track `Project English - Development.docx` as source of truth or keep local — not both half-done.
