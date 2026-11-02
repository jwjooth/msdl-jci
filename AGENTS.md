# MSDL-JCI

Thesis repo: predict JCI direction (t+5) via **LSTM (technical) + MLP (macro) + frozen IndoBERT (news) + soft-gating fusion**. Not for live trading. **End goal is journal publication, not defense alone** — Ch.3 revision tracked in issue #47.

The whole project is **one notebook**: `notebooks/development.ipynb` (config → utils → model → run → viz → checks), plus read-only `scripts/diagnose/` helpers and `tests/`.

## Commands

- Setup: `uv sync --extra dev` (`uv.lock` is source of truth; Python 3.12; ruff is the only dev dep).
- Run: `uv run jupyter execute notebooks/development.ipynb --inplace` (needs `python3` kernelspec: `uv run python -m ipykernel install --user --name python3`). First run downloads `indobenchmark/indobert-base-p1` and writes `database/news_emb_cache.npz`.
- Lint: `uv run ruff check .` — clean. `ruff format` is NOT enforced.
- Tests: `python tests/test_stats.py` (AST-extracts `trading_stats`/`paired_stats`/`youden_threshold` from the notebook and checks their edge cases). The notebook's `## Checks` cell (train-prefix scaler proof, shapes, finiteness) remains the end-to-end verification step.

## Gotchas

- `Config` cell owns all hyperparameters (thesis Table 5): lookback 28, horizon 5, seed 42, lstm 64, proj 64, MLP 3→32→16, fusion Linear(144→3)+softmax. `DB_PATH` reads `MSDL_DB` env var (default: `database/main_database.db`).
- All SQLite reads go through `read_table(name, con=None)`, which enforces `ENTITY_TABLES` schema. Pass a shared connection from `load_frames()` to avoid reopening.
- News: frozen `indobenchmark/indobert-base-p1` CLS, cached to `database/news_emb_cache.npz`. Attached past-only (strictly-before-date; Sat/Sun → Monday). Archive re-scrape (2026-10-09): detik 4953 + cnbc 5556 (2019–2025) + kontan 10275 (2018–2025) all usable. Never cite `csebuetnlp/mubi-bert-base` (nonexistent) — see issue #37.
- Missing DB → notebook runs a synthetic seed-42 fallback; results from fallback runs are meaningless.
- `.gitignore` excludes `database/` and `*.npz` — large binaries are NOT tracked. Check `git status` before committing.
- Single-layer LSTM with dropout=0.2 emits a `dropout/num_layers=1` UserWarning every run — harmless, by design.
- `attach_news` uses a Python loop (O(N×M)); acceptable at current scale but will hurt if data grows.
- `news_embeddings` downloads IndoBERT on first cache miss; fails offline without pre-download. Set `TRANSFORMERS_OFFLINE=1` in CI.
- `trading_stats` buy-hold uses overlapping 5d returns (overstates wealth ~3.5×). Strategy uses daily compounding. Apples-to-oranges in reported Sharpe.
- `classical_baseline` scales news embeddings with tech+macro — inconsistent with deep path (news proj is frozen). Baselines not directly comparable.
- `paired_stats` uses `scipy.stats.t.ppf(0.975, df=n-1)` (12.706 fallback for n=1); works for any fold count. CI covers zero on identical folds (edge case tested).

## Do not "fix" — leak-free methodology

- Order: align → train-only scaling (70% prefix, `train_end_idx`) → window → t+5 labels (`future_close > Close`). Scalers must never see the test prefix.
- Macro alignment is ffill-only with leading NaNs dropped. Never `bfill` (injects future macro values).
- Result is an honest weak signal (pooled AUC ≈ 0.50 post-rescrape, no ablation/trading edge, news gate γ ≈ 0.09). The checks cell passes — do not "improve" metrics by loosening the above.
- `v1.0-bab4` is the frozen thesis-Ch.4 run; later E1–E3/optimization cells sit on top — never quote post-tag numbers as Bab-4 results. (Tag not present in this clone; verify via `git tag` upstream.)

## Open thesis issues

- #36: §2.1 corpus claim wrong — only detik informs training; re-scrape archive or rewrite as detik-primary.
- #37: Table 5 must carry full IndoBERT ID + spec (12L/12H/768, frozen CLS, mean-pool, 768→64).
- #38: report Youden-J thresholds + paired ΔAUC CIs; focal/grid were negative (keep BCE, lookback 28 / LR 1e-3).
- #39: §4.x EMH discussion (1.5 pp max) + practical-implication fallback; both drafts live as an issue comment.
- #43: thesis close-out checklist (add/remove/change in docx) — the remaining doc work; close #36–#39 through it.
- #44: repo hygiene (LFS decision + docx tracking). #40 closed (lab side done, pushed). #41/#42 are post-defense only.