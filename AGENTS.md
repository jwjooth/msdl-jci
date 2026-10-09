# MSDL-JCI

Thesis repo: predict JCI direction (t+5) via **LSTM (technical) + MLP (macro) + frozen IndoBERT (news) + soft-gating fusion**. Not for live trading.

The whole project is **one notebook**: `notebooks/development.ipynb` (config → utils → model → run → viz → checks). No `src/`, tests, or scripts.

## Commands

- Setup: `uv sync --extra dev` (`uv.lock` is source of truth; Python 3.12; ruff is the only dev dep).
- Run: `uv run jupyter execute notebooks/development.ipynb --inplace` (needs `python3` kernelspec: `uv run python -m ipykernel install --user --name python3`). First run downloads `indobenchmark/indobert-base-p1` and writes `database/news_emb_cache.npz`.
- Lint: `uv run ruff check .` — clean. `ruff format` is NOT enforced.
- No test suite; the notebook's `## Checks` cell (train-prefix scaler proof, shapes, finiteness) is the verification step.

## Gotchas

- `Config` cell owns all hyperparameters (thesis Table 5): lookback 28, horizon 5, seed 42, lstm 64, proj 64, MLP 3→32→16, fusion Linear(144→3)+softmax. `DB_PATH` is hardcoded to `database/main_database.db` — no env vars.
- All SQLite reads go through `read_table`, which enforces `ENTITY_TABLES` schema: `jci_historical`, `bi_rate`, `inflation_data`, `kurs_usdidr` + `cnbc/detik/kontan_ihsg_articles`. Keep it as the single reader.
- News: frozen `indobenchmark/indobert-base-p1` CLS, cached to `database/news_emb_cache.npz`. Attached past-only (strictly-before-date; Sat/Sun → Monday). Archive re-scrape (2026-10-09, merged from `database/new_database.db` into `main_database.db`): detik 4953 + cnbc 5556 (2019–2025) + kontan 10275 (2018–2025) all usable — all 3 portals now inform training. Never cite `csebuetnlp/mubi-bert-base` (nonexistent) — see issue #37.
- Missing DB → notebook runs a synthetic seed-42 fallback; results from fallback runs are meaningless.
- Current `database/` + `.gitignore`: `.gitignore` does NOT list `database/` or `*.db`, so the DB/cache are committable — check `git status` before committing large binaries.
- Single-layer LSTM with dropout=0.2 emits a `dropout/num_layers=1` UserWarning every run — harmless, by design.

## Do not "fix" — leak-free methodology

- Order: align → train-only scaling (70% prefix, `train_end_idx`) → window → t+5 labels (`future_close > Close`). Scalers must never see the test prefix.
- Macro alignment is ffill-only with leading NaNs dropped. Never `bfill` (injects future macro values).
- Result is an honest weak signal (pooled AUC ≈ 0.50 post-rescrape, no ablation/trading edge, news gate γ ≈ 0.03). The checks cell passes — do not "improve" metrics by loosening the above.
- `v1.0-bab4` is the frozen thesis-Ch.4 run; later E1–E3/optimization cells sit on top — never quote post-tag numbers as Bab-4 results. (Note: tag not present in this clone; verify via `git tag` upstream.)

## Open thesis issues

- #36: §2.1 corpus claim wrong — only detik informs training; re-scrape archive or rewrite as detik-primary.
- #37: Table 5 must carry full IndoBERT ID + spec (12L/12H/768, frozen CLS, mean-pool, 768→64).
- #38: report Youden-J thresholds + paired ΔAUC CIs; focal/grid were negative (keep BCE, lookback 28 / LR 1e-3).
- #39: §4.x EMH discussion (1.5 pp max) + practical-implication fallback; both drafts live as an issue comment.
- #40: defense checklist — frozen numbers table + ordered tasks (do this first).
