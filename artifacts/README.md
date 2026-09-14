# Artifacts

Large/uncommitted outputs belong here (checkpoints `*.pt`, full-size dumps).
Committed research evidence lives under `reports/` (CSVs, PNGs, Markdown).

- Do not commit `*.pt / *.pth / *.ckpt` (git-ignored; LFS rules in `.gitattributes` if ever needed).
- `reports/` files are intentionally committed: they ARE the thesis evidence.
- Regenerate via `make run-proposed`, `make run-ablation`, or `scripts/*.py`.
