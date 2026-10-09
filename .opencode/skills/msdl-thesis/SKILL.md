---
name: msdl-thesis
description: Write or revise thesis text (Bab 1–4) for the MSDL-JCI undergraduate thesis. Use when drafting, translating, or framing results for the defense document.
---

# MSDL Thesis — writing guardrails

Source doc: `Project English - Development.docx`. Status: Ch.1–2 drafted, Results/Discussion still TODO placeholders — notebook numbers feed Ch.4, not the reverse.

## RQs (every claim must map to one)

1. MLP macro extractor vs direct input (acc/F1)?
2. Frozen IndoBERT vs pure-technical baseline (UP/DOWN acc)?
3. Soft-gating fusion vs static/baseline (Sharpe, stability)?

## Safe language for the negative result

- Always "gagal menolak" / "konsisten dengan", never "membuktikan".
- Frame: AUC ≈ chance + paired ΔAUC CIs covering 0 + Sharpe ≈ buy-hold = evidence *consistent with* semi-strong EMH on JCI t+5 (issue #39).
- Scope limiters always attached: horizon t+5 only, 2018–2025, no transaction costs.

## Open thesis issues (check `gh issue list` before writing)

- #36: §2.1 corpus — only detik informs training; CNBC/Kontan are 2026-dated.
- #37: Table 5 needs full `indobenchmark/indobert-base-p1` spec; never cite `csebuetnlp/mubi-bert-base` (nonexistent).
- #38: report Youden-J thresholds + paired ΔAUC CIs.
- #39: EMH discussion drafts (both options) live as an issue comment.
