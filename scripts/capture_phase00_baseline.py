"""Phase 0 baseline capture: short-epoch proposed run + ablation (CPU-friendly).

Full 40-epoch runs are documented from the reported training log; this script
reproduces the pipeline end-to-end at small epoch counts to freeze behavior.
Saves logs + metrics to reports/phase_00_baseline/.
"""

import io
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "reports" / "phase_00_baseline"
OUT.mkdir(parents=True, exist_ok=True)


def _capture(fn, log_path):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        result = fn()
    finally:
        root.removeHandler(handler)
    Path(log_path).write_text(buf.getvalue())
    return result


def main():
    from msdl_jci.main import run_proposed_deep_learning_pipeline

    _capture(lambda: run_proposed_deep_learning_pipeline(epochs=3, seed=42),
             OUT / "proposed_main_log.txt")

    from msdl_jci.experiments.run_ablation import run_thesis_experiments

    results = _capture(
        lambda: run_thesis_experiments(
            output_dir=OUT / "ablation_out", epochs=2, seed=42),
        OUT / "ablation_log.txt")

    summary = {}
    for name, res in results.items():
        summary[name] = {
            "classification": res.classification_metrics.to_dict(),
            "trading": res.financial_metrics.to_dict(),
            "best_threshold": res.best_threshold,
            "prob_std": float(res.predictions.std()),
        }
    (OUT / "reproduced_short_run_metrics.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
