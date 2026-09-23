"""Ablation Study and Thesis Experiment Runner for MSDL-JCI.

Executes comparative benchmarks matching Thesis Section 2.6:
1. Pure LSTM (Technical features only)
2. LSTM + Macro (Technical + Macro features)
3. LSTM + News (Technical + News features)
4. LSTM + Static Fusion (Technical + Macro + News without Soft Gating)
5. Proposed Model (Adaptive Soft Gating Fusion)

Produces:
- Classification metrics comparative table (Accuracy, F1-Score, Precision, Recall, ROC-AUC)
- Financial simulation comparative table (Total Return, Sharpe Ratio, Max Drawdown, Win Rate)
- Gating weight distribution analysis (alpha, beta, gamma)
- Evaluation artifact CSVs for thesis defense presentation
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdl_jci.config.settings import get_settings
from msdl_jci.evaluation.walk_forward import EvaluationResults, evaluate_model_walk_forward
from msdl_jci.models.fusion import (
    AdaptiveSoftGatingFusionModel,
    LSTMMacroModel,
    LSTMNewsModel,
    PureLSTMModel,
    StaticFusionModel,
)
from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder
from msdl_jci.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def run_thesis_experiments(
    output_dir: str | Path | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    seed: int = 42,
) -> dict[str, EvaluationResults]:
    """Run full suite of ablation models and produce comparative tables."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    settings = get_settings()
    configure_logging(level=settings.LOG_LEVEL)

    out_path = Path(output_dir) if output_dir else settings.DATA_PROCESSED_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING MSDL-JCI THESIS DEFENSE EXPERIMENTATION SUITE")
    logger.info("=" * 60)

    # 1. Prepare Aligned Multi-Source Dataset (leak-free: scalers fit on train prefix)
    logger.info("Building multi-source dataset and sliding window tensors...")
    builder = MultiSourceDatasetBuilder()
    df_aligned = builder.build_aligned_dataframe()
    _lb = builder.look_back
    _n_win = len(df_aligned) - _lb + 1
    _train_end_df = int(_n_win * 0.70) + _lb - 1
    tensors, _, _ = builder.create_multisource_tensors(df_aligned, train_end_idx=_train_end_df)
    logger.info("Tensors constructed: N=%d, Lookback=%d", len(tensors.y), tensors.X_tech.shape[1])

    # 2. Define Model Configurations for Ablation Study
    _pos_rate = float(np.mean(tensors.y[: int(len(tensors.y) * 0.70)]))
    experiments = [
        ("Pure LSTM (Technical only)", lambda: PureLSTMModel()),
        ("LSTM + Macro", lambda: LSTMMacroModel()),
        ("LSTM + News", lambda: LSTMNewsModel()),
        ("LSTM + Static Fusion", lambda: StaticFusionModel()),
        (
            "Proposed (Adaptive Soft Gating)",
            lambda: AdaptiveSoftGatingFusionModel(
                pos_rate=_pos_rate, modality_dropout=0.05
            ),
        ),
    ]

    results: dict[str, EvaluationResults] = {}
    clf_rows: list[dict] = []
    fin_rows: list[dict] = []

    # 3. Run Walk-Forward Evaluation for Each Model
    for model_name, model_fn in experiments:
        logger.info(">>> Evaluating: %s ...", model_name)
        res = evaluate_model_walk_forward(
            model_fn=model_fn,
            model_name=model_name,
            tensors=tensors,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
        )
        results[model_name] = res

        # Collect classification row
        cd: dict = dict(res.classification_metrics.to_dict())
        cd["Model"] = model_name
        clf_rows.append(cd)

        # Collect financial row
        fd: dict = dict(res.financial_metrics.to_dict())
        fd["Model"] = model_name
        fin_rows.append(fd)

    # 4. Format & Export Comparative Tables
    df_clf = pd.DataFrame(clf_rows)[
        ["Model", "Accuracy", "F1-Score", "Precision", "Recall", "ROC-AUC"]
    ]
    df_fin = pd.DataFrame(fin_rows)[
        ["Model", "Total Return (%)", "Benchmark Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)", "Total Trades"]
    ]

    clf_csv = out_path / "ablation_classification_metrics.csv"
    fin_csv = out_path / "ablation_financial_metrics.csv"
    df_clf.to_csv(clf_csv, index=False)
    df_fin.to_csv(fin_csv, index=False)

    logger.info("=" * 70)
    logger.info("TABLE 1: PREDICTIVE PERFORMANCE EVALUATION (CLASSIFICATION)")
    logger.info("=" * 70)
    print(df_clf.to_string(index=False))

    logger.info("=" * 70)
    logger.info("TABLE 2: FINANCIAL & ECONOMIC PERFORMANCE (TRADING SIMULATION)")
    logger.info("=" * 70)
    print(df_fin.to_string(index=False))

    # 5. Soft Gating Dynamic Weights Analysis
    proposed_res = results.get("Proposed (Adaptive Soft Gating)")
    if proposed_res and proposed_res.gating_weights is not None:
        gw = proposed_res.gating_weights
        mean_alpha = float(np.mean(gw[:, 0]))
        mean_beta = float(np.mean(gw[:, 1]))
        mean_gamma = float(np.mean(gw[:, 2]))

        gating_summary = pd.DataFrame({
            "Modality": ["Technical (LSTM)", "Macroeconomic (MLP)", "Financial News (IndoBERT)"],
            "Weight Parameter": ["alpha", "beta", "gamma"],
            "Mean Weight": [round(mean_alpha, 4), round(mean_beta, 4), round(mean_gamma, 4)],
            "Contribution (%)": [round(mean_alpha * 100, 2), round(mean_beta * 100, 2), round(mean_gamma * 100, 2)],
        })
        gating_csv = out_path / "gating_weights_summary.csv"
        gating_summary.to_csv(gating_csv, index=False)

        logger.info("=" * 70)
        logger.info("TABLE 3: ADAPTIVE SOFT GATING WEIGHT ALLOCATION")
        logger.info("=" * 70)
        print(gating_summary.to_string(index=False))

        # Save trajectory of gating weights
        df_gw = pd.DataFrame(gw, columns=["alpha_tech", "beta_macro", "gamma_news"])
        df_gw.insert(0, "Date", proposed_res.test_dates)
        df_gw.to_csv(out_path / "gating_weights_trajectory.csv", index=False)

    # 6. Generate Financial Equity Curve Comparison Plot
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, res in results.items():
            ax.plot(res.financial_metrics.equity_curve, label=name)
        # Benchmark curve from any result
        any_res = next(iter(results.values()))
        ax.plot(any_res.financial_metrics.benchmark_curve, label="Buy-and-Hold JCI", linestyle="--", color="black", alpha=0.7)
        ax.set_title("MSDL-JCI: Cumulative Portfolio Returns across Ablation Models")
        ax.set_xlabel("Rebalancing Trade Steps (5-Day Horizon)")
        ax.set_ylabel("Portfolio Value (Base=100)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        chart_path = out_path / "cumulative_returns_comparison.png"
        fig.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Cumulative return chart saved to %s", chart_path)
    except Exception as e:
        logger.warning("Could not generate matplotlib chart: %s", e)

    logger.info("All thesis ablation experiments completed successfully!")
    return results


def main(argv=None) -> int:
    """CLI entrypoint for `msdl-jci-ablation`."""
    import argparse

    ap = argparse.ArgumentParser(description="MSDL-JCI ablation suite")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args(argv)
    run_thesis_experiments(output_dir=args.output_dir, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
