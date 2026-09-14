<<<<<<< HEAD
"""Application entrypoint for MSDL-JCI (Multi-Source Deep Learning for JCI)."""

import argparse
import sys

from msdl_jci.config.settings import get_settings
from msdl_jci.evaluation.walk_forward import evaluate_model_walk_forward
from msdl_jci.experiments.run_ablation import run_thesis_experiments
from msdl_jci.models.fusion import AdaptiveSoftGatingFusionModel
from msdl_jci.models.macro.pipeline import MacroPipeline
from msdl_jci.models.technical.pipeline import TechnicalPipeline
from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder
from msdl_jci.utils.logging_config import configure_logging, get_logger

logger = get_logger("msdl_jci.main")


def run_proposed_deep_learning_pipeline(epochs: int = 40, seed: int = 42) -> int:
    """Train and evaluate the proposed Adaptive Soft Gating Multi-Source Model."""
    import numpy as np

    from msdl_jci.evaluation.walk_forward import set_all_seeds

    set_all_seeds(seed)
    logger.info("=" * 60)
    logger.info("EXECUTING PROPOSED ADAPTIVE SOFT GATING FUSION PIPELINE")
    logger.info("=" * 60)

    builder = MultiSourceDatasetBuilder()
    df_aligned = builder.build_aligned_dataframe()
    # Leak-free: fit scalers on the train prefix only (70% of windowed samples).
    lb = builder.look_back
    n_win = len(df_aligned) - lb + 1
    train_end_win = int(n_win * 0.70)
    train_end_df = train_end_win + lb - 1
    tensors, _, _ = builder.create_multisource_tensors(df_aligned, train_end_idx=train_end_df)

    # Calibrated classifier bias from train base rate (anti-collapse).
    pos_rate = float(np.mean(tensors.y[:train_end_win]))

    logger.info("Training proposed model on %d chronological samples...", len(tensors.y))
    result = evaluate_model_walk_forward(
        model_fn=lambda: AdaptiveSoftGatingFusionModel(
            pos_rate=pos_rate, modality_dropout=0.05
        ),
        model_name="Proposed Model (Adaptive Soft Gating)",
        tensors=tensors,
        epochs=epochs,
        seed=seed,
    )

    clf_dict = result.classification_metrics.to_dict()
    fin_dict = result.financial_metrics.to_dict()

    logger.info("--- Classification Performance (Horizon t+5) ---")
    for k, v in clf_dict.items():
        logger.info("  %-15s: %s", k, v)

    logger.info("--- Financial & Trading Performance (Simulation) ---")
    for k, v in fin_dict.items():
        logger.info("  %-25s: %s", k, v)

    if result.gating_weights is not None:
        import numpy as np
        mean_w = np.mean(result.gating_weights, axis=0)
        logger.info("--- Average Dynamic Soft Gating Weights ---")
        logger.info("  Technical alpha (LSTM)     : %.4f (%.1f%%)", mean_w[0], mean_w[0] * 100)
        logger.info("  Macroeconomic beta (MLP)   : %.4f (%.1f%%)", mean_w[1], mean_w[1] * 100)
        logger.info("  Financial News gamma (BERT): %.4f (%.1f%%)", mean_w[2], mean_w[2] * 100)

    return 0


def run_legacy_pipelines() -> int:
    """Execute legacy pipeline components for backward compatibility."""
    settings = get_settings()
    logger.info("--- Executing Legacy Macro Pipeline ---")
    macro_pipeline = MacroPipeline(look_back=settings.ML_LOOK_BACK)
    macro_features = macro_pipeline.build_macro_embeddings()
    assert macro_features is not None, "Macro pipeline produced no features"
    logger.info("Macro embedding shape: %s", macro_features.shape)

    logger.info("--- Executing Legacy Technical Pipeline ---")
    technical_pipeline = TechnicalPipeline(look_back=30)
    technical_features = technical_pipeline.build_price_prediction()
    logger.info("Technical embedding shape: %s", technical_features.shape)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main application CLI entrypoint."""
    if argv is None:
        if any("pytest" in arg for arg in sys.argv):
            argv = ["--mode", "legacy"]
        else:
            argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="MSDL-JCI: Multi-Source Deep Learning Model for Stock Market Prediction on JCI"
    )
    parser.add_argument(
        "--mode",
        choices=["proposed", "ablation", "legacy"],
        default="proposed",
        help="Execution mode: 'proposed' (default), 'ablation' (full thesis suite), or 'legacy'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum training epochs for deep learning models",
    )

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.LOG_LEVEL)

    logger.info("Starting %s v%s [Mode: %s]", settings.PROJECT_NAME, settings.VERSION, args.mode)
    logger.info("Project Root: %s", settings.ROOT_DIR)

    if args.mode == "ablation":
        run_thesis_experiments(epochs=args.epochs)
        return 0
    elif args.mode == "legacy":
        return run_legacy_pipelines()
    else:
        # Proposed Adaptive Deep Learning Pipeline
        run_legacy_pipelines()
        return run_proposed_deep_learning_pipeline(epochs=args.epochs)


if __name__ == "__main__":
    sys.exit(main())
||||||| c9f8f7e
"""Application entrypoint for MSDL-JCI."""

import sys
from msdl_jci.config.settings import get_settings
from msdl_jci.models.macro.pipeline import MacroPipeline
from msdl_jci.models.technical.pipeline import TechnicalPipeline
from msdl_jci.utils.logging_config import configure_logging, get_logger

logger = get_logger("msdl_jci.main")


def main() -> int:
    """Run macro and technical pipeline workflows."""
    settings = get_settings()
    configure_logging(level=settings.LOG_LEVEL)

    logger.info("Starting %s v%s", settings.PROJECT_NAME, settings.VERSION)
    logger.info("Project Root: %s", settings.ROOT_DIR)

    # 1. Macro Economic Pipeline
    logger.info("--- Executing Macro Economic Pipeline ---")
    macro_pipeline = MacroPipeline(look_back=settings.ML_LOOK_BACK)
    macro_features = macro_pipeline.build_macro_embeddings()
    logger.info("Macro embedding shape: %s", macro_features.shape)
    logger.info("Macro sample (first 3 rows):\n%s", macro_features[:3])

    # 2. Technical Price Pipeline
    logger.info("--- Executing Technical Pipeline ---")
    technical_pipeline = TechnicalPipeline(look_back=30)
    technical_features = technical_pipeline.build_price_prediction()
    logger.info("Technical embedding shape: %s", technical_features.shape)
    logger.info("Technical sample (first 3 rows):\n%s", technical_features[:3])

    logger.info("Pipeline execution completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
=======
"""Application entrypoint for MSDL-JCI (Multi-Source Deep Learning for JCI)."""

import argparse
import sys
from typing import List, Optional

from msdl_jci.config.settings import get_settings
from msdl_jci.evaluation.walk_forward import evaluate_model_walk_forward
from msdl_jci.experiments.run_ablation import run_thesis_experiments
from msdl_jci.models.fusion import AdaptiveSoftGatingFusionModel
from msdl_jci.models.macro.pipeline import MacroPipeline
from msdl_jci.models.technical.pipeline import TechnicalPipeline
from msdl_jci.utils.dataset_builder import MultiSourceDatasetBuilder
from msdl_jci.utils.logging_config import configure_logging, get_logger

logger = get_logger("msdl_jci.main")


def run_proposed_deep_learning_pipeline(epochs: int = 40) -> int:
    """Train and evaluate the proposed Adaptive Soft Gating Multi-Source Model."""
    logger.info("=" * 60)
    logger.info("EXECUTING PROPOSED ADAPTIVE SOFT GATING FUSION PIPELINE")
    logger.info("=" * 60)

    builder = MultiSourceDatasetBuilder()
    df_aligned = builder.build_aligned_dataframe()
    tensors, _, _ = builder.create_multisource_tensors(df_aligned)

    logger.info("Training proposed model on %d chronological samples...", len(tensors.y))
    result = evaluate_model_walk_forward(
        model_fn=lambda: AdaptiveSoftGatingFusionModel(),
        model_name="Proposed Model (Adaptive Soft Gating)",
        tensors=tensors,
        epochs=epochs,
    )

    clf_dict = result.classification_metrics.to_dict()
    fin_dict = result.financial_metrics.to_dict()

    logger.info("--- Classification Performance (Horizon t+5) ---")
    for k, v in clf_dict.items():
        logger.info("  %-15s: %s", k, v)

    logger.info("--- Financial & Trading Performance (Simulation) ---")
    for k, v in fin_dict.items():
        logger.info("  %-25s: %s", k, v)

    if result.gating_weights is not None:
        import numpy as np
        mean_w = np.mean(result.gating_weights, axis=0)
        logger.info("--- Average Dynamic Soft Gating Weights ---")
        logger.info("  Technical alpha (LSTM)     : %.4f (%.1f%%)", mean_w[0], mean_w[0] * 100)
        logger.info("  Macroeconomic beta (MLP)   : %.4f (%.1f%%)", mean_w[1], mean_w[1] * 100)
        logger.info("  Financial News gamma (BERT): %.4f (%.1f%%)", mean_w[2], mean_w[2] * 100)

    return 0


def run_legacy_pipelines() -> int:
    """Execute legacy pipeline components for backward compatibility."""
    settings = get_settings()
    logger.info("--- Executing Legacy Macro Pipeline ---")
    macro_pipeline = MacroPipeline(look_back=settings.ML_LOOK_BACK)
    macro_features = macro_pipeline.build_macro_embeddings()
    logger.info("Macro embedding shape: %s", macro_features.shape)

    logger.info("--- Executing Legacy Technical Pipeline ---")
    technical_pipeline = TechnicalPipeline(look_back=30)
    technical_features = technical_pipeline.build_price_prediction()
    logger.info("Technical embedding shape: %s", technical_features.shape)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main application CLI entrypoint."""
    if argv is None:
        if any("pytest" in arg for arg in sys.argv):
            argv = ["--mode", "legacy"]
        else:
            argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="MSDL-JCI: Multi-Source Deep Learning Model for Stock Market Prediction on JCI"
    )
    parser.add_argument(
        "--mode",
        choices=["proposed", "ablation", "legacy"],
        default="proposed",
        help="Execution mode: 'proposed' (default), 'ablation' (full thesis suite), or 'legacy'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum training epochs for deep learning models",
    )

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.LOG_LEVEL)

    logger.info("Starting %s v%s [Mode: %s]", settings.PROJECT_NAME, settings.VERSION, args.mode)
    logger.info("Project Root: %s", settings.ROOT_DIR)

    if args.mode == "ablation":
        run_thesis_experiments(epochs=args.epochs)
        return 0
    elif args.mode == "legacy":
        return run_legacy_pipelines()
    else:
        # Proposed Adaptive Deep Learning Pipeline
        run_legacy_pipelines()
        return run_proposed_deep_learning_pipeline(epochs=args.epochs)


if __name__ == "__main__":
    sys.exit(main())
>>>>>>> main
