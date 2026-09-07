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
