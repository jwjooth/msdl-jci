"""Application entrypoint for the project."""
from models.technical.pipeline import TechnicalPipeline
from utils.logging_config import configure_logging, get_logger
from models.macro.pipeline import MacroPipeline

logger = get_logger(__name__)


def main() -> None:
    """Run the current macro encoding pipeline."""
    configure_logging()

    macro_pipeline = MacroPipeline(look_back=3)
    macro_features = macro_pipeline.build_macro_embeddings(
        bi_rate_csv="data/raw/bi_rate.csv",
        inflation_csv="data/raw/inflation_data.csv",
        kurs_csv="data/raw/kurs_usdidr.csv",
    )

    logger.info("Macro embedding shape: %s", macro_features.shape)
    logger.info("First 5 rows:\n%s", macro_features[:5])

    technical_pipeline = TechnicalPipeline(look_back=30)
    technical_features = technical_pipeline.build_price_prediction(
        price_csv="data/raw/jci_historical.csv",
    )
    logger.info("Technical embedding shape: %s", technical_features.shape)
    logger.info("First 5 rows:\n%s", technical_features[:5])


if __name__ == "__main__":
    main()
