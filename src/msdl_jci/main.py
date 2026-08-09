"""Application entrypoint for the project."""

from msdl_jci.core.logging import configure_logging, get_logger
from msdl_jci.services.macro_service import MacroEncodingService

logger = get_logger(__name__)


def main() -> None:
    """Run the current macro encoding pipeline."""
    configure_logging()

    service = MacroEncodingService(look_back=3)
    embeddings = service.build_macro_embeddings(
        bi_rate_csv="data/raw/bi_rate.csv",
        inflation_csv="data/raw/inflation_data.csv",
        kurs_csv="data/raw/kurs_usdidr.csv",
    )

    logger.info("Macro embedding shape: %s", embeddings.shape)
    logger.info("First 5 rows:\n%s", embeddings[:5])


if __name__ == "__main__":
    main()
