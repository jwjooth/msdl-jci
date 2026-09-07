"""Financial news pipeline orchestration."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from msdl_jci.config.settings import get_settings
from msdl_jci.models.news.encoder import NewsDataEncoder


class NewsPipeline:
    """Orchestration pipeline for news feature prediction."""

    def __init__(self, look_back: int = 30) -> None:
        self.encoder = NewsDataEncoder(look_back=look_back)

    def build_news_prediction(
        self,
        news_csv: Optional[Union[str, Path]] = None,
        feature_column: str = "mean_embedding",
    ) -> NDArray[np.float64]:
        """Train model on news embeddings and return predictions."""
        settings = get_settings()
        target_path = (
            news_csv
            if news_csv is not None
            else settings.DATA_PROCESSED_DIR / "daily_news_embeddings.csv"
        )
        return self.encoder.fit_and_predict(target_path, feature_column=feature_column)
