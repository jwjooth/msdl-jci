"""Unit tests for technical encoder and pipeline."""

import numpy as np

from msdl_jci.config.settings import get_settings
from msdl_jci.models.technical.encoder import TechnicalDataEncoder
from msdl_jci.models.technical.pipeline import TechnicalPipeline


def test_technical_encoder_fit_and_predict():
    settings = get_settings()
    encoder = TechnicalDataEncoder(look_back=10)
    preds = encoder.fit_and_predict(settings.JCI_HISTORICAL_CSV)

    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 2
    assert preds.shape[1] == 1
    assert len(preds) > 0


def test_technical_pipeline_default_run():
    pipeline = TechnicalPipeline(look_back=10)
    preds = pipeline.build_price_prediction()
    assert isinstance(preds, np.ndarray)
    assert preds.shape[1] == 1
