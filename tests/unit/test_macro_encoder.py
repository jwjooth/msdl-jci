"""Unit tests for macro data encoder and pipeline."""

import numpy as np

from msdl_jci.config.settings import get_settings
from msdl_jci.models.macro.encoder import MacroDataEncoder
from msdl_jci.models.macro.pipeline import MacroPipeline


def test_macro_encoder_fit_and_encode():
    settings = get_settings()
    encoder = MacroDataEncoder(look_back=settings.ML_LOOK_BACK)

    features = encoder.fit_and_encode()

    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert features.shape[1] == 3
    assert len(features) > 0


def test_macro_pipeline_default_run():
    pipeline = MacroPipeline()
    features = pipeline.build_macro_embeddings()
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 3
