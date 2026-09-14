"""MSDL-JCI: Multi-Source Deep Learning Framework for JCI Forecasting."""

from msdl_jci.config.settings import Settings, get_settings
from msdl_jci.main import main

__version__ = "0.1.0"
__all__ = ["Settings", "get_settings", "main", "__version__"]
