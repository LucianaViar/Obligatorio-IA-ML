"""Carga compatible del artefacto congelado de Etapa 5."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

from src.booking_transformers import HotelBookingFeatureEngineer, QuantileClipper


def load_frozen_artifact(path: str | Path):
    """Carga el artefacto sin ejecutar el notebook.

    El archivo original fue serializado desde un notebook y, por ello, pickle
    registró las dos clases personalizadas bajo ``__main__``. Se publican las
    mismas definiciones en ese módulo antes de llamar a joblib. El artefacto no
    se reescribe ni se modifica.
    """

    main_module = sys.modules["__main__"]
    setattr(main_module, "HotelBookingFeatureEngineer", HotelBookingFeatureEngineer)
    setattr(main_module, "QuantileClipper", QuantileClipper)
    return joblib.load(Path(path))
