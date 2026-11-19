from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

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


def predict_cancellation_probability(artifact, input_frame):
    """Predice aplicando también el calibrador congelado, si está presente."""

    raw_probability = np.asarray(
        artifact["pipeline"].predict_proba(input_frame)[:, 1], dtype=float
    )
    calibrator = artifact.get("probability_calibrator")
    if calibrator is None:
        return raw_probability

    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    raw_logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return np.asarray(calibrator.predict_proba(raw_logit)[:, 1], dtype=float)
