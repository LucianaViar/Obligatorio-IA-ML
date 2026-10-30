"""Carga el modelo congelado y predice una reserva sin ejecutar el notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_io import load_frozen_artifact


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_cancel_pipeline.joblib"
DEFAULT_INPUT = PROJECT_ROOT / "examples" / "new_reservation.json"


def load_reservation(path: Path) -> dict:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError("El archivo de entrada debe contener un objeto JSON.")
    return payload


def predict_reservation(artifact: dict, reservation: dict) -> dict:
    required_features = artifact["base_model_features"]
    missing = [feature for feature in required_features if feature not in reservation]
    if missing:
        raise ValueError(f"Faltan variables requeridas: {', '.join(missing)}")

    # Las claves adicionales se ignoran y JSON null representa un valor faltante.
    input_frame = pd.DataFrame(
        [{
            feature: np.nan if reservation[feature] is None else reservation[feature]
            for feature in required_features
        }],
        columns=required_features,
    )

    probability = float(artifact["pipeline"].predict_proba(input_frame)[0, 1])
    threshold = float(artifact["threshold"])
    return {
        "probabilidad_cancelacion": probability,
        "umbral": threshold,
        "prediccion_cancelacion": int(probability >= threshold),
        "configuracion": artifact["config_id"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Reserva en JSON (por defecto: examples/new_reservation.json).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Artefacto joblib congelado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_frozen_artifact(args.model)
    result = predict_reservation(artifact, load_reservation(args.input))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
