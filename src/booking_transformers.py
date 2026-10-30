"""Transformaciones personalizadas usadas por el pipeline congelado.

Estas clases reproducen las definiciones con las que se entrenó el artefacto.
No aprenden información fuera de ``fit`` ni consultan el objetivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HotelBookingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea las ocho variables de Etapa 3 a partir de una reserva."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        data["total_guests"] = data[["adults", "children", "babies"]].sum(
            axis=1, min_count=3
        )
        data["total_nights"] = (
            data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
        )
        data["weekend_ratio"] = data["stays_in_weekend_nights"].div(
            data["total_nights"].where(data["total_nights"] > 0)
        )

        children_or_babies = (data["children"].fillna(0) + data["babies"]) > 0
        data["has_children"] = children_or_babies.astype(float)
        unknown_children = data["children"].isna() & data["babies"].eq(0)
        data.loc[unknown_children, "has_children"] = np.nan

        data["lead_time_category"] = pd.cut(
            data["lead_time"],
            bins=[-np.inf, 7, 30, 90, 180, 365, np.inf],
            labels=["0-7", "8-30", "31-90", "91-180", "181-365", ">365"],
        ).astype("object")

        data["total_previous_bookings"] = (
            data["previous_cancellations"]
            + data["previous_bookings_not_canceled"]
        )
        data["previous_cancellation_rate"] = np.divide(
            data["previous_cancellations"],
            data["total_previous_bookings"],
            out=np.zeros(len(data), dtype=float),
            where=data["total_previous_bookings"].to_numpy() > 0,
        )

        season_by_month = {
            "December": "Winter",
            "January": "Winter",
            "February": "Winter",
            "March": "Spring",
            "April": "Spring",
            "May": "Spring",
            "June": "Summer",
            "July": "Summer",
            "August": "Summer",
            "September": "Autumn",
            "October": "Autumn",
            "November": "Autumn",
        }
        data["arrival_season"] = data["arrival_date_month"].map(season_by_month)
        return data


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Recorta valores a cuantiles aprendidos durante ``fit``."""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.quantile(values, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(values, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)
