"""Minimal skeleton for future multivariate bottleneck regression.

This file intentionally contains only a placeholder estimator interface so that
downstream code can import it without introducing behavior yet. Flesh out later
with real feature engineering, forecasting models (e.g., sktime forecasters),
and evaluation utilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:  # Optional dependency (sktime). If absent, provide a tiny stub.
    from sktime.base import BaseEstimator  # type: ignore
except Exception:  # pragma: no cover

    class BaseEstimator:  # Minimal stub
        pass


class BottleneckRegressor(BaseEstimator):
    """Placeholder multivariate model.

    Planned responsibilities:
      * Transform production log to multivariate time series of WIP per step.
      * Fit multivariate forecasting/regression model (sktime strategy).
      * Predict future WIP trajectories and identify likely bottlenecks.

    Current state: purely a skeleton raising NotImplementedError.
    """

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon

    def fit(self, X, y: Optional[Any] = None, **kwargs):  # noqa: D401
        raise NotImplementedError("Implement fit() in future iteration.")

    def predict(self, n_periods: Optional[int] = None):  # noqa: D401
        raise NotImplementedError("Implement predict() in future iteration.")

    def predict_bottleneck(self, n_periods: Optional[int] = None):  # noqa: D401
        raise NotImplementedError(
            "Implement predict_bottleneck() after forecast logic exists."
        )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # noqa: D401
        return {"horizon": self.horizon}

    def set_params(self, **params) -> "BottleneckRegressor":  # noqa: D401
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter {k}")
            setattr(self, k, v)
        return self


__all__ = ["BottleneckRegressor"]
