"""Supplier portfolio: customer load minus own renewable assets = residual demand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Portfolio:
    """Shares of the national quantities attributable to this supplier.

    Example: ``load_share=0.01`` means the supplier serves 1% of the zone's
    demand; ``wind_share=0.008`` means it owns 0.8% of the zone's wind output.
    """

    load_share: float = 0.01
    wind_share: float = 0.008
    solar_share: float = 0.006

    def __post_init__(self) -> None:
        for name in ("load_share", "wind_share", "solar_share"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def residual_demand(
        self,
        load: NDArray[Any],
        wind: NDArray[Any],
        solar: NDArray[Any],
    ) -> NDArray[np.float64]:
        """MWh to source from the market each hour (negative = surplus to sell)."""
        return (
            self.load_share * np.asarray(load, dtype=np.float64)
            - self.wind_share * np.asarray(wind, dtype=np.float64)
            - self.solar_share * np.asarray(solar, dtype=np.float64)
        )
