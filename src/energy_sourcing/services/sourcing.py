"""Sourcing service: forecasts -> scenarios -> optimal hedge -> backtest of the decision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from energy_sourcing.domain import ALL_TARGETS, Target
from energy_sourcing.optimization.optimizer import (
    CostEvaluator,
    CostMetrics,
    CVaRSourcingOptimizer,
    FloatArray,
    MarketParams,
    ProductType,
    RiskParams,
    SourcingProblem,
)
from energy_sourcing.optimization.portfolio import Portfolio
from energy_sourcing.optimization.scenarios import ResidualBootstrapGenerator
from energy_sourcing.services.forecasting import ForecastService

#: Positions are bounded at this multiple of the largest absolute scenario demand.
_POSITION_LIMIT_FACTOR = 1.5


@dataclass(frozen=True, slots=True)
class SourcingRequest:
    day: date
    portfolio: Portfolio = field(default_factory=Portfolio)
    market: MarketParams = field(default_factory=MarketParams)
    risk: RiskParams = field(default_factory=RiskParams)
    product: ProductType = ProductType.HOURLY
    n_scenarios: int = 500
    seed: int = 42


@dataclass(frozen=True, slots=True)
class StrategyOutcome:
    name: str
    positions: FloatArray
    scenario_costs: FloatArray
    metrics: CostMetrics
    realized_cost: float


@dataclass(frozen=True, slots=True)
class SourcingResult:
    day: date
    timestamps: pd.DatetimeIndex
    forward_price: FloatArray
    demand_mean: FloatArray
    demand_p05: FloatArray
    demand_p95: FloatArray
    demand_actual: FloatArray
    price_actual: FloatArray
    strategies: dict[str, StrategyOutcome]


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    risk_aversion: float
    expected_cost: float
    cvar: float
    hedge_ratio: float


class SourcingService:
    def __init__(self, forecasts: ForecastService, scenarios: ResidualBootstrapGenerator) -> None:
        self._forecasts = forecasts
        self._scenarios = scenarios

    def optimize(self, request: SourcingRequest) -> SourcingResult:
        problem, forecast = self._build_problem(request)
        actual_demand, actual_price = self._actuals(request.portfolio, forecast)
        optimal = CVaRSourcingOptimizer(request.risk, request.product).solve(problem)

        mean_demand = problem.demand.mean(axis=0)
        benchmarks = {
            "optimal": optimal.positions,
            "full_hedge": self._as_product(mean_demand, request.product),
            "no_hedge": np.zeros(problem.n_hours),
        }
        strategies = {}
        for name, positions in benchmarks.items():
            costs = optimal.scenario_costs if name == "optimal" else \
                CostEvaluator.scenario_costs(positions, problem)
            realized = CostEvaluator.costs(
                positions, actual_demand, actual_price, problem.forward, request.market
            )[0]
            strategies[name] = StrategyOutcome(
                name=name,
                positions=positions,
                scenario_costs=costs,
                metrics=CostEvaluator.metrics(costs, request.risk.alpha),
                realized_cost=float(realized),
            )

        return SourcingResult(
            day=request.day,
            timestamps=pd.DatetimeIndex(forecast.index),
            forward_price=problem.forward,
            demand_mean=mean_demand,
            demand_p05=np.quantile(problem.demand, 0.05, axis=0),
            demand_p95=np.quantile(problem.demand, 0.95, axis=0),
            demand_actual=actual_demand,
            price_actual=actual_price,
            strategies=strategies,
        )

    def frontier(self, request: SourcingRequest, risk_aversions: Sequence[float]) -> list[FrontierPoint]:
        """Efficient frontier; the same scenarios are reused for every lambda."""
        problem, _ = self._build_problem(request)
        expected_volume = float(problem.demand.mean(axis=0).sum())
        points = []
        for lam in sorted(risk_aversions):
            risk = RiskParams(risk_aversion=lam, alpha=request.risk.alpha)
            solution = CVaRSourcingOptimizer(risk, request.product).solve(problem)
            points.append(
                FrontierPoint(
                    risk_aversion=lam,
                    expected_cost=solution.metrics.expected,
                    cvar=solution.metrics.cvar,
                    hedge_ratio=_safe_ratio(float(solution.positions.sum()), expected_volume),
                )
            )
        return points

    def _build_problem(self, request: SourcingRequest) -> tuple[SourcingProblem, pd.DataFrame]:
        forecast = self._forecasts.forecast(request.day)
        point = {t: forecast[f"{t}_forecast"].to_numpy() for t in ALL_TARGETS}
        scenarios = self._scenarios.generate(point, request.n_scenarios, request.seed)
        demand = request.portfolio.residual_demand(
            scenarios[Target.LOAD], scenarios[Target.WIND], scenarios[Target.SOLAR]
        )
        limit = _POSITION_LIMIT_FACTOR * max(float(np.abs(demand).max()), 1.0)
        forward = point[Target.PRICE].astype(np.float64) + request.market.forward_premium
        problem = SourcingProblem(
            demand=demand,
            spot=scenarios[Target.PRICE],
            forward=forward,
            market=request.market,
            position_bounds=(-limit, limit),
        )
        return problem, forecast

    @staticmethod
    def _actuals(portfolio: Portfolio, forecast: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
        demand = portfolio.residual_demand(
            forecast["load_actual"].to_numpy(),
            forecast["wind_actual"].to_numpy(),
            forecast["solar_actual"].to_numpy(),
        )
        return demand, forecast["price_actual"].to_numpy(dtype=np.float64)

    @staticmethod
    def _as_product(positions: FloatArray, product: ProductType) -> FloatArray:
        if product is ProductType.BASELOAD:
            return np.full_like(positions, positions.mean())
        return positions


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-9 else 0.0
