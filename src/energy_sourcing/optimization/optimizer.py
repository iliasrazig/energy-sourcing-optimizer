r"""Risk-averse day-ahead sourcing as a two-stage stochastic linear program.

Stage 1 (today): buy ``q_h`` MWh forward for each delivery hour at price ``F_h``.
Stage 2 (delivery): demand ``D_sh`` and spot price ``P_sh`` are revealed;
shortfalls are bought at ``P + c_buy`` and surpluses sold at ``P - c_sell``.

Scenario cost::

    C_s(q) = sum_h F_h q_h + (P_sh + c_buy) (D_sh - q_h)^+ - (P_sh - c_sell) (q_h - D_sh)^+

Objective (Rockafellar & Uryasev, 2000)::

    min_q  E[C(q)] + lambda * CVaR_alpha(C(q))

    CVaR_alpha = min_eta  eta + 1/((1-alpha) S) * sum_s (C_s - eta)^+

Introducing ``b_sh`` (shortfall), ``l_sh`` (surplus) and ``u_s`` (tail excess)
yields a linear program solved exactly by HiGHS. It is convex because
``c_buy + c_sell >= 0`` (you never buy back cheaper than you sell).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import linprog

from energy_sourcing.exceptions import OptimizationError

FloatArray = NDArray[np.float64]


class ProductType(StrEnum):
    HOURLY = "hourly"  # one position per delivery hour
    BASELOAD = "baseload"  # one flat position across the day (standard forward product)


@dataclass(frozen=True, slots=True)
class MarketParams:
    """Settlement terms of the second stage, in EUR/MWh."""

    forward_premium: float = 1.5
    shortfall_spread: float = 8.0
    surplus_spread: float = 8.0

    def __post_init__(self) -> None:
        if self.shortfall_spread + self.surplus_spread < 0:
            raise ValueError("shortfall_spread + surplus_spread must be >= 0 for convexity")


@dataclass(frozen=True, slots=True)
class RiskParams:
    risk_aversion: float = 0.5  # lambda
    alpha: float = 0.95

    def __post_init__(self) -> None:
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be >= 0")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class SourcingProblem:
    demand: FloatArray  # (S, H) residual demand scenarios, MWh
    spot: FloatArray  # (S, H) spot price scenarios, EUR/MWh
    forward: FloatArray  # (H,) forward price, EUR/MWh
    market: MarketParams
    position_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        if self.demand.shape != self.spot.shape or self.demand.ndim != 2:
            raise ValueError("demand and spot must share shape (n_scenarios, n_hours)")
        if self.forward.shape != (self.demand.shape[1],):
            raise ValueError("forward must have one price per hour")
        low, high = self.position_bounds
        if low > high:
            raise ValueError("invalid position bounds")

    @property
    def n_scenarios(self) -> int:
        return int(self.demand.shape[0])

    @property
    def n_hours(self) -> int:
        return int(self.demand.shape[1])


@dataclass(frozen=True, slots=True)
class CostMetrics:
    expected: float
    std: float
    var: float
    cvar: float
    worst: float


class CostEvaluator:
    """Exact (vectorised) evaluation of any position against the scenarios."""

    @staticmethod
    def scenario_costs(positions: FloatArray, problem: SourcingProblem) -> FloatArray:
        return CostEvaluator.costs(positions, problem.demand, problem.spot, problem.forward,
                                   problem.market)

    @staticmethod
    def costs(
        positions: FloatArray,
        demand: FloatArray,
        spot: FloatArray,
        forward: FloatArray,
        market: MarketParams,
    ) -> FloatArray:
        demand = np.atleast_2d(demand)
        spot = np.atleast_2d(spot)
        imbalance = demand - positions[np.newaxis, :]
        shortfall = np.clip(imbalance, 0.0, None)
        surplus = np.clip(-imbalance, 0.0, None)
        second_stage = (spot + market.shortfall_spread) * shortfall - (
            spot - market.surplus_spread
        ) * surplus
        return float(forward @ positions) + second_stage.sum(axis=1)

    @staticmethod
    def metrics(costs: FloatArray, alpha: float) -> CostMetrics:
        var = float(np.quantile(costs, alpha))
        # Rockafellar-Uryasev formula evaluated at eta = VaR: exact for discrete samples.
        cvar = var + float(np.mean(np.clip(costs - var, 0.0, None))) / (1.0 - alpha)
        return CostMetrics(
            expected=float(np.mean(costs)),
            std=float(np.std(costs)),
            var=var,
            cvar=cvar,
            worst=float(np.max(costs)),
        )


@dataclass(frozen=True, slots=True)
class SourcingSolution:
    positions: FloatArray  # (H,) MWh bought forward per hour
    scenario_costs: FloatArray  # (S,)
    metrics: CostMetrics
    objective: float


class CVaRSourcingOptimizer:
    def __init__(self, risk: RiskParams, product: ProductType = ProductType.HOURLY) -> None:
        self.risk = risk
        self.product = product

    def solve(self, problem: SourcingProblem) -> SourcingSolution:
        c, A_ub, b_ub, A_eq, b_eq, bounds, n_q = self._build_lp(problem)
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
        )
        if result.status != 0:
            raise OptimizationError(f"LP solver failed: {result.message}")

        positions = self._expansion(problem.n_hours, n_q) @ result.x[:n_q]
        costs = CostEvaluator.scenario_costs(positions, problem)
        return SourcingSolution(
            positions=positions,
            scenario_costs=costs,
            metrics=CostEvaluator.metrics(costs, self.risk.alpha),
            objective=float(result.fun),
        )

    def _expansion(self, n_hours: int, n_q: int) -> sp.csr_matrix:
        """Maps decision variables to hourly positions (identity or a column of ones)."""
        if self.product is ProductType.HOURLY:
            return sp.identity(n_hours, format="csr")
        return sp.csr_matrix(np.ones((n_hours, n_q)))

    def _build_lp(
        self, problem: SourcingProblem
    ) -> tuple[FloatArray, sp.csr_matrix | None, FloatArray | None, sp.csr_matrix, FloatArray,
               list[tuple[float | None, float | None]], int]:
        """Variables: ``[q (n_q), eta, u (S), b (S*H), l (S*H)]``; sparse throughout.

        With 500 scenarios x 24 hours the dense constraint matrix would hold
        ~300M entries (2.4 GB); the sparse one stores ~60k.
        """
        S, H = problem.n_scenarios, problem.n_hours
        n_q = H if self.product is ProductType.HOURLY else 1
        SH = S * H
        lam, alpha = self.risk.risk_aversion, self.risk.alpha
        use_cvar = lam > 0
        n_cvar = 1 + S if use_cvar else 0
        market = problem.market

        M = self._expansion(H, n_q)
        forward_q = np.asarray(M.T @ problem.forward).ravel()  # forward cost per decision unit
        buy = (problem.spot + market.shortfall_spread).ravel()  # (S*H,) scenario-major
        sell = (problem.spot - market.surplus_spread).ravel()

        # Objective: E[C] + lambda * (eta + sum u / ((1 - alpha) S))
        c_parts = [forward_q]
        if use_cvar:
            c_parts += [np.array([lam]), np.full(S, lam / ((1.0 - alpha) * S))]
        c_parts += [buy / S, -sell / S]
        c = np.concatenate(c_parts)

        # Balance: q_h + b_sh - l_sh = D_sh   for every (s, h)
        eye_sh = sp.identity(SH, format="csr")
        eq_blocks = [sp.vstack([M] * S, format="csr")]
        if use_cvar:
            eq_blocks.append(sp.csr_matrix((SH, n_cvar)))
        eq_blocks += [eye_sh, -eye_sh]
        A_eq = sp.hstack(eq_blocks, format="csr")
        b_eq = problem.demand.ravel()

        A_ub: sp.csr_matrix | None = None
        b_ub: FloatArray | None = None
        if use_cvar:
            # Tail excess: C_s - eta - u_s <= 0
            row_ptr = np.arange(0, SH + 1, H)
            cols = np.arange(SH)
            buy_rows = sp.csr_matrix((buy, cols, row_ptr), shape=(S, SH))
            sell_rows = sp.csr_matrix((-sell, cols, row_ptr), shape=(S, SH))
            A_ub = sp.hstack(
                [
                    sp.csr_matrix(np.tile(forward_q, (S, 1))),
                    sp.csr_matrix(-np.ones((S, 1))),
                    -sp.identity(S, format="csr"),
                    buy_rows,
                    sell_rows,
                ],
                format="csr",
            )
            b_ub = np.zeros(S)

        low, high = problem.position_bounds
        bounds: list[tuple[float | None, float | None]] = [(low, high)] * n_q
        if use_cvar:
            bounds += [(None, None)] + [(0.0, None)] * S
        bounds += [(0.0, None)] * (2 * SH)
        return c, A_ub, b_ub, A_eq, b_eq, bounds, n_q
