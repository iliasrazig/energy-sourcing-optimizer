"""Streamlit front-end: train, inspect forecasts, optimise and backtest the sourcing decision.

The UI talks to the services in-process (no API hop): it loads the model store
through :func:`build_container`, exactly like a serving process would.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make `energy_sourcing` importable when the app is run with `streamlit run` without
# installing the package (src/ layout: this file is src/energy_sourcing/ui/app.py).
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from energy_sourcing.config import DataSourceKind, get_settings
from energy_sourcing.data.repository import build_repository
from energy_sourcing.domain import ALL_TARGETS, Target
from energy_sourcing.exceptions import EnergySourcingError, QualityGateError
from energy_sourcing.logging_config import configure_logging
from energy_sourcing.models.store import ModelStore
from energy_sourcing.optimization import MarketParams, Portfolio, ProductType, RiskParams
from energy_sourcing.pipelines.train import TrainingPipeline
from energy_sourcing.services import ServiceContainer, SourcingRequest, build_container
from energy_sourcing.services.sourcing import SourcingResult
from energy_sourcing.training.tracking import NullTracker

# Categorical slots 1-3 of the reference palette (validated all-pairs), plus a neutral for actuals.
STRATEGY_COLORS = {"optimal": "#2a78d6", "full_hedge": "#eb6834", "no_hedge": "#1baf7a"}
STRATEGY_LABELS = {"optimal": "Optimal (CVaR)", "full_hedge": "Full hedge", "no_hedge": "No hedge"}
FORECAST_COLOR = "#2a78d6"
ACTUAL_COLOR = "#8a8984"
UNITS = {Target.LOAD: "MW", Target.WIND: "MW", Target.SOLAR: "MW", Target.PRICE: "EUR/MWh"}

st.set_page_config(page_title="Energy Sourcing Optimizer", page_icon="⚡", layout="wide")


# --------------------------------------------------------------------------- #
# Loading and training
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models from the store...")
def load_container(artifacts_dir: str) -> ServiceContainer:
    return build_container(ModelStore(Path(artifacts_dir)))


def train_models(source: DataSourceKind, profile: str, artifacts_dir: Path) -> tuple[bool, str]:
    settings = get_settings().model_copy(
        update={"data_source": source, "training_profile": profile, "artifacts_dir": artifacts_dir}
    )
    configure_logging(settings.log_level)
    pipeline = TrainingPipeline(
        settings, build_repository(settings), NullTracker(), ModelStore(settings.artifacts_dir)
    )
    try:
        pipeline.run()
    except QualityGateError as exc:  # artifacts are still written; only the release is blocked
        return True, f"Models trained, but the quality gate failed: {exc}"
    except EnergySourcingError as exc:
        return False, f"Training failed: {exc}"
    return True, "Models trained and all quality gates passed."


def render_training_controls(artifacts_dir: Path, *, expanded: bool) -> None:
    with st.sidebar.expander("Train models", expanded=expanded):
        source = st.radio(
            "Data source",
            list(DataSourceKind),
            index=list(DataSourceKind).index(DataSourceKind.SYNTHETIC),
            format_func=lambda k: (
                "Synthetic (offline, ~1 min)"
                if k is DataSourceKind.SYNTHETIC
                else "OPSD real data (downloads ~120 MB)"
            ),
        )
        profile = st.radio("Profile", ["fast", "full"], horizontal=True) or "fast"
        if st.button("Train", type="primary", width="stretch"):
            with st.spinner("Training... (see the terminal for logs)"):
                ok, message = train_models(
                    DataSourceKind(source or DataSourceKind.SYNTHETIC), profile, artifacts_dir
                )
            load_container.clear()
            (st.success if ok else st.error)(message)
            if ok:
                st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar: decision parameters
# --------------------------------------------------------------------------- #
def sidebar_request(days: list[date]) -> SourcingRequest:
    sb = st.sidebar
    sb.header("Delivery day")
    day = sb.selectbox("Day (UTC, out-of-sample)", days, index=0, format_func=str)

    sb.header("Portfolio")
    sb.caption("Share of the zone's quantities that belong to the supplier.")
    load_share = sb.number_input("Load share", 0.0, 1.0, 0.01, 0.001, format="%.3f")
    wind_share = sb.number_input("Wind share", 0.0, 1.0, 0.008, 0.001, format="%.3f")
    solar_share = sb.number_input("Solar share", 0.0, 1.0, 0.006, 0.001, format="%.3f")

    sb.header("Market (EUR/MWh)")
    premium = sb.number_input("Forward premium", -50.0, 50.0, 1.5, 0.5)
    shortfall = sb.number_input("Shortfall spread", 0.0, 200.0, 8.0, 1.0)
    surplus = sb.number_input("Surplus spread", 0.0, 200.0, 8.0, 1.0)

    sb.header("Risk")
    risk_aversion = sb.slider("Risk aversion λ", 0.0, 5.0, 0.5, 0.05)
    alpha = sb.slider("CVaR confidence alpha", 0.80, 0.99, 0.95, 0.01)
    product = sb.radio(
        "Product",
        list(ProductType),
        horizontal=True,
        format_func=lambda p: "Hourly" if p is ProductType.HOURLY else "Baseload",
    )
    n_scenarios = sb.slider("Scenarios", 50, 1000, 300, 50)
    seed = int(sb.number_input("Seed", 0, 10_000, 42, 1))

    return SourcingRequest(
        day=day,
        portfolio=Portfolio(load_share, wind_share, solar_share),
        market=MarketParams(premium, shortfall, surplus),
        risk=RiskParams(risk_aversion, alpha),
        product=ProductType(product or ProductType.HOURLY),
        n_scenarios=n_scenarios,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def _layout(fig: go.Figure, title: str, y_title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.15},
        yaxis_title=y_title,
    )
    return fig


def forecast_tab(container: ServiceContainer, day: date) -> None:
    frame = container.forecasts.forecast(day)
    st.caption(
        f"Forecasts for {day} issued with D-1 information only; actuals are shown for evaluation."
    )
    rows = []
    for target in ALL_TARGETS:
        error = frame[f"{target}_forecast"] - frame[f"{target}_actual"]
        rows.append(
            {
                "target": str(target),
                "model": container.manifest.targets[target.value].best_model,
                "MAE": float(error.abs().mean()),
                "bias": float(error.mean()),
                "unit": UNITS[target],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    columns = st.columns(2)
    for i, target in enumerate(ALL_TARGETS):
        fig = go.Figure()
        if target is Target.PRICE:
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame["price_q90"],
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame["price_q10"],
                    fill="tonexty",
                    fillcolor="rgba(42,120,214,0.15)",
                    line={"width": 0},
                    name="P10-P90",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[f"{target}_forecast"],
                name="Forecast",
                line={"color": FORECAST_COLOR, "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[f"{target}_actual"],
                name="Actual",
                line={"color": ACTUAL_COLOR, "width": 2, "dash": "dot"},
            )
        )
        with columns[i % 2]:
            st.plotly_chart(_layout(fig, str(target).capitalize(), UNITS[target]), width="stretch")


@st.cache_data(show_spinner="Solving the stochastic LP...", max_entries=32)
def _optimize(_container: ServiceContainer, key: str, _request: SourcingRequest) -> SourcingResult:
    # ``key`` (the request's repr) is the cache key; underscored args are not hashed.
    return _container.sourcing.optimize(_request)


def optimize_tab(container: ServiceContainer, request: SourcingRequest, key: str) -> None:
    result = _optimize(container, key, request)
    alpha = request.risk.alpha

    rows = []
    for name, outcome in result.strategies.items():
        m = outcome.metrics
        rows.append(
            {
                "strategy": STRATEGY_LABELS[name],
                "volume bought (MWh)": float(outcome.positions.sum()),
                "expected cost": m.expected,
                "std": m.std,
                f"CVaR {alpha:.0%}": m.cvar,
                "worst": m.worst,
                "realized cost": outcome.realized_cost,
            }
        )
    table = pd.DataFrame(rows)

    optimal = result.strategies["optimal"]
    best_benchmark = min(
        (result.strategies[n] for n in ("full_hedge", "no_hedge")), key=lambda s: s.realized_cost
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimal expected cost", f"€{optimal.metrics.expected:,.0f}")
    c2.metric(f"Optimal CVaR {alpha:.0%}", f"€{optimal.metrics.cvar:,.0f}")
    c3.metric("Optimal realized cost", f"€{optimal.realized_cost:,.0f}")
    c4.metric(
        f"Realized vs {STRATEGY_LABELS[best_benchmark.name]}",
        f"€{optimal.realized_cost - best_benchmark.realized_cost:+,.0f}",
        help="Negative = the optimal strategy was cheaper on the actuals than the best benchmark.",
    )
    st.dataframe(table.style.format(precision=0, thousands=","), hide_index=True, width="stretch")

    left, right = st.columns(2)
    x = result.timestamps
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=result.demand_p95, line={"width": 0}, showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=result.demand_p05,
            fill="tonexty",
            line={"width": 0},
            fillcolor="rgba(138,137,132,0.2)",
            name="Demand P5-P95",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=result.demand_actual,
            name="Actual demand",
            line={"color": ACTUAL_COLOR, "width": 2, "dash": "dot"},
        )
    )
    for name, outcome in result.strategies.items():
        if name == "no_hedge":
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=outcome.positions,
                name=STRATEGY_LABELS[name],
                line={"color": STRATEGY_COLORS[name], "width": 2, "shape": "hv"},
            )
        )
    left.plotly_chart(_layout(fig, "Forward position vs residual demand", "MWh"), width="stretch")

    fig = go.Figure()
    for name, outcome in result.strategies.items():
        fig.add_trace(
            go.Histogram(
                x=outcome.scenario_costs,
                name=STRATEGY_LABELS[name],
                marker_color=STRATEGY_COLORS[name],
                opacity=0.6,
                nbinsx=50,
            )
        )
    fig.add_vline(
        x=optimal.metrics.cvar,
        line_dash="dash",
        line_color=STRATEGY_COLORS["optimal"],
        annotation_text=f"Optimal CVaR {alpha:.0%}",
    )
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig = _layout(fig, "Scenario cost distribution", "scenarios")
    fig.update_layout(hovermode="closest", xaxis_title="EUR")
    right.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=result.forward_price,
            name="Forward price",
            line={"color": FORECAST_COLOR, "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=result.price_actual,
            name="Actual spot price",
            line={"color": ACTUAL_COLOR, "width": 2, "dash": "dot"},
        )
    )
    st.plotly_chart(
        _layout(fig, "Forward vs realized spot price", "EUR/MWh", height=280),
        width="stretch",
    )


def frontier_tab(container: ServiceContainer, request: SourcingRequest) -> None:
    st.caption(
        "Re-solves the day for several risk-aversion levels on the same scenarios: "
        "the trade-off between expected cost and tail risk (CVaR)."
    )
    raw = st.text_input("Risk-aversion values λ", "0, 0.1, 0.25, 0.5, 1, 2, 5")
    try:
        lambdas = sorted({float(v) for v in raw.split(",") if v.strip()})
    except ValueError:
        st.error("Enter comma-separated numbers.")
        return
    if any(v < 0 for v in lambdas) or not lambdas:
        st.error("λ values must be non-negative.")
        return
    if not st.button("Compute frontier", type="primary"):
        return
    with st.spinner(f"Solving {len(lambdas)} LPs..."):
        points = container.sourcing.frontier(request, lambdas)
    frame = pd.DataFrame([asdict(p) for p in points])

    fig = go.Figure(
        go.Scatter(
            x=frame["cvar"],
            y=frame["expected_cost"],
            mode="lines+markers+text",
            text=[f"λ={v:g}" for v in frame["risk_aversion"]],
            textposition="top right",
            marker={"size": 9, "color": STRATEGY_COLORS["optimal"]},
            line={"color": STRATEGY_COLORS["optimal"], "width": 2},
            customdata=frame["hedge_ratio"],
            hovertemplate="CVaR €%{x:,.0f}<br>Expected €%{y:,.0f}<br>Hedge ratio %{customdata:.0%}"
            "<extra></extra>",
        )
    )
    fig = _layout(fig, "Efficient frontier", "Expected cost (EUR)", height=420)
    fig.update_layout(xaxis_title=f"CVaR {request.risk.alpha:.0%} (EUR)", hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        frame.style.format(
            {"expected_cost": "{:,.0f}", "cvar": "{:,.0f}", "hedge_ratio": "{:.1%}"}
        ),
        hide_index=True,
        width="stretch",
    )


def backtest_tab(container: ServiceContainer, request: SourcingRequest, days: list[date]) -> None:
    st.caption(
        "Applies the current parameters to a range of days and compares the realized cost "
        "of each strategy on the actuals."
    )
    c1, c2, c3 = st.columns(3)
    start = c1.selectbox("From", days, index=0, format_func=str)
    end = c2.selectbox("To", days, index=min(13, len(days) - 1), format_func=str)
    n_scenarios = c3.slider(
        "Scenarios per day",
        50,
        500,
        150,
        50,
        help="Fewer scenarios keeps a multi-day backtest fast.",
    )
    selected = [d for d in days if start <= d <= end]
    if not selected:
        st.warning("Empty range.")
        return
    if not st.button(f"Run backtest on {len(selected)} days", type="primary"):
        return

    progress = st.progress(0.0)
    rows = []
    for i, day in enumerate(selected):
        result = container.sourcing.optimize(replace(request, day=day, n_scenarios=n_scenarios))
        rows.append(
            {
                "day": day,
                **{n: s.realized_cost for n, s in result.strategies.items()},
                "volume": float(np.clip(result.demand_actual, 0, None).sum()),
            }
        )
        progress.progress((i + 1) / len(selected), text=f"{day}")
    progress.empty()

    frame = pd.DataFrame(rows).set_index("day")
    totals = frame[list(STRATEGY_COLORS)].sum()
    cols = st.columns(3)
    for col, (name, total) in zip(cols, totals.items(), strict=True):
        col.metric(
            f"{STRATEGY_LABELS[name]} total",
            f"€{total:,.0f}",
            delta=None if name == "optimal" else f"{total - totals['optimal']:+,.0f} vs optimal",
            delta_color="inverse",
        )

    fig = go.Figure()
    cumulative = frame[list(STRATEGY_COLORS)].cumsum()
    for name, color in STRATEGY_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative[name],
                name=STRATEGY_LABELS[name],
                line={"color": color, "width": 2},
            )
        )
    st.plotly_chart(_layout(fig, "Cumulative realized cost", "EUR", height=380), width="stretch")
    wins = (frame["optimal"] <= frame[["full_hedge", "no_hedge"]].min(axis=1)).mean()
    st.write(f"Optimal strategy is the cheapest on **{wins:.0%}** of the days.")
    st.dataframe(frame.style.format("{:,.0f}"), width="stretch")


def models_tab(container: ServiceContainer) -> None:
    manifest = container.manifest
    st.write(
        f"Data source **{manifest.data_source}**, zone **{manifest.zone}**, "
        f"data {manifest.data_start:%Y-%m-%d} → {manifest.data_end:%Y-%m-%d}, "
        f"train ≤ {manifest.train_end}, validation ≤ {manifest.validation_end}, "
        f"trained {manifest.created_at:%Y-%m-%d %H:%M} UTC."
    )
    if manifest.all_gates_passed:
        st.success("All quality gates passed.")
    else:
        st.warning(
            "At least one quality gate failed: this model set would be blocked from release."
        )

    rows = []
    for target, report in manifest.targets.items():
        rows.append(
            {
                "target": target,
                "selected model": report.best_model,
                "test MAE": report.test_metrics.get("mae"),
                "naive MAE": report.naive_test_metrics.get("mae"),
                "skill vs naive": report.skill_vs_naive,
                "gate": "✅ pass" if report.gate_passed else "❌ fail",
                **{f"bench {k}": v for k, v in report.benchmarks.items()},
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {
                "test MAE": "{:,.2f}",
                "naive MAE": "{:,.2f}",
                "skill vs naive": "{:+.1%}",
                "bench tso_test_mae": "{:,.2f}",
            },
            na_rep="-",
        ),
        hide_index=True,
        width="stretch",
    )

    target = st.selectbox("Validation MAE of every candidate", list(manifest.targets))
    val = manifest.targets[target].validation_metrics
    st.dataframe(pd.DataFrame(val).T.style.format("{:,.3f}"), width="stretch")
    if manifest.quantile_metrics:
        st.subheader("Price quantile model (test)")
        st.json(manifest.quantile_metrics)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    settings = get_settings()
    artifacts_dir = settings.artifacts_dir.resolve()
    st.title("⚡ Energy Sourcing Optimizer")

    try:
        container = load_container(str(artifacts_dir))
    except EnergySourcingError as exc:
        st.info(f"No usable models in `{artifacts_dir}` ({exc}). Train them from the sidebar.")
        render_training_controls(artifacts_dir, expanded=True)
        return

    days = container.forecasts.available_days
    if not days:
        st.error("The model store has no out-of-sample day with complete data.")
        render_training_controls(artifacts_dir, expanded=True)
        return

    request = sidebar_request(days)
    render_training_controls(artifacts_dir, expanded=False)
    key = repr(request)

    tabs = st.tabs(["📈 Forecast", "🧮 Optimize", "🎯 Frontier", "🔁 Backtest", "🧠 Models"])
    try:
        with tabs[0]:
            forecast_tab(container, request.day)
        with tabs[1]:
            optimize_tab(container, request, key)
        with tabs[2]:
            frontier_tab(container, request)
        with tabs[3]:
            backtest_tab(container, request, days)
        with tabs[4]:
            models_tab(container)
    except (EnergySourcingError, ValueError) as exc:
        st.error(str(exc))


main()
