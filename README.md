# Energy Sourcing Optimizer

Day-ahead electricity buying for a supplier with a renewable portfolio, using
ML forecasts and a risk-aware optimizer.

For a bidding zone (Germany–Luxembourg by default), the platform forecasts four
hourly quantities: **load, wind, solar and the day-ahead price**. It turns
those forecasts into joint probabilistic scenarios. It then solves a
**CVaR-penalised stochastic linear program** to decide how much energy to buy
forward for each delivery hour. Each decision is backtested against what
actually happened.

```
 OPSD / synthetic data ─▶ validation ─▶ features ─▶ model selection ─▶ quality gate ─▶ artifacts/
                                                                                         │
       realized cost ◀─ backtest ◀─ CVaR LP (HiGHS) ◀─ residual bootstrap ◀─ forecasts ◀─┘
```

---

## How it works

### 1. Data
- **`OPSDDataSource`** downloads the [Open Power System Data](https://data.open-power-system-data.org/time_series/)
  hourly package (2020-10-06 release). It maps the zone's columns to a standard layout: `load`,
  `wind`, `solar`, `price`, `load_tso_forecast`.
- **`SyntheticDataSource`** generates a realistic fake market for tests, CI and offline demos.
  Load has daily, weekly and yearly patterns, wind is a persistent random process, and solar follows
  the sun. Price rises with residual load (merit-order effect).
- `TimeSeriesRepository` puts the data on a complete hourly UTC grid and fills gaps of up to 3 hours
  by interpolation. `DataValidator` checks the columns, the index, completeness and value ranges.
  The clean result is cached as a Parquet file.

### 2. Features without leakage
The forecast for day *D* is made on day *D-1*. So every feature built from past
values of the series uses a lag of **at least 24 h**, and `FeatureSpec`
enforces this. All 24 hours are predicted at once rather than one hour after
another.

| Group      | Features                                                                                      |
|------------|-----------------------------------------------------------------------------------------------|
| Calendar   | hour, day of week, month, weekend, public holiday (`holidays`), cyclic hour / day-of-year     |
| Lags       | 24 / 48 / 72 / 168 h (each target uses a subset)                                              |
| Rolling    | mean and std over 24 / 72 / 168 h windows                                                     |
| External   | price model only: forecast load, wind and solar, plus residual load and renewable share       |

### 3. Forecasting and model selection
Each target gets four candidate models. They are trained on a **chronological**
split: train up to 2019-12-31, validation up to 2020-03-31, test after that.

- baselines: `naive_d-1` (same hour yesterday) and `seasonal_profile` (average by day of week and hour)
- ML: Random Forest, gradient boosting (scikit-learn `HistGradientBoostingRegressor`)

The model with the lowest validation MAE is kept. It must then beat the naive
D-1 forecast on the untouched test set by a set margin, measured as skill
`1 − MAE / MAE_naive`. If it doesn't, the **quality gate** blocks the release.

The models are **chained**. The price model trains on actual load, wind and
solar. It is evaluated on *forecasts* of those values, because forecasts are
all that is known when the market closes. A gradient-boosting quantile model adds price
intervals (P10/P50/P90), and the report shows how often prices fall inside
them. The load forecast is also compared with the grid operator's own
day-ahead forecast.

### 4. Scenarios
`ResidualBootstrapGenerator` picks **whole days** of past forecast errors from
the validation period, for all four targets at once. It adds them to today's
forecasts. This keeps two real patterns in the scenarios. First, an error early
in the day tends to persist all day. Second, errors move together across
variables: when wind is under-forecast, price tends to be over-forecast.

### 5. Optimization
The decision has two stages (CVaR formulation of Rockafellar & Uryasev):

- **Stage 1:** buy `q_h` MWh forward for each hour at price `F_h`, which is the price forecast plus a
  forward premium.
- **Stage 2:** the actual residual demand `D` and spot price `P` become known. Any shortfall is
  bought at `P + c_buy`, and any surplus is sold at `P − c_sell`.

```
min_q  E[C(q)] + λ · CVaR_α(C(q))
```

Here λ sets how much tail risk matters compared with expected cost. Residual
demand is the supplier's share of load minus its own wind and solar output:
`load_share·load − wind_share·wind − solar_share·solar` (see `Portfolio`).

The problem is solved exactly as a linear program with SciPy's HiGHS solver.
Its matrices are sparse: 500 scenarios × 24 hours need about 60k stored values,
compared with roughly 2.4 GB for a dense matrix. Two product types are
supported: `hourly` (one position per hour) and `baseload` (one flat position
for the whole day).

`SourcingService` compares three strategies, both on the scenarios and on the
real outcome:

| Strategy     | Positions                                   |
|--------------|---------------------------------------------|
| `optimal`    | CVaR-optimal hedge                          |
| `full_hedge` | expected residual demand                    |
| `no_hedge`   | none: buy everything on the spot market     |

It can also trace an **efficient frontier**, showing expected cost against CVaR
for a range of λ values.

---

## Project layout

```
src/energy_sourcing/
├── config.py            # settings, read from ES_* environment variables
├── domain.py            # Target enum, standard column names
├── exceptions.py        # error classes, all based on EnergySourcingError
├── data/                # data sources (OPSD, synthetic), repository, validation
├── features/            # FeatureSpec, CalendarFeatures, FeatureBuilder
├── models/              # Forecaster base class, baselines, ML wrappers, model registry, model store
├── training/            # chronological splitter, trainer, metrics, experiment-tracking hook
├── pipelines/train.py   # end-to-end training command (es-train)
├── optimization/        # portfolio, scenario bootstrap, CVaR LP optimizer
└── services/            # ForecastService, SourcingService, startup wiring
```

---

## Installation

Requires Python 3.11 or later.

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e ".[ml]"     # training + optimization
pip install -e ".[ui]"     # + the Streamlit front-end
pip install -e ".[dev]"    # everything, plus pytest, ruff, mypy, pre-commit
```

The core package depends only on numpy, pandas and pydantic. Optional extras
are `ml`, `api` (FastAPI + Uvicorn) and `ui` (Streamlit + Plotly). The API has
not been built yet.

All models come from scikit-learn, so no system library is needed. XGBoost and
LightGBM were dropped because on macOS they need `libomp` (`brew install libomp`).

---

## Usage

### Train

```bash
# Real data (downloads OPSD on first run, about 120 MB)
es-train

# Offline, fast test run
es-train --source synthetic --profile fast

# Rebuild the dataset instead of using the cached Parquet file
es-train --refresh-data
```

Exit codes: `0` success, `1` training error, `2` release blocked by the quality
gate. Artifacts are still written when the gate fails.

### Web UI

```bash
es-ui                     # or: streamlit run src/energy_sourcing/ui/app.py
```

Then open http://localhost:8501. If `artifacts/` is empty, train from the
sidebar (synthetic data takes about a minute). The left sidebar holds the
delivery day, portfolio shares, market spreads, risk aversion, CVaR α, product
type and scenario count. The tabs are:

| Tab      | What it shows                                                                         |
|----------|---------------------------------------------------------------------------------------|
| Forecast | Forecast vs actual for load, wind, solar and price (with the P10–P90 band), daily MAE  |
| Optimize | Optimal forward position vs the demand band, scenario cost distribution, and expected cost, CVaR and realized cost for optimal, full-hedge and no-hedge |
| Frontier | Expected cost vs CVaR for a list of λ values                                           |
| Backtest | Cumulative realized cost of each strategy over a range of days                       |
| Models   | Manifest: selected model, test skill vs naive, quality gates, candidate scores        |

Artifacts are written to `artifacts/`:

```
artifacts/
├── manifest.json            # git SHA, data range, split, metrics, feature lists, gate results
├── dataset.parquet          # past data needed to build features when serving forecasts
├── residuals.parquet        # validation-period errors used to generate scenarios
└── models/
    ├── load.joblib, wind.joblib, solar.joblib, price.joblib
    └── price_quantiles.joblib
```

> joblib files are pickles. Only load artifacts produced by your own pipeline.

### Forecast and optimize from Python

```python
from energy_sourcing.models.store import ModelStore
from energy_sourcing.services import build_container, SourcingRequest
from energy_sourcing.optimization import Portfolio, RiskParams, ProductType

app = build_container(ModelStore("artifacts"))   # fails at startup if the feature code no longer matches the trained models

day = app.forecasts.available_days[0]            # out-of-sample backtest days
forecast = app.forecasts.forecast(day)           # <target>_forecast / _actual, price_q10/50/90

result = app.sourcing.optimize(SourcingRequest(
    day=day,
    portfolio=Portfolio(load_share=0.01, wind_share=0.008, solar_share=0.006),
    risk=RiskParams(risk_aversion=0.5, alpha=0.95),
    product=ProductType.HOURLY,
    n_scenarios=500,
))
for name, s in result.strategies.items():
    print(f"{name:11s} E[cost]={s.metrics.expected:,.0f}  CVaR={s.metrics.cvar:,.0f}  "
          f"realized={s.realized_cost:,.0f} EUR")

frontier = app.sourcing.frontier(SourcingRequest(day=day), [0, 0.25, 0.5, 1, 2, 5])
```

Forecasts are only available for days in the test period. Before features are
built, the delivery day's actual values are hidden. So each backtest uses only
what was known on D-1.

If you change the feature definitions, retrain. `build_container` refuses to
start when its feature lists differ from those recorded in the manifest.

---

## Configuration

All settings come from environment variables (or a `.env` file) starting with
`ES_`. Nested settings use `__` as a separator.

| Variable                         | Default         | Description                                     |
|----------------------------------|-----------------|-------------------------------------------------|
| `ES_DATA_SOURCE`                 | `opsd`          | `opsd` or `synthetic`                           |
| `ES_DATA_DIR`                    | `data`          | raw download and processed Parquet cache        |
| `ES_ARTIFACTS_DIR`               | `artifacts`     | where trained models and data are stored        |
| `ES_TRAINING_PROFILE`            | `full`          | `full` or `fast` (smaller models, for CI)       |
| `ES_RANDOM_STATE`                | `42`            | random seed                                     |
| `ES_LOG_LEVEL`                   | `INFO`          | logging level                                   |
| `ES_SPLIT__TRAIN_END`            | `2019-12-31`    | last training day                               |
| `ES_SPLIT__VALIDATION_END`       | `2020-03-31`    | last validation day                             |
| `ES_GATE__MIN_SKILL_VS_NAIVE`    | `0.0`           | minimum test skill compared with naive D-1      |
| `ES_ZONE__CODE`                  | `DE_LU`         | bidding zone                                    |
| `ES_ZONE__TIMEZONE`              | `Europe/Berlin` | zone's local time zone                          |
| `ES_ZONE__HOLIDAYS_COUNTRY`      | `DE`            | country used for public holidays                |

Default optimization settings: forward premium 1.5 EUR/MWh, shortfall and
surplus spreads 8.0 EUR/MWh, λ = 0.5, α = 0.95.

---

## Development

```bash
ruff check . && ruff format --check .
mypy src
pytest                       # tests/unit and tests/integration
pytest -m "not integration"  # skip slower end-to-end tests
pytest --cov
```

Test markers: `integration` (end-to-end train and serve, slower) and `network`
(needs internet, skipped in CI).

---

## License

MIT © Ilias Razig, Claude Code
