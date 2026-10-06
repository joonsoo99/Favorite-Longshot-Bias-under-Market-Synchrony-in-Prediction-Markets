# Favorite-Longshot Bias under Market Synchrony in Prediction Markets

Code for a study of favorite-longshot bias (FLB) in Polymarket BTC/ETH/SOL/XRP
"Up or Down" prediction markets (5m / 15m / 60m horizons), and how it relates
to cross-sectional dispersion (CSD) across the four assets.

## Data

Raw data is not included in this repository. It is fully reproducible from
Polymarket's public Gamma API (market metadata) and CLOB API
(`/prices-history`) via `download/collect_polymarket_updown.py`.

## Pipeline

```
download/collect_polymarket_updown.py --tf {5m,15m,60m,all}
    → download/raw_output/{5m,15m,60m}_poly_raw.pkl

build_panel.py
    → panels/panel_{5m,15m,60m}.pkl

analysis/build_paper_tables.py   → output/{slot,episode}/table1, 2, 4 + output/table3 (calibration, controls, CSD models)
analysis/csd_distribution.py     → output/FIG1 (CSD distribution)
analysis/csd_quantile_slope.py   → output/FIG2 (calibration slope vs. CSD decile)
analysis/equity_curve.py         → output/FIG3, backtest_results.csv (Table 5: backtest + risk-adjusted metrics)

model_eval/model_comparison.py   → model_eval/output/ (robustness: model comparison, walk-forward out-of-sample)
```

`analysis/common.py` holds the utilities (panel loading/feature construction,
CSD computation, cluster-robust GLM fitting, backtest trade logic, plot
styling) shared by the analysis scripts.

### Tables

| Table | Content |
|---|---|
| Table 1 | calibration regression logit P(Y=1) = α + β ln(p/(1-p)); tests α = 0 and β = 1 |
| Table 2 | control-variable candidates (1)-(7); (5) is the base model |
| Table 3 | correlation matrix |
| Table 4 | CSD-augmented models (1)-(8) |

Standard errors are cluster-robust, and Tables 1, 2 and 4 are written once
per clustering (coefficients are identical; SEs, p-values, stars and Wald
tests differ):

- `output/slot/`: clustered by time window (`slot_epoch`), the main
  specification. All 4 assets in a slot share the same CSD value and common
  crypto shocks, so episodes in the same slot are not independent.
- `output/episode/`: clustered by episode (asset × slot), as a robustness check.

Table 3 does not depend on clustering (`output/table3_correlation.csv`).
FIG2's confidence intervals are slot-clustered.

Tables 2 and 4 are written both as CSV (all statistics) and as paper-layout
Markdown (`*.md`: one panel per horizon, coefficients with SEs, then
explanatory-power rows). The explanatory-power rows are McFadden pseudo R²,
its gain over a reference model (ΔR²), and a cluster-robust Wald test (same
clustering as the SEs) that the added terms are zero:

| Table | Reference model |
|---|---|
| Table 2 | same model without the candidate term |
| Table 4 | same model without the CSD terms; (1) and (5) vs. price only (Table 2, (1)) |

### Robustness (`model_eval/`)

`model_comparison.py` reports, on YES-token rows:
- in-sample log-likelihood, AIC/BIC, LR tests and slot-clustered Wald tests for
  M1 (price) ⊂ M2 (base) ⊂ M3 (+CSD);
- expanding-window monthly walk-forward log-loss/Brier vs. the market price,
  with Diebold-Mariano-style tests (CSD rank recomputed from the training
  window only);
- the walk-forward backtest next to the in-sample-fit backtest.

`--insample-only` runs just the in-sample part.

## Requirements

Python 3.12+, with `requests`, `pandas`, `numpy`, `tqdm`, `statsmodels`,
`scipy`, `matplotlib`.

## Usage

```
python download/collect_polymarket_updown.py --tf all
python build_panel.py
python analysis/build_paper_tables.py
python analysis/csd_distribution.py
python analysis/csd_quantile_slope.py
python analysis/equity_curve.py
python model_eval/model_comparison.py   # optional robustness checks
```
