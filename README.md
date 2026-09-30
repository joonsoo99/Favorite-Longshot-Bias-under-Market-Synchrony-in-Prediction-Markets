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

analysis/build_paper_tables.py   → output/table1-4, tableA1 (calibration, CSD models)
analysis/csd_distribution.py     → output/FIG1 (CSD distribution)
analysis/csd_quantile_slope.py   → output/FIG2 (calibration slope vs. CSD decile)
analysis/equity_curve.py         → output/FIG3, backtest_results.csv (Table 5: backtest + risk-adjusted metrics)

model_eval/model_comparison.py   → model_eval/output/ (robustness: model comparison, walk-forward out-of-sample)
```

`analysis/common.py` holds the utilities (panel loading/feature construction,
CSD computation, cluster-robust GLM fitting, backtest trade logic, plot
styling) shared by the analysis scripts.

### Tables

Table 2, Table 4 and Table A1 are written both as CSV (all statistics) and as
paper-layout Markdown (`output/*.md`: one panel per horizon, coefficients with
SEs, then explanatory-power rows). Table 2 builds the base model up one
control at a time (`price` → `+delta` → `base`). The explanatory-power rows
are McFadden pseudo R², its gain over a reference model (ΔR²), and a
cluster-robust Wald test (by episode) that the added terms are zero:

| Table | Reference model |
|---|---|
| Table 2 | previous step |
| Table 4 | same model without the CSD terms; (1) and (5) vs. price only |
| Table A1 | same model without the candidate term |

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
