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
analysis/equity_curve.py         → output/FIG3, backtest_results.csv (backtest + risk-adjusted metrics)
```

`analysis/common.py` holds the utilities (panel loading/feature construction,
CSD computation, cluster-robust GLM fitting, backtest trade logic, plot
styling) shared by the four analysis scripts.

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
```
