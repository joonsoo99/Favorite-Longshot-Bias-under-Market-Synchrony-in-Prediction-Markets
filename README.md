# Favorite–Longshot Bias under Market Synchrony in Prediction Markets

Code for the paper "Favorite–Longshot Bias under Market Synchrony in
Prediction Markets". It studies the favorite–longshot bias (FLB) in
Polymarket's BTC/ETH/SOL/XRP "Up or Down" markets at the 5-, 15- and
60-minute horizons, and how it varies with the cross-sectional dispersion
(CSD) of price log-odds across the four assets.

## Repository layout

| Path | Purpose |
|---|---|
| `download/` | data collection from Polymarket's public API |
| `build_panel.py` | sample construction: the four steps of the paper's Section 2, producing the minute-level analysis sample |
| `analysis/` | main tables and figures |
| `model_eval/` | variance inflation factors, Newey–West tests and transaction-cost analysis |

## Data

Raw data are not included. They can be re-collected from Polymarket's
public API (Gamma API for market metadata, CLOB API `/prices-history` for
minute-level prices) with `download/collect_polymarket_updown.py`.

## Environment setup

The results were produced with Python 3.12.2 and the package versions pinned
in `requirements.txt`.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Figures use the Arial font; on systems without it, matplotlib falls back to
its default font (layout may differ slightly, numbers do not).

## Usage

Run from the repository root, in this order:

```
python download/collect_polymarket_updown.py --tf all   # raw data -> download/raw_output/
python build_panel.py                                   # panels   -> panels/
python analysis/build_paper_tables.py                   # Tables 1-5
python analysis/csd_quantile_slope.py                   # Figure 1
python analysis/equity_curve.py                         # Table 7, Figure 2
python analysis/csd_distribution.py                     # Figure B.1
python model_eval/vif.py                                # Table 6
python model_eval/backtest_stats.py                     # Table 7 (NW t column), Table C.1
```

`model_eval/backtest_stats.py` reads `results/table7_backtest.csv`, so it
runs after `analysis/equity_curve.py`.

## Outputs

All outputs are written to `results/` and named after the paper's tables and
figures.

| Paper | File |
|---|---|
| Table 1 — sample period and size | `table1_sample.csv` |
| Table 2 — basic calibration model | `table2_basic_calibration.csv` |
| Table 3 — candidate control specifications | `table3_control_candidates.{csv,md}` |
| Table 4 — correlation matrices | `table4_correlation.csv` |
| Table 5 — CSD specifications | `table5_csd_specifications.{csv,md}` |
| Table 6 — variance inflation factors | `table6_vif.csv` |
| Table 7 — backtest | `table7_backtest.csv`, `table7_backtest_nw_tests.csv` (NW t column) |
| Table C.1 — transaction costs | `tableC1_transaction_costs.csv` |
| Figure 1 — calibration slope by CSD decile | `fig1_csd_decile_slope.{pdf,png,csv}` |
| Figure 2 — cumulative PnL | `fig2_cumulative_pnl.{pdf,png}` |
| Figure B.1 — distribution of the raw CSD | `figB1_csd_distribution.{pdf,png}` |

The `.md` versions of Tables 3 and 5 lay the estimates out as in the paper.
Standard errors are clustered by time window throughout.
`model_eval/backtest_stats.py` also writes the entry-price summary and
histogram used for Table C.1 to `model_eval/results/`.

`analysis/common.py` holds the shared utilities: panel loading and feature
construction, the CSD computation, cluster-robust logit estimation, the
backtest trading rule, and plot styling.
