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

analysis/build_paper_tables.py   → Tables 1-5
analysis/csd_quantile_slope.py   → Figure 1
analysis/equity_curve.py         → Table 7, Figure 2
model_eval/vif.py                → Table 6
model_eval/backtest_stats.py     → Table 7 (NW t column), Table A.1
analysis/csd_distribution.py     → Figure A.1

model_eval/model_comparison.py   → model_eval/results/ (robustness: model comparison, walk-forward out-of-sample)
```

### Output files (named after the manuscript's tables and figures)

| Manuscript | File (in `results/`) |
|---|---|
| Table 1 — sample period and size | `table1_sample.csv` |
| Table 2 — basic calibration | `slot/table2_basic_calibration.csv` |
| Table 3 — candidate control specifications | `slot/table3_control_candidates.{csv,md}` |
| Table 4 — correlation matrices | `table4_correlation.csv` |
| Table 5 — CSD specifications | `slot/table5_csd_specifications.{csv,md}` |
| Table 6 — variance inflation factors | `table6_vif.csv` |
| Table 7 — backtest | `table7_backtest.csv`, `table7_backtest_nw_tests.csv` (NW t column) |
| Table A.1 — transaction costs | `tableA1_transaction_costs.csv` |
| Figure 1 — calibration slope by CSD decile | `fig1_csd_decile_slope.{pdf,png,csv}` |
| Figure 2 — cumulative PnL | `fig2_cumulative_pnl.{pdf,png}` |
| Figure A.1 — CSD distribution | `figA1_csd_distribution.{pdf,png}` |

`results/episode/` holds Tables 2, 3 and 5 with episode-clustered standard
errors (robustness).

See [METHODS.md](METHODS.md) for the statistical methods, assumptions,
sample definitions and checkpoint values needed to verify a reproduction.

`analysis/common.py` holds the utilities (panel loading/feature construction,
CSD computation, cluster-robust GLM fitting, backtest trade logic, plot
styling) shared by the analysis scripts.

### Tables

| Table | Content |
|---|---|
| Table 1 | sample period and size |
| Table 2 | calibration regression logit P(Y=1) = α + β ln(p/(1-p)); tests α = 0 and β = 1 |
| Table 3 | control-variable candidates (1)-(7); (5) is the base model |
| Table 4 | correlation matrix |
| Table 5 | CSD specifications (1)-(8) |

Standard errors are cluster-robust, and Tables 2, 3 and 5 are written once
per clustering (coefficients are identical; SEs, p-values, stars and Wald
tests differ):

- `results/slot/`: clustered by time window (`slot_epoch`), the main
  specification. All 4 assets in a slot share the same CSD value and common
  crypto shocks, so episodes in the same slot are not independent.
- `results/episode/`: clustered by episode (asset × slot), as a robustness check.

Tables 1 and 4 do not depend on clustering. Figure 1's confidence
intervals are slot-clustered.

Tables 3 and 5 are written both as CSV (all statistics) and as paper-layout
Markdown (`*.md`: one panel per horizon, coefficients with SEs, then
explanatory-power rows). The explanatory-power rows are McFadden pseudo R²,
its gain over a reference model (ΔR²), and a cluster-robust Wald test (same
clustering as the SEs) that the added terms are zero:

| Table | Reference model |
|---|---|
| Table 3 | same model without the candidate term |
| Table 5 | same model without the CSD terms; (1) and (5) vs. price only (Table 3, (1)) |

### `model_eval/`

- `vif.py`: variance inflation factors of the CSD-augmented model,
  unweighted and weighted by p̂(1−p̂) (Table 6).
- `backtest_stats.py`: Newey–West tests on backtest PnL aggregated by time
  window (Table 7, NW t column) and the transaction-cost analysis (Table A.1).
- `model_comparison.py`: additional robustness checks, not reported in the
  manuscript (outputs in `model_eval/results/`).

`model_comparison.py` reports, on YES-token rows:
- in-sample log-likelihood, AIC/BIC, LR tests and slot-clustered Wald tests for
  M1 (price) ⊂ M2 (base) ⊂ M3 (+CSD);
- expanding-window monthly walk-forward log-loss/Brier vs. the market price,
  with Diebold-Mariano-style tests (CSD rank recomputed from the training
  window only);
- the walk-forward backtest next to the in-sample-fit backtest.

`--insample-only` runs just the in-sample part.

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

```
python download/collect_polymarket_updown.py --tf all
python build_panel.py
python analysis/build_paper_tables.py
python analysis/csd_distribution.py
python analysis/csd_quantile_slope.py
python analysis/equity_curve.py
python model_eval/vif.py                # Table 6
python model_eval/backtest_stats.py     # Table 7 NW t column, Table A.1 (after equity_curve.py)
python model_eval/model_comparison.py   # optional robustness checks
```
