# Statistical methods and reproduction notes

This document records, for every table, figure and test, what the code
actually computes, the assumptions behind each test, and the values a
reproduction should hit. Every statement refers to a function or constant in
the code; where the code and the paper's wording may differ, this is flagged
under [Known issues](#9-known-issues-and-open-questions).

State documented: commit `f5cb73a` plus uncommitted changes (one common
analysis sample of complete 4-asset time windows for every script, with ttm
and CSD centered on that sample;
`model_eval/vif.py`,
`model_eval/backtest_stats.py`). Checkpoint values in section 8 were read
from the output files produced by this code.

## 0. Environment

| Component | Version used |
|---|---|
| Python | 3.12.2 |
| numpy | 2.4.5 |
| pandas | 2.3.2 |
| statsmodels | 0.14.6 |
| scipy | 1.17.1 |
| matplotlib | 3.9.4 |

Raw data were collected in August 2026 (`download/raw_output/*.pkl`, file
dates 2026-08-19). The collector queries live APIs, so a fresh download may
differ slightly from this snapshot (late-resolved markets, API changes).

## 1. Data collection — `download/collect_polymarket_updown.py`

**Markets.** Polymarket "Up or Down" binary markets on BTC, ETH, SOL, XRP.

| Horizon | Window (UTC) | Slot length | Metadata endpoint |
|---|---|---|---|
| 5m | 2026-02-19 → 2026-07-31 | 300 s | Gamma `/events/slug` (`markets[0]`) |
| 15m | 2025-11-01 → 2026-07-31 | 900 s | Gamma `/markets/slug` |
| 60m | 2025-07-01 → 2026-07-31 | 3600 s | Gamma `/markets/slug` |

- Slugs: `slug_5m`, `slug_15m`, `slug_60m`. The 60m slug format changed
  several times; the boundaries are hard-coded (`_SLUG_*_EPOCH`), and 10 slots
  confirmed never opened are skipped (`_HOURLY_MISSING_SLOTS`).
- **Prices.** CLOB `/prices-history` with `fidelity=1` (1-minute), fetched
  separately for the YES ("Up") and NO token over `[slot_epoch, slot_epoch + W]`.
  The price is treated as the **mid-price**.
- **Time key.** `minute_left_dict`: `ttm = floor((end_ts − t) / 60)`, minutes
  remaining until close, kept for `0 ≤ ttm ≤ W`.
- **YES token.** `resolve_yes_no_index`: the outcome labelled "Up"
  (case-insensitive fallback); markets without it are skipped, not defaulted.
- **Outcome label.** From `outcomePrices` at fetch time: `Y = 1` if
  `p_yes ≥ 0.98`, `Y = 0` if `p_yes ≤ 0.02`; anything in between is treated as
  unresolved and skipped (`market_not_resolved`).
- Failures are logged to `*_poly_missing.pkl` (5m: 3, 15m: 113, 60m: 153).

## 2. Panel construction — `build_panel.py`

1. **Completeness** (`is_complete`): both token histories contain every
   `ttm ∈ {0, …, W−1}` (superset check; `ttm = W` is rarely present and not
   required).
2. **Simultaneity** (`filter_complete_and_simultaneous`): keep a slot only if
   all 4 assets survive step 1.
3. **Expansion** (`expand_price_history`): one row per episode (asset × slot)
   × token × ttm, with `1 ≤ ttm ≤ W−1` (`ttm = 0`, the close, is excluded).
   - `p` clipped to `[1e−6, 1 − 1e−6]`; `log_odds = ln(p / (1 − p))` (natural log).
   - `outcome` = `Y` for YES rows, `1 − Y` for NO rows.
   - `obs_epoch = slot_epoch + 60·W − 60·ttm` (wall-clock time of the tick).
4. **`delta_logit`** (`add_delta_logit`): within episode × token, sorted by
   decreasing ttm, `log_odds(ttm) − log_odds(ttm + 1)`. The first tick of each
   episode × token is NaN (never filled) and set to NaN for any gap ≠ 1 minute.
5. **`poly_vol_prev`** (`add_poly_vol_prev`): the market volume (USD) of the
   **previous** slot of the same asset, after filtering (so the current
   slot's own volume, known only at close, is never used).

| Horizon | Panel rows | Episodes | Slots | ttm range |
|---|---|---|---|---|
| 5m | 1,370,208 | 171,276 | 42,819 | 1–4 |
| 15m | 2,629,984 | 93,928 | 23,482 | 1–14 |
| 60m | 3,917,600 | 33,200 | 8,300 | 1–59 |

**YES/NO symmetry.** NO rows are near mirror images of YES rows
(`log_odds_no ≈ −log_odds_yes`, opposite outcome); |yes + no| < 0.05 in logit
for 87% (5m), 94% (15m) and 98% (60m) of rows. Consequences are listed in
sections 4 and 9.

## 3. Analysis features and samples — `analysis/common.py::load_analysis_panel`

Rows missing any variable in `ANALYSIS_NEED` (`log_odds`, `delta_logit`,
`poly_vol_log`, `csd_raw`) are dropped **first**; then every time window
(slot) that no longer has all 4 assets is dropped entirely (`N_ASSETS = 4`).
The centering and all interactions are computed on the remaining analysis
sample:

| Variable | Definition |
|---|---|
| `ttm_c` | `ttm − median(ttm)` over the analysis sample: 2.0 (5m), 7.0 (15m), 29.5 (60m) |
| `lo_x_ttm` | `log_odds × ttm_c` |
| `lo_x_delta` | `log_odds × delta_logit` |
| `poly_vol_log` | `ln(1 + max(poly_vol_prev, 0))` (`np.log1p`) |
| `lo_x_vol` | `log_odds × poly_vol_log` |
| `csd_raw` | see below |
| `csd_q_c` | `rank(csd_raw, pct=True) − 0.5` over the analysis-sample rows (both tokens) |
| `lo_x_csd` | `log_odds × csd_q_c` |
| `price` | `1 / (1 + exp(−log_odds))` (the clipped mid-price) |

**CSD** (`compute_csd`): for each `obs_epoch`, the sample standard deviation
(ddof = 1) of the 4 assets' YES `log_odds`; only obs_epochs with all 4 assets
are kept. Every row (both tokens, all 4 assets) at that obs_epoch gets the same
value. CSD is computed on the full panel before rows are dropped, because it is
a property of the time point, not of which rows pass the volume filter.

The result is **one analysis sample used by every table, figure, backtest and
robustness script**:

| Horizon | N (rows, both tokens) | Episodes | Slots (time windows) | First slot opens (UTC) | Last slot closes (UTC) |
|---|---|---|---|---|---|
| 5m | 777,720 | 129,620 | 32,405 | 2026-02-19 00:10 | 2026-07-30 23:50 |
| 15m | 2,216,136 | 85,236 | 21,309 | 2025-11-01 00:15 | 2026-07-31 00:00 |
| 60m | 3,675,344 | 31,684 | 7,921 | 2025-07-01 01:00 | 2026-07-31 00:00 |

Dropping removes the first tick of every episode (`delta_logit` NaN), so the
analysis ttm ranges are 1–3, 1–13 and 1–58 and each episode contributes
(T − 2) × 2 rows. It also removes episodes without the previous slot's volume
(the first slot of each asset and slots whose previous slot has no recorded
volume), and with them the rest of their time window, so every window in the
sample holds all 4 assets and every timestamp has exactly 8 rows (4 assets ×
2 tokens). Ranking rows for `csd_q_c` is therefore equivalent to ranking
timestamps. (History: before the volume filter was applied to every script,
Table 7, Figure 1 and `model_eval` used 1,027,656 / 2,442,128 / 3,851,200 rows;
before incomplete windows were dropped, the sample had 838,158 / 2,237,742 /
3,685,900 rows.)

Both centered variables have mean 0 in the analysis sample, so in models
with interactions the coefficient on `log_odds` is the calibration slope at
the sample-median ttm and the sample-median CSD. Shifting the ttm center is a
pure reparametrization (fitted values unchanged); re-ranking CSD within the
sample changes the regressor itself. (Before this change both were centered
on the full panel: ttm medians 2.5 / 7.5 / 30.0.)

## 4. Estimation and inference

### 4.1 Model

Every regression is a binomial GLM with logit link (`common.fit`, statsmodels
IRLS):

$$\operatorname{logit} P(Y_i = 1) = \alpha + \beta\,\ell_i + \textstyle\sum_k \gamma_k x_{ki}, \qquad \ell_i = \ln\frac{p_i}{1-p_i}.$$

The calibration slope `β = 1` means prices are calibrated; `β > 1` means
prices are compressed toward 0.5 (favorites underpriced, longshots
overpriced), i.e. the favorite-longshot bias.

**Assumptions.** Logit-linear calibration curve (plus the listed
interactions); correct conditional mean; observations in different clusters
independent.

### 4.2 Standard errors

Cluster-robust sandwich covariance (statsmodels `cov_type="cluster"`):

$$\hat V = c\,(X'WX)^{-1}\Big(\sum_g s_g s_g'\Big)(X'WX)^{-1},\qquad c = \frac{G}{G-1}\cdot\frac{N-1}{N-K}$$

where `s_g` is the score summed within cluster `g`. Settings verified on the
fitted object: `use_correction=True`, `use_t=False` (normal reference
distribution), `adjust_df=True`.

- **Main: cluster = time window** (`slot_epoch`): the 4 assets in a slot share
  the same CSD value and common crypto shocks, so episodes in a slot are not
  independent. Output: `results/slot/`, Figure 1.
- **Robustness: cluster = episode** (`episode_id`): `results/episode/`. Slot
  clustering inflated SEs by roughly ×1.35–1.5 relative to episode
  clustering (β in Table 2: ×1.38–1.45; ℓ×CSD in Table 5 (3): ×1.40–1.46).
- **Assumed:** independence across slots. Serial correlation between
  adjacent time windows is not modelled.
- YES and NO rows of an episode are always in the same cluster, so the
  mirrored duplication does not shrink the clustered SEs. It does pin the
  intercept near 0 and shrink its SE (see 9).

### 4.3 Tests

| Quantity | Code | Statistic | Reference distribution |
|---|---|---|---|
| Single coefficient vs Null | `coef_se_p` | `z = (b − null)/SE`; null = 1 for `log_odds`, 0 otherwise | N(0,1), two-sided |
| Joint test of added terms | `fit_row` → `res.wald_test(..., use_f=False)` | `W = b_A' V_A⁻¹ b_A` (cluster-robust V) | χ²(df = #added) |
| Explanatory power | `fit_row` | McFadden `R² = 1 − llf / llf₀`; `llf₀` from an intercept-only GLM on the same sample; `ΔR²` vs reference model | descriptive, no test |

Significance marks (`sig`): `***` p < 0.001, `**` p < 0.01, `*` p < 0.05,
`.` p < 0.1.

The Wald test has the same null as an LR test (added coefficients = 0) but
uses the cluster-robust covariance; the LR test assumes independent
observations and is reported only in `model_eval` (section 6.1).

## 5. Tables and figures

### Table 2 — `build_paper_tables.py` (`results/{slot,episode}/table2_basic_calibration.csv`)
Model `logit P(Y=1) = α + β ℓ`. Reports α with `H0: α = 0` (`alpha_p`,
`alpha_sig`) and β with `H0: β = 1` (`p_H0_beta_eq_1`, `sig`). Table sample.

### Table 3 — control-variable candidates (`table3_control_candidates.{csv,md}`)

| Model | Regressors | Reference ("vs.") |
|---|---|---|
| (1) | ℓ | — |
| (2) | ℓ, Δℓ | (1) |
| (3) | ℓ, ℓ×Δℓ | (1) |
| (4) | ℓ, Δℓ, TTM | (2) |
| (5) base | ℓ, Δℓ, ℓ×TTM | (2) |
| (6) | (5) + ln(1+vol) | (5) |
| (7) | (5) + ℓ×ln(1+vol) | (5) |

Reference = same model without the candidate term; ΔR² and the Wald χ² are
relative to it.

### Table 4 — correlations (`results/table4_correlation.csv`)
Pearson r between `log_odds`, `delta_logit`, `ttm`, `csd_q_c` on the table
sample (both tokens); p-values from `scipy.stats.pearsonr` assume i.i.d. rows
(no clustering). Correlations of `log_odds`/`delta_logit` with `ttm`/CSD are
mechanically ≈ 0 because of YES/NO mirroring (see 9).

### Table 5 — CSD-augmented models (`table5_csd_specifications.{csv,md}`)

| Model | Regressors | Reference |
|---|---|---|
| (1) | ℓ, Δℓ, ℓ×TTM | price only (Table 3, (1)) |
| (2) | (1) + CSD | (1) |
| (3) | (1) + ℓ×CSD | (1) |
| (4) | (1) + ℓ×CSD + CSD | (1) |
| (5) | ℓ, Δℓ | price only |
| (6) | (5) + CSD | (5) |
| (7) | (5) + ℓ×CSD | (5) |
| (8) | (5) + ℓ×CSD + CSD | (5) |

CSD enters as `csd_q_c` (percentile rank), ℓ×CSD as `lo_x_csd`.

### Figure B.1 — `csd_distribution.py`
Distinct obs_epochs of the analysis sample, one CSD value each (the value
itself is computed from all 4 assets in the panel); histogram with 80 bins on
`[0, 1.6 × p95]`; median, mean and pandas skewness annotated. Because the
first tick of each episode is not in the sample, those (early, low-CSD)
obs_epochs are excluded.

### Figure 1 — `csd_quantile_slope.py` (`results/fig1_csd_decile_slope.csv`)
Analysis sample, both tokens. Rows are split into deciles of `csd_raw` with
`pd.qcut(q=10)` (row-weighted, all ttm pooled). In each decile the base model
(ℓ, ℓ×TTM, Δℓ) is fit separately, slot-clustered; plotted: β and
`β ± 1.96·SE` against the β = 1 line. No formal test across deciles is
computed.

### Table 7 and Figure 2 — `equity_curve.py` (`results/table7_backtest.csv`)

| Item | Definition |
|---|---|
| Models | `BACKTEST_MODEL_SPECS`: M1 = ℓ; M2 = ℓ, Δℓ, ℓ×TTM; **M3 = M2 + ℓ×CSD** (no CSD level term) |
| Fit | in-sample, full analysis sample, both tokens |
| Rule (`trade_mask`) | buy 1 share when `p̂ − price ≥ 0.015` |
| PnL (`trade_cost_pnl`) | `Y − (price + 0.015)` per trade |
| ROI | `Σ PnL / Σ(price + 0.015) × 100` |
| Win rate | mean of `Y` over trades |
| Equity curve | cumulative PnL sorted by `obs_epoch` (simple, non-compounding) |
| MDD | `min(cumPnL − running max)` in dollars |
| Annualized PnL | `total PnL × 365 / n_days`, `n_days = max(span of trade times in days, 1)` |
| Calmar | `annualized PnL / |MDD|` |

Assumptions: execution at the mid-price, flat cost 0.015 per share, no
capital or position limits, several trades allowed per episode and on both
tokens, and the model is estimated on the same data it trades (look-ahead).

## 6. Robustness scripts — `model_eval/`

### 6.1 `model_comparison.py`
Analysis sample. In-sample part on **YES rows only** (NO rows would
double-count the likelihood):
- LR = `2(llf₁ − llf₀)` ~ χ²(df); AIC = `−2 llf + 2k`; BIC = `−2 llf + k ln n`
  (assume independence);
- slot-clustered Wald p for the added terms; slot-clustered SE of β and ℓ×CSD.

Walk-forward part:
- monthly expanding window by `slot_epoch` (UTC); test months start at the 3rd
  calendar month (`MIN_TRAIN_MONTHS = 2`); training = all earlier months;
- non-clustered GLM per fold (only predictions are used);
- `csd_q_c` recomputed from the training window only; test CSD mapped to the
  training empirical CDF (`searchsorted(..., side="right") / n_train`);
- log-loss (p clipped to [1e−6, 1 − 1e−6]) and Brier on YES rows, all rows and
  prices in [0.05, 0.95];
- Diebold–Mariano style test: per-day mean loss difference, OLS on a constant
  with Newey–West (Bartlett, 5 lags, no small-sample correction), two-sided
  normal p. For nested models (M2 ⊂ M3) this DM test has low power; the
  Clark–West test would be the standard alternative (not implemented);
- walk-forward backtest with the Table 7 rule next to the full-sample-fit
  backtest on the same test months.

`--insample-only` runs only the in-sample part.

### 6.2 `vif.py` — Table 6 (`results/table6_vif.csv`)
Analysis sample, both tokens. For Eq. c4 (= Table 5 (4)) and Eq. c3 (= Table 5
(3)):
- (A) unweighted VIF: `variance_inflation_factor` on `[const, X]`;
- (B) GLM-weighted VIF: Eq. c4 fit (slot-clustered; clustering does not
  change fitted values), every column including the constant multiplied by
  `sqrt(p̂(1−p̂))`, then (A). The same c4 `p̂` is used for c3;
- Pearson corr(`lo_x_ttm`, `lo_x_csd`).

`csd_q_c` is exactly orthogonal to the other regressors (VIF 1.000) because
it is identical on YES and NO rows while ℓ and Δℓ flip sign, so c3 and c4
give the same VIFs for the shared variables.

### 6.3 `backtest_stats.py` — Table 7 NW t column (`results/table7_backtest_nw_tests.csv`), Table C.1 (`results/tableC1_transaction_costs.csv`)
Reproduces Table 7 first and stops on any mismatch in trade count or total
PnL. Then:
- **PnL tests**: PnL summed per time window (slot) over **all** windows in the
  analysis sample (0 if no trade), sorted by start time; series M1, M2 − M1,
  M3 − M2 (a model with no trades contributes 0); OLS on a constant,
  Newey–West (Bartlett, `lags = floor(4 (T/100)^(2/9))`: 14 / 13 / 10, no
  small-sample correction), one-sided normal p for `H1: mean > 0`. Rows where
  both compared models have no trades are blank.
- **Entry mid-price distribution** over actual trades: summary statistics and
  a 20-bin histogram on [0, 1]; figure `FIG_mid_price_hist` shows shares per
  bin for x ≥ 0.4 and reports the count of trades below 0.4.

## 7. Reproduction

```
python download/collect_polymarket_updown.py --tf all   # network; slow
python build_panel.py
python analysis/build_paper_tables.py    # Tables 1-4 (slot + episode)
python analysis/csd_distribution.py      # Figure B.1
python analysis/csd_quantile_slope.py    # Figure 1
python analysis/equity_curve.py          # Table 7, Figure 2
python model_eval/model_comparison.py    # optional
python model_eval/vif.py                 # Table 6
python model_eval/backtest_stats.py      # Table 7 NW t column, Table C.1; needs results/table7_backtest.csv
```

Outputs are deterministic given the raw pickles (no random sampling;
`vif.py` would sample 500k rows with seed 42 only on a MemoryError, which did
not occur).

## 8. Checkpoints

**Samples** — see sections 2 and 3.

**Table 2 (slot clustering)**

| | α | p(α=0) | β | SE(β) | p(β=1) |
|---|---|---|---|---|---|
| 5m | 0.000070 | 0.285 | 1.0376 | 0.00773 | 1.2e−6 |
| 15m | 0.000013 | 0.554 | 1.0707 | 0.00820 | 6.1e−18 |
| 60m | −0.000021 | 0.016 | 1.1108 | 0.01235 | 3.0e−19 |

**Table 5, model (3) (slot clustering)**

| | ℓ×CSD | SE | p | Pseudo R² | ΔR² vs (1) | Wald χ²(1) |
|---|---|---|---|---|---|---|
| 5m | −0.0872 | 0.0276 | 1.6e−3 | 0.24038 | 5.9e−5 | 10.00 |
| 15m | −0.1752 | 0.0334 | 1.6e−7 | 0.31769 | 1.60e−4 | 27.45 |
| 60m | −0.1744 | 0.0567 | 2.1e−3 | 0.33627 | 1.26e−4 | 9.46 |

**Table 7**

| | M1 trades / PnL | M2 trades / PnL | M3 trades / PnL |
|---|---|---|---|
| 5m | 0 / 0 | 0 / 0 | 8,866 / −30.86 |
| 15m | 117,319 / −263.98 | 138,873 / −239.37 | 210,149 / 673.87 |
| 60m | 861,579 / 1,227.74 | 646,679 / 3,070.70 | 633,467 / 3,979.92 |

**Backtest PnL tests (`results/table7_backtest_nw_tests.csv`, one-sided p)**: M3 − M2
t = −0.54 (5m, p = 0.706), 2.36 (15m, p = 0.009), 1.62 (60m, p = 0.052);
M2 − M1 t = 0.16 (15m, p = 0.435), 1.83 (60m, p = 0.034).

## 9. Known issues and open questions

1. **M3 label.** Table 7 and `BACKTEST_MODEL_SPECS` use M2 + ℓ×CSD (Eq. c3).
   Using Eq. c4 (+ CSD level) changes the trade counts slightly. The paper
   should name the specification actually used.
2. **CSD–ttm confounding.** corr(ttm, `csd_q_c`) = −0.55 / −0.70 / −0.73. At
   60m the ℓ×CSD effect disappears without ℓ×TTM (Table 5 (7) vs (3)). A ttm
   fixed-effect specification has not been run.
3. **YES/NO mirroring.** It pins α near 0 and makes its SE very small, which
   is why the 60m intercept (−2e−5) is significant (p = 0.016). It also makes
   Table 4's correlations involving ℓ or Δℓ mechanically ≈ 0, and doubles the
   LR statistics in any two-token likelihood comparison.
4. **Multicollinearity.** All VIFs are below 5. The largest is ℓ at 5m:
   4.11 unweighted (2.42 weighted); ℓ×TTM and ℓ×CSD are at most 3.55 / 1.47
   (unweighted) and 2.01 / 2.03 (weighted).
5. **Backtest.** In-sample fit; mid-price execution; no bid/ask data, so
   spreads and taker fees cannot be measured. At the average entry mid ≈ 0.8,
   an assumed 1-tick spread plus the taker fee 0.07·p(1−p) is ≈ 0.016,
   slightly above the 0.015 cost used.
6. **Inference.** Normal/χ² approximations with large G; independence across
   slots assumed; Table 4 p-values are i.i.d.; LR/AIC/BIC in `model_eval`
   assume independence.
