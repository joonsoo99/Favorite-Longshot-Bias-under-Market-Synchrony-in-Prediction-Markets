# -*- coding: utf-8 -*-
# Supplementary statistics for the Table 7 backtest (analysis/equity_curve.py).
# Same setup as Table 7: BACKTEST_MODEL_SPECS (M1 price only, M2 base,
# M3 = M2 + logit x CSD), in-sample fits, edge threshold and fee 0.015, the
# common analysis sample, load_analysis_panel(tf) (first tick of each
# episode excluded via delta_logit). Trade counts and total PnL are
# checked against results/table7_backtest.csv before anything is computed.
#
# 1. PnL significance (per horizon): PnL is summed per time window (slot:
#    same horizon and start time, 4 assets), windows without trades are 0,
#    sorted by start time. Each model is compared with the previous one
#    (M1 vs 0, M2 - M1, M3 - M2); a model with no trades has PnL 0 in every
#    window. The series is regressed on a constant with Newey-West SEs,
#    lags = floor(4 (T/100)^(2/9)), one-sided test H1: mean > 0. Rows where
#    both compared models have no trades are left blank.
# 2. Mid-price at entry (actual trades only): summary statistics and a
#    histogram on [0, 1] with 20 bins of width 0.05.
# 3. Transaction costs (manuscript Table A.1): entry mid-price statistics,
#    all-taker cost per trade, and total PnL under the fixed $0.015 cost vs
#    the all-taker cost.
#
# Outputs: results/table7_backtest_nw_tests.csv (NW t column of manuscript
# Table 7), results/tableA1_transaction_costs.csv (Table A.1), and
# model_eval/results/mid_price_summary.csv, mid_price_hist.csv,
# FIG_mid_price_hist. results/table7_backtest.csv is only read.
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

import matplotlib.pyplot as plt

from common import (TF_CFG, BACKTEST_MODEL_SPECS, BACKTEST_MODEL_COLORS,
                     BACKTEST_EDGE_THRESH, BACKTEST_FEE, FIG_DPI, fit, load_analysis_panel,
                     trade_mask, trade_cost_pnl, style_axes, save_fig, setup_plot_rc)

OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {"M1": "M1_price_only", "M2": "M2_baseline", "M3": "M3_plus_csd"}
PREV = {"M1": None, "M2": "M1", "M3": "M2"}
BINS = np.linspace(0.0, 1.0, 21)

PAPER_DIR = os.path.join(ROOT, "results")  # manuscript tables
table7 = pd.read_csv(os.path.join(PAPER_DIR, "table7_backtest.csv"))

test_rows, summary_rows, hist_rows = [], [], []
for tf in TF_CFG:
    df = load_analysis_panel(tf)
    y = df["outcome"].values.astype(float)
    price = df["price"].values
    slots = df["slot_epoch"].values
    windows = np.sort(np.unique(slots))
    T = len(windows)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    print(f"[{tf}] rows={len(df):,}  windows={T:,}  NW lags={lags}")

    window_pnl, n_trades = {}, {}
    for m, spec in MODELS.items():
        res = fit(y, slots, df[BACKTEST_MODEL_SPECS[spec]])
        trade = trade_mask(res, price, BACKTEST_EDGE_THRESH)
        n = int(trade.sum())
        _, pnl = trade_cost_pnl(trade, price, y, BACKTEST_FEE)

        # reproduce Table 7 before using anything
        ref = table7[(table7.tf == tf) & (table7.model == spec)].iloc[0]
        if n != int(ref.n_trades) or not np.isclose(pnl.sum(), ref.total_pnl, atol=1e-6):
            sys.exit(f"[{tf} {m}] does not reproduce Table 7: trades {n:,} vs {int(ref.n_trades):,}, "
                     f"PnL {pnl.sum():.4f} vs {ref.total_pnl:.4f}; stopping.")
        print(f"  {m}: trades={n:,}  total PnL={pnl.sum():,.2f}  (matches Table 7)")

        window_pnl[m] = pd.Series(pnl, index=slots[trade]).groupby(level=0).sum() \
                          .reindex(windows, fill_value=0.0).values
        n_trades[m] = n

        # 2. mid-price at entry
        mids = price[trade]
        row = {"tf": tf, "model": m, "n_trades": n}
        if n:
            row.update({"mean": mids.mean(), "std": mids.std(ddof=1) if n > 1 else np.nan,
                        **{f"p{q}": np.percentile(mids, q) for q in (5, 10, 25, 50, 75, 90, 95)}})
        summary_rows.append(row)
        counts, _ = np.histogram(mids, bins=BINS)
        for lo, hi, c in zip(BINS[:-1], BINS[1:], counts):
            hist_rows.append({"tf": tf, "model": m, "bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                              "count": int(c)})

    # 1. PnL significance vs the previous model
    for m, prev in PREV.items():
        comparison = "vs 0" if prev is None else f"vs {prev}"
        row = {"tf": tf, "model": m, "comparison": comparison}
        if n_trades[m] == 0 and (prev is None or n_trades[prev] == 0):
            test_rows.append(row)
            continue
        d = window_pnl[m] - (0.0 if prev is None else window_pnl[prev])
        r = sm.OLS(d, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        t = float(r.tvalues[0])
        row.update({"total": float(d.sum()), "t_stat": t, "p_one_sided": float(sp_stats.norm.sf(t))})
        test_rows.append(row)

tests = pd.DataFrame(test_rows, columns=["tf", "model", "comparison", "total", "t_stat", "p_one_sided"])
summary = pd.DataFrame(summary_rows, columns=["tf", "model", "n_trades", "mean", "std",
                                              "p5", "p10", "p25", "p50", "p75", "p90", "p95"])
hist = pd.DataFrame(hist_rows)
tests.to_csv(os.path.join(PAPER_DIR, "table7_backtest_nw_tests.csv"), index=False, encoding="utf-8-sig")
summary.to_csv(os.path.join(OUT_DIR, "mid_price_summary.csv"), index=False, encoding="utf-8-sig")
hist.to_csv(os.path.join(OUT_DIR, "mid_price_hist.csv"), index=False, encoding="utf-8-sig")


# 3. Transaction costs (manuscript Table A.1). All-taker cost per trade = half
# of a one-tick ($0.01) spread + taker fee 0.07 p (1 - p), with p evaluated at
# the midpoint of each 0.05 entry-price bin and averaged over trades. The
# all-taker PnL replaces the fixed $0.015 cost of Table 7 with this cost, so
# it is approximate (bin midpoints, not trade-level prices).
HALF_TICK, TAKER_FEE_RATE = 0.005, 0.07
cost_rows = []
for _, s in summary[summary.n_trades > 0].iterrows():
    h = hist[(hist.tf == s.tf) & (hist.model == s.model)]
    mid = (h.bin_lo + h.bin_hi) / 2
    cost = HALF_TICK + TAKER_FEE_RATE * mid * (1 - mid)
    n = h["count"].sum()
    fixed_pnl = float(table7[(table7.tf == s.tf) & (table7.model == MODELS[s.model])].total_pnl.iloc[0])
    cost_rows.append({
        "tf": s.tf, "model": s.model, "n_trades": int(s.n_trades),
        "mid_mean": s["mean"], "mid_p10": s.p10, "mid_p50": s.p50, "mid_p90": s.p90,
        "all_taker_cost_per_trade": float((cost * h["count"]).sum() / n),
        "pnl_fixed_0015": fixed_pnl,
        "pnl_all_taker": fixed_pnl + float(((BACKTEST_FEE - cost) * h["count"]).sum()),
    })
costs = pd.DataFrame(cost_rows)
costs.to_csv(os.path.join(PAPER_DIR, "tableA1_transaction_costs.csv"), index=False, encoding="utf-8-sig")



# Entry mid-price histogram as a figure: one panel per horizon, each model as
# a step line of the share of its trades per 0.05 bin (models differ in trade
# count by ~80x, so counts are normalized). Line style doubles the color so
# models stay distinguishable in grayscale / for color-vision deficiency.
# Trades cluster on favorites, so the x-axis starts at X_MIN; any trades
# below it are counted in a panel note rather than silently cropped.
LINESTYLE = {"M1": "-", "M2": "--", "M3": ":"}
X_MIN = 0.4
setup_plot_rc()
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9), dpi=FIG_DPI, sharey=True)
fig.patch.set_facecolor("#fcfcfb")
for ax, tf in zip(axes, TF_CFG):
    style_axes(ax)
    absent, below = [], 0
    keep = BINS[:-1] >= X_MIN - 1e-9
    for m, spec in MODELS.items():
        h = hist[(hist.tf == tf) & (hist.model == m)]
        total = h["count"].sum()
        if total == 0:
            absent.append(m)
            continue
        share = h["count"].values / total * 100
        below += int(h["count"].values[~keep].sum())
        ax.stairs(share[keep], BINS[np.r_[keep, True]], color=BACKTEST_MODEL_COLORS[spec],
                  linestyle=LINESTYLE[m], linewidth=2, label=f"{m} (n={total:,})")
    ax.set_title(tf, fontsize=9, fontweight="bold")
    ax.set_xlim(X_MIN, 1)
    notes = ([f"{', '.join(absent)}: no trades"] if absent else []) + \
            ([f"{below:,} trades below {X_MIN} not shown"] if below else [])
    if notes:
        ax.text(0.03, 0.62, "\n".join(notes), transform=ax.transAxes, fontsize=6.5, color="#52514e")
    ax.set_xlabel("Mid-price at entry", fontsize=7.5)
    if ax is axes[0]:
        ax.set_ylabel("Share of trades (%)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=2.2)
fig.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "FIG_mid_price_hist"))

pd.set_option("display.width", 200)
print("\n" + tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nsaved → {OUT_DIR}")
