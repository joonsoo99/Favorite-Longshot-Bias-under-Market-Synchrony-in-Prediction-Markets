# -*- coding: utf-8 -*-
# Figure 3 + Table 5/6 (merged): backtest performance and risk-adjusted
# metrics. Strategy: buy 1 share whenever the model's fitted probability
# exceeds the market price by >= EDGE_THRESH (buy price = market price +
# FEE); pnl = realized outcome (0/1) - buy price. In-sample backtest (fit
# and evaluated on the same data).
#
# MDD/Calmar use the dollar cumulative-PnL curve in chronological
# (obs_epoch) order, simple (non-compounding) interest. MDD is defined as
# the absolute-dollar peak-to-trough drawdown rather than a %-normalized
# one, since normalizing by cumulative cost-to-date exaggerates early-period
# swings while cost is still small. Annualized PnL = final cum. PnL *
# (365/backtest_days); Calmar = annualized PnL / |MDD|.
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common import (TF_CFG, OUT_DIR, BACKTEST_MODEL_SPECS,
                     BACKTEST_MODEL_COLORS, BACKTEST_FEE, BACKTEST_EDGE_THRESH, BACKTEST_NEED, FIG_DPI,
                     fit, load_analysis_panel, trade_mask, trade_cost_pnl, style_axes, save_fig,
                     setup_plot_rc)

setup_plot_rc()

FEE, EDGE_THRESH, NEED = BACKTEST_FEE, BACKTEST_EDGE_THRESH, BACKTEST_NEED

EMPTY_ROW = dict(n_trades=0, total_cost=0.0, total_pnl=0.0, avg_pnl=np.nan, roi_pct=np.nan,
                  win_rate=np.nan, mdd_dollar=np.nan, n_days=np.nan, ann_pnl=np.nan, calmar=np.nan)


# One row of performance + risk-adjusted stats for >=1 trade;
# the trade.sum()==0 case is handled by the caller via EMPTY_ROW.
def backtest_row(trade, cost, pnl, outcome, t_dates):
    n_trades = int(trade.sum())
    order = np.argsort(t_dates)
    cum_pnl = np.cumsum(pnl[order])
    mdd = float((cum_pnl - np.maximum.accumulate(cum_pnl)).min())
    n_days = max((t_dates[order][-1] - t_dates[order][0]).total_seconds() / 86400, 1)
    total_pnl = float(cum_pnl[-1])
    total_cost = float(cost.sum())
    ann_pnl = total_pnl * 365 / n_days

    return dict(
        n_trades=n_trades, total_cost=total_cost, total_pnl=total_pnl,
        avg_pnl=float(pnl.mean()), roi_pct=total_pnl / total_cost * 100,
        win_rate=float((outcome[trade] == 1).mean()),
        mdd_dollar=mdd, n_days=n_days, ann_pnl=ann_pnl,
        calmar=ann_pnl / abs(mdd) if mdd != 0 else np.nan,
    ), cum_pnl, order


# figsize width 7.5in ~= 190mm = Elsevier double-column (full page width) spec
fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.1), dpi=FIG_DPI, sharey=False)
fig.patch.set_facecolor("#fcfcfb")
rows = []

for ax, tf in zip(axes, TF_CFG):
    print(f"\n{'='*70}\n  [{tf}]\n{'='*70}")
    df = load_analysis_panel(tf, extra_need=NEED)
    y  = df["outcome"].values.astype(float)
    groups = df["slot_epoch"].values  # SEs unused here; kept consistent with tables
    price = df["price"].values
    t_dates_all = pd.to_datetime(df["obs_epoch"].values, unit="s", utc=True)

    style_axes(ax)
    ax.axhline(0, color="#c3c2b7", linewidth=1.2, linestyle="--", zorder=1)

    for mname, feats in BACKTEST_MODEL_SPECS.items():
        res = fit(y, groups, df[feats])
        trade = trade_mask(res, price, EDGE_THRESH)
        if trade.sum() == 0:
            print(f"  [{mname}] no trades")
            rows.append({"tf": tf, "model": mname, **EMPTY_ROW})
            continue
        cost, pnl = trade_cost_pnl(trade, price, y, FEE)
        row, cum_pnl, order = backtest_row(trade, cost, pnl, y, t_dates_all[trade])
        rows.append({"tf": tf, "model": mname, **row})
        print(f"  [{mname}]  trades={row['n_trades']:,}  total_pnl=${row['total_pnl']:,.1f}  "
              f"roi={row['roi_pct']:.2f}%  win_rate={row['win_rate']*100:.1f}%  "
              f"mdd=${row['mdd_dollar']:,.1f}  calmar={row['calmar']:.2f}")

        # Legend shows just the model name (n/final are already in
        # backtest_results.csv) — panel width is narrow (190mm spec), so
        # longer labels would overlap/clip.
        ax.plot(t_dates_all[trade].values[order], cum_pnl, color=BACKTEST_MODEL_COLORS[mname],
                linewidth=1.3, label=mname.split("_")[0])

    ax.set_title(tf, fontsize=9, fontweight="bold")
    ax.set_xlabel("Time", fontsize=7.5)
    if ax is axes[0]:
        ax.set_ylabel("Cumulative PnL ($)", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.2, borderaxespad=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(labelsize=7)

fig.tight_layout()
base_path = os.path.join(OUT_DIR, "FIG3")
save_fig(fig, base_path)
print(f"\nplot saved → {base_path}.png / .pdf")

out = pd.DataFrame(rows)[["tf", "model", "n_trades", "total_cost", "total_pnl", "avg_pnl",
                           "roi_pct", "win_rate", "mdd_dollar", "ann_pnl", "calmar"]]
csv_path = os.path.join(OUT_DIR, "backtest_results.csv")
out.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\n{'='*96}\n  Table 5+6 combined: backtest performance and risk-adjusted metrics\n{'='*96}")
header = (f"{'tf':>4} {'model':<14} {'trades':>9} {'ROI%':>7} {'win%':>6} "
          f"{'PnL$':>10} {'MDD$':>10} {'AnnPnL$':>10} {'Calmar':>7}")
print(header)
for _, r in out.iterrows():
    print(f"{r['tf']:>4} {r['model']:<14} {r['n_trades']:>9,} {r['roi_pct']:>7.2f} "
          f"{r['win_rate']*100:>6.1f} {r['total_pnl']:>10,.1f} {r['mdd_dollar']:>10,.1f} "
          f"{r['ann_pnl']:>10,.1f} {r['calmar']:>7.2f}")
print(f"\ntable saved → {csv_path}")
