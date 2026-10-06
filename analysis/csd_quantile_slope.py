# -*- coding: utf-8 -*-
# Figure 2 (headline result): calibration slope beta(log_odds) declines as
# CSD (cross-sectional dispersion across the 4 assets) rises. Bins the
# sample into CSD deciles per timeframe, fits the base model (log_odds +
# lo_x_ttm + delta_logit) separately in each bin (cluster-robust by
# time window, slot_epoch), and plots beta with its 95% CI against a beta=1 reference line.
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (TF_CFG, OUT_DIR, TF_COLORS, TF_MARKERS, FIG_DPI, fit, style_axes,
                     load_analysis_panel, save_fig, setup_plot_rc)

setup_plot_rc()

N_BINS = 10


def fit_slope(sub: pd.DataFrame):
    res = fit(sub["outcome"].values.astype(float), sub["slot_epoch"].values,
               sub[["log_odds", "lo_x_ttm", "delta_logit"]])
    return float(res.params["log_odds"]), float(res.bse["log_odds"])


NEED = ["log_odds", "lo_x_ttm", "delta_logit", "csd_raw"]
rows = []

for tf in TF_CFG:
    print(f"\n{'='*60}\n  [{tf}] load + decile fit\n{'='*60}")
    df = load_analysis_panel(tf, extra_need=NEED)
    print(f"  sample: {len(df):,} rows  episodes: {df['episode_id'].nunique():,}")
    print(f"  CSD: median={df['csd_raw'].median():.3f}  "
          f"IQR=[{df['csd_raw'].quantile(.25):.3f}, {df['csd_raw'].quantile(.75):.3f}]")

    df["csd_bin"] = pd.qcut(df["csd_raw"], q=N_BINS, labels=False, duplicates="drop")

    for b in sorted(df["csd_bin"].dropna().unique()):
        sub = df[df["csd_bin"] == b]
        beta, se = fit_slope(sub)
        ci_lo, ci_hi = beta - 1.96 * se, beta + 1.96 * se
        row = {
            "tf": tf, "bin": int(b) + 1,
            "csd_lo": float(sub["csd_raw"].min()), "csd_hi": float(sub["csd_raw"].max()),
            "csd_median": float(sub["csd_raw"].median()),
            "beta": beta, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n": len(sub), "n_ep": int(sub["episode_id"].nunique()),
            "n_slots": int(sub["slot_epoch"].nunique()),
        }
        rows.append(row)
        print(f"  Q{int(b)+1}  CSD∈[{row['csd_lo']:.2f},{row['csd_hi']:.2f}]  "
              f"n={row['n']:,}  beta={beta:.4f}  95%CI=[{ci_lo:.4f},{ci_hi:.4f}]")

df_res = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "csd_quantile_slope.csv")
df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\nresults saved → {csv_path}")

# figsize width 7.5in ~= 190mm = Elsevier double-column (full page width) spec
fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=FIG_DPI)
fig.patch.set_facecolor("#fcfcfb")
style_axes(ax)

ax.axhline(1.0, color="#c3c2b7", linewidth=1.5, linestyle="--", zorder=1,
           label="Perfect calibration (β=1)")

for tf in TF_CFG:
    d = df_res[df_res["tf"] == tf].sort_values("bin")
    yerr = np.vstack([d["beta"] - d["ci_lo"], d["ci_hi"] - d["beta"]])
    ax.errorbar(d["bin"], d["beta"], yerr=yerr,
                color=TF_COLORS[tf], marker=TF_MARKERS[tf], markersize=7,
                linewidth=2, capsize=4, capthick=1.5, elinewidth=1.5,
                label=tf, zorder=3)

ax.set_xticks(range(1, N_BINS + 1))
ax.set_xticklabels([f"Q{i}\n(low)" if i == 1 else (f"Q{i}\n(high)" if i == N_BINS else f"Q{i}")
                     for i in range(1, N_BINS + 1)])
ax.set_xlabel("CSD quantile (cross-sectional dispersion)", color="#0b0b0b", fontsize=11)
ax.set_ylabel("Estimated calibration slope  β(log_odds)", color="#0b0b0b", fontsize=11)
ax.legend(frameon=False, loc="upper right", fontsize=9)

fig.tight_layout()
base_path = os.path.join(OUT_DIR, "FIG2")
save_fig(fig, base_path)
print(f"plot saved → {base_path}.png / .pdf")
