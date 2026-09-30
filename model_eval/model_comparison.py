# -*- coding: utf-8 -*-
# Model-comparison checks that the coefficient tests in analysis/ don't cover:
#   (A) In-sample: log-likelihood, AIC/BIC, LR tests for nested models
#       M1 ⊂ M2 ⊂ M3, plus a cluster-robust Wald test clustered by slot_epoch
#       (all 4 assets in a slot share CSD and common crypto shocks, so
#       episode-level clustering alone understates SEs).
#   (B) Out-of-sample walk-forward (expanding window, monthly test folds):
#       log-loss / Brier vs the raw market price and between models, with a
#       Diebold-Mariano-style test on daily mean loss differences (HAC SE).
#   (C) Out-of-sample backtest with the same trade rule as equity_curve.py,
#       next to the in-sample-fit backtest on the same test rows.
#
# Likelihood-based metrics use YES-token rows only: NO rows are (almost
# exactly) mirror images of YES rows, so including them double-counts the
# likelihood. Fitting for the backtest keeps both tokens, as in equity_curve.py.
#
# csd_q_c (percentile rank of CSD) is computed on the full sample in
# common.load_analysis_panel(), which leaks the future distribution. In the
# walk-forward it is recomputed from the training window only and test CSD
# values are mapped onto the training empirical CDF.
#
# Usage: python model_eval/model_comparison.py [--insample-only]
import sys, io, os, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common import (TF_CFG, BACKTEST_MODEL_SPECS, BACKTEST_MODEL_COLORS, BACKTEST_FEE,
                     BACKTEST_EDGE_THRESH, BACKTEST_NEED, FIG_DPI, sig, load_analysis_panel,
                     style_axes, save_fig, setup_plot_rc)

OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_TRAIN_MONTHS = 2          # first test fold = 3rd calendar month in the data
EPS = 1e-6
PRICE_BAND = (0.05, 0.95)     # common filter in the calibration literature
MODELS = list(BACKTEST_MODEL_SPECS)

parser = argparse.ArgumentParser()
parser.add_argument("--insample-only", action="store_true", help="run only part (A)")
INSAMPLE_ONLY = parser.parse_args().insample_only


def glm_fit(y, X, groups=None):
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    m = sm.GLM(y, Xc, family=sm.families.Binomial())
    if groups is None:
        return m.fit(disp=False)
    return m.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)}, disp=False)


def glm_predict(res, X):
    return np.asarray(res.predict(sm.add_constant(X.astype(float), has_constant="add")))


def logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(y, p):
    return (p - y) ** 2


def add_csd_rank(train: pd.DataFrame, test: pd.DataFrame):
    """csd_q_c from the training window's CSD distribution only."""
    ref = np.sort(train["csd_raw"].values)
    train = train.copy(); test = test.copy()
    train["csd_q_c"] = train["csd_raw"].rank(pct=True) - 0.5
    test["csd_q_c"] = np.searchsorted(ref, test["csd_raw"].values, side="right") / len(ref) - 0.5
    for d in (train, test):
        d["lo_x_csd"] = d["log_odds"] * d["csd_q_c"]
    return train, test


def dm_test(loss_a, loss_b, day):
    """H0: equal expected loss. Positive mean => model b is better than a.
    Losses are averaged per day first (absorbs intra-day dependence), then a
    HAC (Newey-West, 5 lags) t-test on the daily differences."""
    d = pd.Series(loss_a - loss_b).groupby(np.asarray(day)).mean()
    r = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(d.mean()), float(r.tvalues[0]), float(r.pvalues[0]), len(d)


def backtest(price, outcome, p_hat):
    trade = (p_hat - price) >= BACKTEST_EDGE_THRESH
    n = int(trade.sum())
    if n == 0:
        return dict(n_trades=0, total_cost=0.0, total_pnl=0.0, roi_pct=np.nan, win_rate=np.nan), trade
    cost = price[trade] + BACKTEST_FEE
    pnl = outcome[trade] - cost
    return dict(n_trades=n, total_cost=float(cost.sum()), total_pnl=float(pnl.sum()),
                roi_pct=float(pnl.sum() / cost.sum() * 100),
                win_rate=float(outcome[trade].mean())), trade


ins_rows, oos_rows, fold_rows, bt_rows = [], [], [], []
curves = {}

for tf in TF_CFG:
    print(f"\n{'='*72}\n  [{tf}]\n{'='*72}")
    df = load_analysis_panel(tf, extra_need=BACKTEST_NEED)
    df["price"] = df["price"].clip(EPS, 1 - EPS)
    ts = pd.to_datetime(df["slot_epoch"], unit="s")
    df["month"] = ts.dt.to_period("M")
    df["day"] = ts.dt.floor("D")
    yes = df[df["token"] == "yes"].reset_index(drop=True)
    print(f"  rows {len(df):,} (YES {len(yes):,})  months {df['month'].min()}..{df['month'].max()}")

    # ---------------- (A) in-sample, YES rows ----------------
    y = yes["outcome"].values.astype(float)
    prev = None
    for mname in MODELS:
        feats = BACKTEST_MODEL_SPECS[mname]
        res = glm_fit(y, yes[feats], groups=yes["slot_epoch"].values)
        k, n, llf = len(res.params), len(y), float(res.llf)
        row = {"tf": tf, "model": mname, "N_yes": n, "k": k, "llf": llf,
               "aic": -2 * llf + 2 * k, "bic": -2 * llf + k * np.log(n),
               "logloss": float(logloss(y, res.fittedvalues).mean()),
               "brier": float(brier(y, res.fittedvalues).mean())}
        if prev is not None:
            added = [f for f in feats if f not in BACKTEST_MODEL_SPECS[prev[0]]]
            lr = 2 * (llf - prev[1])
            row.update({"vs": prev[0], "added": "+".join(added), "lr_stat": lr, "lr_df": len(added),
                        "lr_p": float(sp_stats.chi2.sf(lr, len(added))),
                        "delta_aic": row["aic"] - prev[2], "delta_bic": row["bic"] - prev[3]})
            w = res.wald_test(", ".join(f"{a} = 0" for a in added), scalar=True, use_f=False)
            row["wald_slotclust_p"] = float(w.pvalue)
        # beta=1 test with slot clustering (compare to table1/table4, episode clustering)
        row["beta_log_odds"] = float(res.params["log_odds"])
        row["beta_se_slotclust"] = float(res.bse["log_odds"])
        if "lo_x_csd" in feats:
            row["lo_x_csd"] = float(res.params["lo_x_csd"])
            row["lo_x_csd_se_slotclust"] = float(res.bse["lo_x_csd"])
            row["lo_x_csd_p_slotclust"] = float(res.pvalues["lo_x_csd"])
        ins_rows.append(row)
        prev = (mname, llf, row["aic"], row["bic"])
        print(f"  [IS] {mname:<14} llf={llf:,.1f}  AIC={row['aic']:,.1f}  BIC={row['bic']:,.1f}"
              + (f"  LR={row['lr_stat']:.1f} (p={row['lr_p']:.3g})  dBIC={row['delta_bic']:.1f}"
                 f"  Wald[slot]p={row['wald_slotclust_p']:.3g}" if "lr_stat" in row else ""))

    if INSAMPLE_ONLY:
        continue

    # ---------------- (B)+(C) walk-forward ----------------
    months = sorted(df["month"].unique())
    test_months = months[MIN_TRAIN_MONTHS:]
    preds = []
    for m in test_months:
        train, test = add_csd_rank(df[df["month"] < m], df[df["month"] == m])
        out = test[["token", "outcome", "price", "obs_epoch", "day", "month"]].copy()
        for mname in MODELS:
            feats = BACKTEST_MODEL_SPECS[mname]
            res = glm_fit(train["outcome"].values.astype(float), train[feats])
            out[mname] = glm_predict(res, test[feats])
        preds.append(out)
        yt = out[out["token"] == "yes"]
        fr = {"tf": tf, "test_month": str(m), "n_train": len(train), "n_test_yes": len(yt),
              "market_logloss": float(logloss(yt["outcome"].values, yt["price"].values).mean())}
        for mname in MODELS:
            fr[f"{mname}_logloss"] = float(logloss(yt["outcome"].values, yt[mname].values).mean())
        fold_rows.append(fr)
        print(f"  [OOS] {m}  train={len(train):,}  test={len(test):,}  "
              f"LL market={fr['market_logloss']:.5f}  "
              + "  ".join(f"{k.split('_')[0]}={fr[f'{k}_logloss']:.5f}" for k in MODELS))
    P = pd.concat(preds, ignore_index=True)

    # forecast metrics on YES rows, full sample and within the price band
    PY = P[P["token"] == "yes"]
    for subset, mask in [("all", np.ones(len(PY), bool)),
                         (f"p{PRICE_BAND[0]}-{PRICE_BAND[1]}",
                          PY["price"].between(*PRICE_BAND).values)]:
        S = PY[mask]
        yv = S["outcome"].values.astype(float)
        losses = {"market": (logloss(yv, S["price"].values), brier(yv, S["price"].values))}
        for mname in MODELS:
            losses[mname] = (logloss(yv, S[mname].values), brier(yv, S[mname].values))
        for name, (ll, br) in losses.items():
            r = {"tf": tf, "subset": subset, "model": name, "N_yes": len(S),
                 "logloss": float(ll.mean()), "brier": float(br.mean())}
            # compare each model against the market and every simpler model
            order = ["market"] + MODELS
            for ref in order[:order.index(name)]:
                dmean, t, p, nd = dm_test(losses[ref][0], ll, S["day"].values)
                r[f"ll_gain_vs_{ref}"] = dmean
                r[f"ll_gain_vs_{ref}_t"] = t
                r[f"ll_gain_vs_{ref}_p"] = p
                dmean_b, t_b, p_b, _ = dm_test(losses[ref][1], br, S["day"].values)
                r[f"brier_gain_vs_{ref}"] = dmean_b
                r[f"brier_gain_vs_{ref}_p"] = p_b
            oos_rows.append(r)
        print(f"  [OOS {subset}] N_yes={len(S):,}  LL: market={losses['market'][0].mean():.5f}  "
              + "  ".join(f"{k.split('_')[0]}={losses[k][0].mean():.5f}" for k in MODELS))

    # backtest on test months: OOS predictions vs in-sample (full-sample) fit
    test_rows = df["month"].isin(test_months).values
    D = df[test_rows]
    price, outc = D["price"].values, D["outcome"].values.astype(float)
    for mname in MODELS:
        feats = BACKTEST_MODEL_SPECS[mname]
        res_is = glm_fit(df["outcome"].values.astype(float), df[feats])
        stats_is, _ = backtest(price, outc, glm_predict(res_is, D[feats]))
        stats_oos, trade = backtest(P["price"].values, P["outcome"].values.astype(float), P[mname].values)
        bt_rows.append({"tf": tf, "model": mname, "fit": "in_sample", **stats_is})
        bt_rows.append({"tf": tf, "model": mname, "fit": "walk_forward", **stats_oos})
        if trade.any():
            order = np.argsort(P["obs_epoch"].values[trade])
            pnl = (P["outcome"].values[trade] - P["price"].values[trade] - BACKTEST_FEE)[order]
            t = pd.to_datetime(P["obs_epoch"].values[trade][order], unit="s")
            curves[(tf, mname)] = (t, np.cumsum(pnl))
        print(f"  [BT] {mname:<14} in-sample: n={stats_is['n_trades']:,} ROI={stats_is['roi_pct']:.3f}%  |  "
              f"walk-forward: n={stats_oos['n_trades']:,} ROI={stats_oos['roi_pct']:.3f}% "
              f"PnL=${stats_oos['total_pnl']:,.1f}")

pd.DataFrame(ins_rows).to_csv(os.path.join(OUT_DIR, "insample_model_comparison.csv"), index=False, encoding="utf-8-sig")
if INSAMPLE_ONLY:
    print(f"\nin-sample results saved → {OUT_DIR}")
    sys.exit(0)
pd.DataFrame(oos_rows).to_csv(os.path.join(OUT_DIR, "oos_forecast_metrics.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(fold_rows).to_csv(os.path.join(OUT_DIR, "oos_folds.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(bt_rows).to_csv(os.path.join(OUT_DIR, "oos_backtest.csv"), index=False, encoding="utf-8-sig")

# cumulative walk-forward PnL, same layout as FIG3
setup_plot_rc()
fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.1), dpi=FIG_DPI)
fig.patch.set_facecolor("#fcfcfb")
for ax, tf in zip(axes, TF_CFG):
    style_axes(ax)
    ax.axhline(0, color="#c3c2b7", linewidth=1.2, linestyle="--", zorder=1)
    for mname in MODELS:
        if (tf, mname) in curves:
            t, c = curves[(tf, mname)]
            ax.plot(t, c, color=BACKTEST_MODEL_COLORS[mname], linewidth=1.3, label=mname.split("_")[0])
    ax.set_title(f"{tf} (walk-forward)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time", fontsize=7.5)
    if ax is axes[0]:
        ax.set_ylabel("Cumulative PnL ($)", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.2)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(labelsize=7)
fig.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "FIG_oos_equity"))
print(f"\nall outputs saved → {OUT_DIR}")
