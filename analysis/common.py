# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # poly_v2/
PNL_DIR = os.path.join(ROOT, "panels")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TF_CFG = {"5m": "panel_5m.pkl", "15m": "panel_15m.pkl", "60m": "panel_60m.pkl"}

ASSET_STD = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL",
    "xrp": "XRP",
}
ASSETS = ["BTC", "ETH", "SOL", "XRP"]

TF_COLORS  = {"5m": "#2a78d6", "15m": "#eb6834", "60m": "#1baf7a"}
TF_MARKERS = {"5m": "o", "15m": "s", "60m": "^"}

BACKTEST_MODEL_SPECS = {
    "M1_price_only": ["log_odds"],
    "M2_baseline":   ["log_odds", "delta_logit", "lo_x_ttm"],
    "M3_plus_csd":   ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd"],
}
BACKTEST_MODEL_LABELS = {
    "M1_price_only": "M1: price only",
    "M2_baseline":   "M2: baseline",
    "M3_plus_csd":   "M3: + CSD",
}
BACKTEST_MODEL_COLORS = {"M1_price_only": "#2a78d6", "M2_baseline": "#eb6834", "M3_plus_csd": "#1baf7a"}

BACKTEST_FEE         = 0.015
BACKTEST_EDGE_THRESH = 0.015
BACKTEST_NEED = ["log_odds", "delta_logit", "ttm_c", "lo_x_ttm", "csd_raw", "csd_q_c", "lo_x_csd"]


def sig(p) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.1 else "")))


# CSD = cross-sectional dispersion: std(log_odds) across the 4 assets at the
# same obs_epoch (YES token only). Only obs_epochs where all 4 assets are
# observed are kept (build_panel.py already enforces this upstream, so
# dropna(how="any") here is a no-op safety net, not the primary filter).
def compute_csd(panel: pd.DataFrame) -> pd.DataFrame:
    yes = panel[panel["token"] == "yes"].copy()
    yes["asset_std"] = yes["asset"].map(ASSET_STD)
    cs = (yes.groupby(["obs_epoch", "asset_std"])["log_odds"]
             .mean()
             .unstack("asset_std"))
    cs = cs[[a for a in ASSETS if a in cs.columns]]
    cs_full = cs.dropna(how="any")
    csd = cs_full.std(axis=1)
    out = csd.reset_index()
    out.columns = ["obs_epoch", "csd_raw"]
    return out


def fit(y, ep_ids, Xdf: pd.DataFrame):
    """Cluster-robust (by episode_id) logistic GLM."""
    Xc = sm.add_constant(Xdf.astype(float), has_constant="add")
    m = sm.GLM(y, Xc, family=sm.families.Binomial())
    return m.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(ep_ids)}, disp=False)


def coef_se_p(res, name: str, null: float = 0.0):
    """Coefficient, SE, two-sided p-value against H0=null. (None,None,None) if not in the model."""
    if name not in res.params.index:
        return None, None, None
    c = float(res.params[name])
    se = float(res.bse[name])
    z = (c - null) / se if se > 0 else np.nan
    p = float(2 * sp_stats.norm.sf(abs(z)))
    return c, se, p


def load_analysis_panel(tf: str, extra_need: list[str] | None = None) -> pd.DataFrame:
    """Load panel_{tf}.pkl, attach derived features, and drop rows with any
    missing value among `extra_need` (or the full derived-feature list if
    not given). delta_logit is used as-is from the panel pkl (computed once
    in build_panel.py) rather than recomputed here, to avoid two copies of
    the same logic drifting out of sync."""
    panel = pd.read_pickle(os.path.join(PNL_DIR, TF_CFG[tf]))

    panel["poly_vol_log"] = np.log1p(panel["poly_vol_prev"].clip(lower=0))
    panel["ttm_c"]        = panel["ttm"] - panel["ttm"].median()
    panel["lo_x_ttm"]     = panel["log_odds"] * panel["ttm_c"]
    panel["lo_x_delta"]   = panel["log_odds"] * panel["delta_logit"]
    panel["lo_x_vol"]     = panel["log_odds"] * panel["poly_vol_log"]

    csd_df = compute_csd(panel)
    panel  = panel.merge(csd_df, on="obs_epoch", how="left")
    panel["csd_q_c"]  = panel["csd_raw"].rank(pct=True) - 0.5
    panel["lo_x_csd"] = panel["log_odds"] * panel["csd_q_c"]

    panel["price"] = 1.0 / (1.0 + np.exp(-panel["log_odds"]))

    need = extra_need if extra_need is not None else [
        "log_odds", "delta_logit", "ttm_c", "lo_x_ttm", "lo_x_delta",
        "poly_vol_log", "lo_x_vol", "csd_raw", "csd_q_c", "lo_x_csd",
    ]
    df = panel.dropna(subset=need).reset_index(drop=True)
    return df


# Backtest rule: buy 1 share of a token whenever the model's fitted
# probability exceeds the market price by at least edge_thresh; buy price =
# market price + fee.
def trade_mask(res, price, edge_thresh: float = BACKTEST_EDGE_THRESH):
    return (res.fittedvalues - price) >= edge_thresh


def trade_cost_pnl(trade, price, outcome, fee: float = BACKTEST_FEE):
    cost = price[trade] + fee
    pnl = outcome[trade] - cost
    return cost, pnl


FIG_DPI = 600  # PNG preview only; submission copy is the vector PDF (dpi-independent)


# Embeds real TrueType outlines in the PDF (Type 42, not the bitmap Type 3
# default) and switches to an Elsevier-approved font, so text stays crisp
# and isn't silently substituted at print time.
def setup_plot_rc() -> None:
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Arial"


def save_fig(fig, path_no_ext: str) -> None:
    fig.savefig(f"{path_no_ext}.png", facecolor=fig.get_facecolor(), dpi=FIG_DPI)
    fig.savefig(f"{path_no_ext}.pdf", facecolor=fig.get_facecolor())


def style_axes(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
