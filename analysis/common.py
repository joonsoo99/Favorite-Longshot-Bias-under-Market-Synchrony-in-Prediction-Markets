# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # poly_v2/
PNL_DIR = os.path.join(ROOT, "panels")
OUT_DIR = os.path.join(ROOT, "results")  # manuscript tables and figures
os.makedirs(OUT_DIR, exist_ok=True)

TF_CFG = {"5m": "panel_5m.pkl", "15m": "panel_15m.pkl", "60m": "panel_60m.pkl"}

TF_COLORS  = {"5m": "#2a78d6", "15m": "#eb6834", "60m": "#1baf7a"}
TF_MARKERS = {"5m": "o", "15m": "s", "60m": "^"}

# Backtest models (Table 7, Figure 2): M1 = Eq. (1) basic calibration,
# M2 = Eq. (7) control-augmented, M3 = Eq. (11) CSD-augmented.
BACKTEST_MODEL_SPECS = {
    "M1_price_only": ["log_odds"],
    "M2_baseline":   ["log_odds", "delta_logit", "lo_x_ttm"],
    "M3_plus_csd":   ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd"],
}
BACKTEST_MODEL_COLORS = {"M1_price_only": "#2a78d6", "M2_baseline": "#eb6834", "M3_plus_csd": "#1baf7a"}

BACKTEST_FEE         = 0.015
BACKTEST_EDGE_THRESH = 0.015


def sig(p) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.1 else "")))


def fit(y, groups, Xdf: pd.DataFrame):
    """Cluster-robust logistic GLM. Callers pass slot_epoch as `groups`: all 4
    assets in a time window share CSD and common shocks, so episodes in the
    same slot are not independent and must sit in one cluster."""
    Xc = sm.add_constant(Xdf.astype(float), has_constant="add")
    m = sm.GLM(y, Xc, family=sm.families.Binomial())
    return m.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)}, disp=False)


def coef_se_p(res, name: str, null: float = 0.0):
    """Coefficient, SE, two-sided p-value against H0=null. (None,None,None) if not in the model."""
    if name not in res.params.index:
        return None, None, None
    c = float(res.params[name])
    se = float(res.bse[name])
    z = (c - null) / se if se > 0 else np.nan
    p = float(2 * sp_stats.norm.sf(abs(z)))
    return c, se, p


def load_analysis_panel(tf: str) -> pd.DataFrame:
    """Load panels/panel_{tf}.pkl, which build_panel.py has already reduced to
    the analysis sample (paper Section 2), and add the analysis variables. The
    centering (ttm_c: ttm minus its median; csd_q_c: percentile rank of the
    raw CSD minus 0.5) is computed on this sample. Every minute in the sample
    has the same number of rows (4 assets x 2 contracts), so ranking rows is
    equivalent to ranking minutes."""
    df = pd.read_pickle(os.path.join(PNL_DIR, TF_CFG[tf]))
    df["poly_vol_log"] = np.log1p(df["poly_vol_prev"].clip(lower=0))
    df["ttm_c"]        = df["ttm"] - df["ttm"].median()
    df["lo_x_ttm"]     = df["log_odds"] * df["ttm_c"]
    df["lo_x_delta"]   = df["log_odds"] * df["delta_logit"]
    df["lo_x_vol"]     = df["log_odds"] * df["poly_vol_log"]
    df["csd_q_c"]      = df["csd_raw"].rank(pct=True) - 0.5
    df["lo_x_csd"]     = df["log_odds"] * df["csd_q_c"]
    df["price"]        = 1.0 / (1.0 + np.exp(-df["log_odds"]))
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
