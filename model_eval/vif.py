# -*- coding: utf-8 -*-
# Multicollinearity diagnostics for the CSD-augmented models of Table 4.
#
#   Eq. c4 = Table 4 model (4): log_odds, delta_logit, lo_x_ttm, lo_x_csd, csd_q_c
#   Eq. c3 = Table 4 model (3): log_odds, delta_logit, lo_x_ttm, lo_x_csd
#
# lo_x_ttm = log_odds * ttm_c and lo_x_csd = log_odds * csd_q_c, exactly as
# built by common.load_analysis_panel(); the sample is that function's
# default (the one build_paper_tables.py uses for Table 4), YES and NO rows.
#
# (A) unweighted VIF: statsmodels variance_inflation_factor on [const, X].
# (B) GLM-weighted VIF: Eq. c4 is fit as in Table 4 (slot-clustered logit;
#     clustering does not affect fitted values), every column including the
#     constant is multiplied by sqrt(p_hat (1 - p_hat)), then VIF as in (A).
#     The same Eq. c4 p_hat is used for both c3 and c4.
#
# Outputs: results/table6_vif.csv (manuscript Table 6) and
# model_eval/results/interaction_corr.csv.
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis"))

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import TF_CFG, fit, load_analysis_panel

OUT_DIR = os.path.join(HERE, "results")
PAPER_DIR = os.path.join(os.path.dirname(HERE), "results")  # manuscript tables
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_N = {"5m": 777_720, "15m": 2_216_136, "60m": 3_675_344}
MODELS = {
    "c4": ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd", "csd_q_c"],
    "c3": ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd"],
}
SAMPLE_N, SEED = 500_000, 42


def vifs(X: np.ndarray) -> list[float]:
    """VIF of every column except column 0 (the constant or weighted constant)."""
    return [float(variance_inflation_factor(X, j)) for j in range(1, X.shape[1])]


vif_rows, corr_rows = [], []
for tf in TF_CFG:
    df = load_analysis_panel(tf)
    if len(df) != EXPECTED_N[tf]:
        sys.exit(f"[{tf}] N={len(df):,} != expected {EXPECTED_N[tf]:,}; stopping.")

    # p_hat from Eq. c4, fitted on the full sample exactly as in Table 4
    res = fit(df["outcome"].values.astype(float), df["slot_epoch"].values, df[MODELS["c4"]])
    p_hat = np.asarray(res.fittedvalues)

    sampled = False
    try:
        idx = np.arange(len(df))
        for model, feats in MODELS.items():
            X = np.column_stack([np.ones(len(df))] + [df[f].values.astype(float) for f in feats])
            w = np.sqrt(p_hat * (1 - p_hat))[:, None]
            for var, vu, vw in zip(feats, vifs(X), vifs(X * w)):
                vif_rows.append({"tf": tf, "model": model, "variable": var, "vif_unweighted": vu,
                                 "vif_weighted": vw, "n_used": len(idx), "sampled": sampled})
    except MemoryError:
        sampled = True
        idx = np.random.default_rng(SEED).choice(len(df), SAMPLE_N, replace=False)
        vif_rows = [r for r in vif_rows if r["tf"] != tf]
        for model, feats in MODELS.items():
            X = np.column_stack([np.ones(len(idx))] + [df[f].values[idx].astype(float) for f in feats])
            w = np.sqrt(p_hat[idx] * (1 - p_hat[idx]))[:, None]
            for var, vu, vw in zip(feats, vifs(X), vifs(X * w)):
                vif_rows.append({"tf": tf, "model": model, "variable": var, "vif_unweighted": vu,
                                 "vif_weighted": vw, "n_used": len(idx), "sampled": sampled})

    r = float(np.corrcoef(df["lo_x_ttm"].values, df["lo_x_csd"].values)[0, 1])
    corr_rows.append({"tf": tf, "corr_lo_ttm_lo_csd": r})
    print(f"[{tf}] N={len(df):,}  sampled={sampled}  corr(lo_x_ttm, lo_x_csd)={r:+.4f}")

vif = pd.DataFrame(vif_rows)
vif.to_csv(os.path.join(PAPER_DIR, "table6_vif.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "interaction_corr.csv"), index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print("\n" + vif.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nsaved → {OUT_DIR}")
