# -*- coding: utf-8 -*-
# Table 6: variance inflation factors of the regressors in Eq. (11), the
# CSD-augmented model (logit, Δlogit, logit×ttm, logit×CSD), by horizon.
#
#   Unweighted: VIF_k = 1 / (1 - R_k^2), R_k^2 from an OLS regression of
#     regressor k on the other regressors and a constant
#     (statsmodels variance_inflation_factor on [const, X]).
#   Weighted: every column of [const, X] is multiplied by sqrt(p(1-p)) before
#     the same computation, which matches the logit information matrix
#     X'DX with D = diag{p(1-p)}. p is the fitted probability from Eq. (12)
#     (Eq. (11) plus the CSD level term); its coefficient on the CSD level
#     term is essentially zero, so the fitted values match those of Eq. (11).
#
# Same sample as Table 5 (common.load_analysis_panel). Output:
# results/table6_vif.csv.
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis"))

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import TF_CFG, OUT_DIR, fit, load_analysis_panel

EQ11 = ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd"]
EQ12 = EQ11 + ["csd_q_c"]


def vifs(X: np.ndarray) -> list[float]:
    """VIF of every column except column 0 (the constant or weighted constant)."""
    return [float(variance_inflation_factor(X, j)) for j in range(1, X.shape[1])]


rows = []
for tf in TF_CFG:
    df = load_analysis_panel(tf)
    res = fit(df["outcome"].values.astype(float), df["slot_epoch"].values, df[EQ12])
    p_hat = np.asarray(res.fittedvalues)

    X = np.column_stack([np.ones(len(df))] + [df[f].values.astype(float) for f in EQ11])
    w = np.sqrt(p_hat * (1 - p_hat))[:, None]
    for var, vu, vw in zip(EQ11, vifs(X), vifs(X * w)):
        rows.append({"tf": tf, "variable": var, "vif_unweighted": vu, "vif_weighted": vw, "N": len(df)})
    print(f"[{tf}] N={len(df):,}")

vif = pd.DataFrame(rows)
vif.to_csv(os.path.join(OUT_DIR, "table6_vif.csv"), index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print("\n" + vif.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nsaved → {OUT_DIR}")
