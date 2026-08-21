# -*- coding: utf-8 -*-
# Produces Table 1-4 and the control-candidate appendix table (output/*.csv).
# Requires panels/panel_{5m,15m,60m}.pkl (see build_panel.py).
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from scipy import stats as sp_stats

from common import TF_CFG, OUT_DIR, sig, fit, coef_se_p, load_analysis_panel


def model_row(res, feat_names, tf, model_id, n_obs, n_ep):
    row = {"tf": tf, "model": model_id, "N": n_obs, "episodes": n_ep}
    for name in feat_names + ["const"]:
        null = 1.0 if name == "log_odds" else 0.0
        c, se, p = coef_se_p(res, name, null=null)
        row[f"{name}_coef"] = c
        row[f"{name}_se"] = se
        row[f"{name}_p"] = p
        row[f"{name}_sig"] = sig(p) if c is not None else ""
    return row


t1_rows, t2_rows, t3_rows, t4_rows, tA_rows = [], [], [], [], []

for tf in TF_CFG:
    print(f"\n{'='*70}\n  [{tf}] 데이터 준비\n{'='*70}")
    df = load_analysis_panel(tf)
    y  = df["outcome"].values.astype(float)
    ep = df["episode_id"].values
    n_obs, n_ep = len(df), int(df["episode_id"].nunique())
    print(f"  분석표본: {n_obs:,}행  {n_ep:,}에피소드")

    # Table 1: logit(Y) = a + b*log_odds, H0: b=1
    res = fit(y, ep, df[["log_odds"]])
    b, b_se, _  = coef_se_p(res, "log_odds", null=0.0)
    a, a_se, _  = coef_se_p(res, "const", null=0.0)
    _, _, p_b1  = coef_se_p(res, "log_odds", null=1.0)
    t1_rows.append({"tf": tf, "alpha": a, "alpha_se": a_se,
                     "beta": b, "beta_se": b_se, "p_H0_beta_eq_1": p_b1,
                     "sig": sig(p_b1), "N": n_obs, "episodes": n_ep})
    print(f"  Table1  alpha={a:.4f}  beta={b:.4f}{sig(p_b1)}  (H0:beta=1, p={p_b1:.4g})")

    # Table 2: base model = log_odds + delta_logit + lo_x_ttm
    feats2 = ["log_odds", "delta_logit", "lo_x_ttm"]
    res2 = fit(y, ep, df[feats2])
    t2_rows.append(model_row(res2, feats2, tf, "base", n_obs, n_ep))
    print(f"  Table2  log_odds={res2.params['log_odds']:.4f}  "
          f"delta={res2.params['delta_logit']:.4f}  "
          f"lo_x_ttm={res2.params['lo_x_ttm']:.4f}")

    # Table 3: correlation matrix (CSD column is csd_q_c throughout)
    corr_vars = {"log_odds": df["log_odds"], "delta_logit": df["delta_logit"],
                 "ttm": df["ttm"].astype(float), "csd": df["csd_q_c"]}
    names = list(corr_vars.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r, p = sp_stats.pearsonr(corr_vars[names[i]], corr_vars[names[j]])
            t3_rows.append({"tf": tf, "var1": names[i], "var2": names[j],
                             "r": r, "p": p, "sig": sig(p), "N": n_obs})

    # Table 4: CSD-augmented models (1)-(8)
    csd_specs = {
        1: ["log_odds", "delta_logit", "lo_x_ttm"],
        2: ["log_odds", "delta_logit", "lo_x_ttm", "csd_q_c"],
        3: ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd"],
        4: ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd", "csd_q_c"],
        5: ["log_odds", "delta_logit"],
        6: ["log_odds", "delta_logit", "csd_q_c"],
        7: ["log_odds", "delta_logit", "lo_x_csd"],
        8: ["log_odds", "delta_logit", "lo_x_csd", "csd_q_c"],
    }
    all_csd_feats = ["log_odds", "lo_x_ttm", "lo_x_csd", "delta_logit", "csd_q_c"]
    for mid, feats in csd_specs.items():
        res_ = fit(y, ep, df[feats])
        t4_rows.append(model_row(res_, all_csd_feats, tf, mid, n_obs, n_ep))
    print(f"  Table4  (1)~(8) 완료")

    # Appendix: control-variable candidates (1)-(7)
    ctrl_specs = {
        1: ["log_odds"],
        2: ["log_odds", "delta_logit"],
        3: ["log_odds", "lo_x_delta"],
        4: ["log_odds", "delta_logit", "ttm_c"],
        5: ["log_odds", "delta_logit", "lo_x_ttm"],
        6: ["log_odds", "delta_logit", "lo_x_ttm", "poly_vol_log"],
        7: ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_vol"],
    }
    all_ctrl_feats = ["log_odds", "delta_logit", "lo_x_delta", "ttm_c",
                       "lo_x_ttm", "poly_vol_log", "lo_x_vol"]
    for mid, feats in ctrl_specs.items():
        res_ = fit(y, ep, df[feats])
        tA_rows.append(model_row(res_, all_ctrl_feats, tf, mid, n_obs, n_ep))
    print(f"  Appendix (1)~(7) 완료")

pd.DataFrame(t1_rows).to_csv(os.path.join(OUT_DIR, "table1_baseline.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(t2_rows).to_csv(os.path.join(OUT_DIR, "table2_base_model.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(t3_rows).to_csv(os.path.join(OUT_DIR, "table3_correlation.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(t4_rows).to_csv(os.path.join(OUT_DIR, "table4_csd_models.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(tA_rows).to_csv(os.path.join(OUT_DIR, "tableA1_control_candidates.csv"), index=False, encoding="utf-8-sig")

print(f"\n모든 표 저장 완료 → {OUT_DIR}")
