# -*- coding: utf-8 -*-
# Produces Table 1-4 and the control-candidate appendix table (output/*.csv).
# Requires panels/panel_{5m,15m,60m}.pkl (see build_panel.py).
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import statsmodels.api as sm
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


# Explanatory-power rows appended to Table 2/4: McFadden pseudo R2, its gain
# over a reference model, and a cluster-robust Wald test (by episode_id, the
# same clustering as the coefficient SEs) that the added terms are zero. The
# Wald test has the same null as an LR test but stays valid under
# within-episode dependence, which LR ignores.
def fit_row(res, feats, ref_id, ref, llf0):
    r2 = 1 - float(res.llf) / llf0
    row = {"pseudo_r2": r2, "ref": ref_id}
    if ref is None:
        return row
    ref_res, ref_feats = ref
    added = [f for f in feats if f not in ref_feats]
    w = res.wald_test(", ".join(f"{a} = 0" for a in added), scalar=True, use_f=False)
    row.update({"added": "+".join(added),
                "d_pseudo_r2": r2 - (1 - float(ref_res.llf) / llf0),
                "wald_chi2": float(w.statistic), "wald_df": len(added),
                "wald_p": float(w.pvalue), "wald_sig": sig(float(w.pvalue))})
    return row


t1_rows, t2_rows, t3_rows, t4_rows, tA_rows = [], [], [], [], []

for tf in TF_CFG:
    print(f"\n{'='*70}\n  [{tf}] preparing data\n{'='*70}")
    df = load_analysis_panel(tf)
    y  = df["outcome"].values.astype(float)
    ep = df["episode_id"].values
    n_obs, n_ep = len(df), int(df["episode_id"].nunique())
    print(f"  analysis sample: {n_obs:,} rows  {n_ep:,} episodes")
    llf0 = float(sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial()).fit().llf)

    # Table 1: logit(Y) = a + b*log_odds, H0: b=1
    res = fit(y, ep, df[["log_odds"]])
    b, b_se, _  = coef_se_p(res, "log_odds", null=0.0)
    a, a_se, _  = coef_se_p(res, "const", null=0.0)
    _, _, p_b1  = coef_se_p(res, "log_odds", null=1.0)
    t1_rows.append({"tf": tf, "alpha": a, "alpha_se": a_se,
                     "beta": b, "beta_se": b_se, "p_H0_beta_eq_1": p_b1,
                     "sig": sig(p_b1), "N": n_obs, "episodes": n_ep})
    print(f"  Table1  alpha={a:.4f}  beta={b:.4f}{sig(p_b1)}  (H0:beta=1, p={p_b1:.4g})")

    # Table 2: base model = log_odds + delta_logit + lo_x_ttm, built up one
    # control at a time so each step's explanatory-power gain is reported.
    feats2 = ["log_odds", "delta_logit", "lo_x_ttm"]
    t2_steps = [("price", ["log_odds"], None),
                ("+delta", ["log_odds", "delta_logit"], "price"),
                ("base", feats2, "+delta")]
    fits = {"price": (res, ["log_odds"])}
    for mid, feats, ref_id in t2_steps:
        res_ = res if mid == "price" else fit(y, ep, df[feats])
        fits[mid] = (res_, feats)
        row = model_row(res_, feats2, tf, mid, n_obs, n_ep)
        row.update(fit_row(res_, feats, ref_id, fits.get(ref_id), llf0))
        t2_rows.append(row)
    res2 = fits["base"][0]
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
    # reference = same model without the CSD terms; (1)/(5) vs price only
    csd_refs = {1: "price", 2: 1, 3: 1, 4: 1, 5: "price", 6: 5, 7: 5, 8: 5}
    for mid, feats in csd_specs.items():
        res_ = fit(y, ep, df[feats])
        fits[mid] = (res_, feats)
        ref_id = csd_refs[mid]
        row = model_row(res_, all_csd_feats, tf, mid, n_obs, n_ep)
        row.update(fit_row(res_, feats, ref_id, fits[ref_id], llf0))
        t4_rows.append(row)
    print(f"  Table4  (1)-(8) done")

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
    # reference = same model without the candidate term
    ctrl_refs = {1: None, 2: 1, 3: 1, 4: 2, 5: 2, 6: 5, 7: 5}
    ctrl_fits = {}
    for mid, feats in ctrl_specs.items():
        res_ = fit(y, ep, df[feats])
        ctrl_fits[mid] = (res_, feats)
        ref_id = ctrl_refs[mid]
        row = model_row(res_, all_ctrl_feats, tf, mid, n_obs, n_ep)
        row.update(fit_row(res_, feats, ref_id, ctrl_fits.get(ref_id), llf0))
        tA_rows.append(row)
    print(f"  Appendix (1)-(7) done")

pd.DataFrame(t1_rows).to_csv(os.path.join(OUT_DIR, "table1_baseline.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(t2_rows).to_csv(os.path.join(OUT_DIR, "table2_base_model.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(t3_rows).to_csv(os.path.join(OUT_DIR, "table3_correlation.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(t4_rows).to_csv(os.path.join(OUT_DIR, "table4_csd_models.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(tA_rows).to_csv(os.path.join(OUT_DIR, "tableA1_control_candidates.csv"), index=False, encoding="utf-8-sig")

# Paper-layout Table 2/4 (markdown): one panel per horizon, columns = models,
# coefficients with stars (vs. Null) and SE in parentheses, then the
# explanatory-power rows (pseudo R2, gain over the reference model, Wald).
VAR_LABEL = {"log_odds": "ln(p/(1-p))", "delta_logit": "Δ ln(p/(1-p))",
             "lo_x_ttm": "ln(p/(1-p)) × TTM", "lo_x_csd": "ln(p/(1-p)) × CSD",
             "csd_q_c": "CSD", "lo_x_delta": "ln(p/(1-p)) × Δ ln(p/(1-p))",
             "ttm_c": "TTM", "poly_vol_log": "log(vol)",
             "lo_x_vol": "ln(p/(1-p)) × log(vol)", "const": "Intercept"}


def model_label(m) -> str:
    if isinstance(m, float) and m.is_integer():  # ref column turns float when it holds None
        m = int(m)
    return f"({m})" if str(m).isdigit() else str(m)


def paper_table_md(tab: pd.DataFrame, feats: list[str], title: str) -> str:
    models = list(tab["model"].drop_duplicates())
    lines = [f"**{title}**"]
    for i, tf in enumerate(TF_CFG):
        T = {m: tab[(tab.tf == tf) & (tab.model == m)].iloc[0] for m in models}
        lines += ["", f"*Panel {chr(65 + i)}: {tf}*", "",
                  "| | Null | " + " | ".join(model_label(m) for m in models) + " |",
                  "|---|---|" + "---|" * len(models)]
        for v in feats + ["const"]:
            coef, se = [], []
            for m in models:
                c = T[m][f"{v}_coef"]
                stars = T[m][f"{v}_sig"] if isinstance(T[m][f"{v}_sig"], str) else ""
                coef.append("" if pd.isna(c) else f"{c:.4f}{stars}")
                se.append("" if pd.isna(c) else f"({T[m][f'{v}_se']:.4f})")
            null = "1" if v == "log_odds" else "0"
            lines.append(f"| {VAR_LABEL[v]} | {null} | " + " | ".join(coef) + " |")
            lines.append("| | | " + " | ".join(se) + " |")
        has_ref = {m: not pd.isna(T[m].get("wald_chi2")) for m in models}
        lines.append("| N | | " + " | ".join(f"{T[m]['N']:,}" for m in models) + " |")
        lines.append("| Pseudo R² | | " + " | ".join(f"{T[m]['pseudo_r2']:.5f}" for m in models) + " |")
        lines.append("| ΔR² ×10⁴ | | " + " | ".join(
            f"{T[m]['d_pseudo_r2'] * 1e4:+.2f}" if has_ref[m] else "" for m in models) + " |")
        lines.append("| Wald χ² (df) | | " + " | ".join(
            f"{T[m]['wald_chi2']:.1f}{T[m]['wald_sig']} ({int(T[m]['wald_df'])})" if has_ref[m] else ""
            for m in models) + " |")
        lines.append("| vs. | | " + " | ".join(
            model_label(T[m]["ref"]) if has_ref[m] else "" for m in models) + " |")
    lines += ["", "Stars on coefficients test against the Null column (1 for ln(p/(1-p)), 0 otherwise); "
              "SEs cluster-robust by episode. Pseudo R² = McFadden. ΔR² and Wald χ² are relative to the "
              "model in the \"vs.\" row; Wald χ² is the cluster-robust (episode) joint test that the "
              "added terms are zero. *** p<0.001, ** p<0.01, * p<0.05, . p<0.1"]
    return "\n".join(lines) + "\n"


for fname, rows_, feats_, title in [
        ("table2_base_model.md", t2_rows, ["log_odds", "delta_logit", "lo_x_ttm"],
         "Table 2. Base model, adding controls one at a time"),
        ("table4_csd_models.md", t4_rows, ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd", "csd_q_c"],
         "Table 4. CSD-augmented models"),
        ("tableA1_control_candidates.md", tA_rows,
         ["log_odds", "delta_logit", "lo_x_delta", "ttm_c", "lo_x_ttm", "poly_vol_log", "lo_x_vol"],
         "Table A1. Control-variable candidates")]:
    md = paper_table_md(pd.DataFrame(rows_), feats_, title)
    with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)

print(f"\nall tables saved → {OUT_DIR}")
