# -*- coding: utf-8 -*-
# Produces Table 1-4. Requires panels/panel_{5m,15m,60m}.pkl (see build_panel.py).
#
# Tables 1, 2 and 4 are written twice, once per clustering of the standard
# errors (coefficients are identical; only SEs, p-values, stars and Wald
# tests differ):
#   output/slot/     cluster = time window (slot_epoch)  -- main specification:
#                    all 4 assets in a slot share CSD and common shocks, so
#                    episodes in the same slot are not independent
#   output/episode/  cluster = episode (asset x slot)    -- robustness
# Table 3 (correlations) does not depend on clustering: output/table3_correlation.csv.
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from common import TF_CFG, OUT_DIR, sig, fit, coef_se_p, load_analysis_panel

# name -> (panel column, label used in table footnotes)
CLUSTERS = {
    "slot":    ("slot_epoch", "time window (slot)"),
    "episode": ("episode_id", "episode"),
}


def model_row(res, feat_names, tf, model_id, n_obs, n_ep, n_slots):
    row = {"tf": tf, "model": model_id, "N": n_obs, "episodes": n_ep, "slots": n_slots}
    for name in feat_names + ["const"]:
        null = 1.0 if name == "log_odds" else 0.0
        c, se, p = coef_se_p(res, name, null=null)
        row[f"{name}_coef"] = c
        row[f"{name}_se"] = se
        row[f"{name}_p"] = p
        row[f"{name}_sig"] = sig(p) if c is not None else ""
    return row


# Explanatory-power columns for Table 2/4: McFadden pseudo R2, its gain over a
# reference model, and a cluster-robust Wald test (same clustering as the
# coefficient SEs) that the added terms are zero. The Wald test has the same
# null as an LR test but stays valid under within-cluster dependence, which
# LR ignores.
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


# Table 2: control-variable candidates (1)-(7); (5) is the base model.
# Reference = same model without the candidate term.
CTRL_SPECS = {
    1: ["log_odds"],
    2: ["log_odds", "delta_logit"],
    3: ["log_odds", "lo_x_delta"],
    4: ["log_odds", "delta_logit", "ttm_c"],
    5: ["log_odds", "delta_logit", "lo_x_ttm"],
    6: ["log_odds", "delta_logit", "lo_x_ttm", "poly_vol_log"],
    7: ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_vol"],
}
CTRL_REFS = {1: None, 2: 1, 3: 1, 4: 2, 5: 2, 6: 5, 7: 5}
CTRL_FEATS = ["log_odds", "delta_logit", "lo_x_delta", "ttm_c", "lo_x_ttm", "poly_vol_log", "lo_x_vol"]

# Table 4: CSD-augmented models (1)-(8).
# Reference = same model without the CSD terms; (1)/(5) vs price only.
CSD_SPECS = {
    1: ["log_odds", "delta_logit", "lo_x_ttm"],
    2: ["log_odds", "delta_logit", "lo_x_ttm", "csd_q_c"],
    3: ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd"],
    4: ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd", "csd_q_c"],
    5: ["log_odds", "delta_logit"],
    6: ["log_odds", "delta_logit", "csd_q_c"],
    7: ["log_odds", "delta_logit", "lo_x_csd"],
    8: ["log_odds", "delta_logit", "lo_x_csd", "csd_q_c"],
}
CSD_REFS = {1: "price", 2: 1, 3: 1, 4: 1, 5: "price", 6: 5, 7: 5, 8: 5}
CSD_FEATS = ["log_odds", "lo_x_ttm", "lo_x_csd", "delta_logit", "csd_q_c"]


rows = {c: {"t1": [], "t2": [], "t4": []} for c in CLUSTERS}
t3_rows = []

for tf in TF_CFG:
    print(f"\n{'='*70}\n  [{tf}] preparing data\n{'='*70}")
    df = load_analysis_panel(tf)
    y  = df["outcome"].values.astype(float)
    n_obs, n_ep = len(df), int(df["episode_id"].nunique())
    n_slots = int(df["slot_epoch"].nunique())
    print(f"  analysis sample: {n_obs:,} rows  {n_ep:,} episodes  {n_slots:,} slots")
    llf0 = float(sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial()).fit().llf)

    # Table 3: correlation matrix (CSD column is csd_q_c throughout)
    corr_vars = {"log_odds": df["log_odds"], "delta_logit": df["delta_logit"],
                 "ttm": df["ttm"].astype(float), "csd": df["csd_q_c"]}
    names = list(corr_vars.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r, p = sp_stats.pearsonr(corr_vars[names[i]], corr_vars[names[j]])
            t3_rows.append({"tf": tf, "var1": names[i], "var2": names[j],
                             "r": r, "p": p, "sig": sig(p), "N": n_obs})

    for cname, (gcol, _) in CLUSTERS.items():
        groups = df[gcol].values
        R = rows[cname]

        # Table 2 (fitted first: its model (1) is Table 1's price-only model)
        fits = {}
        for mid, feats in CTRL_SPECS.items():
            res_ = fit(y, groups, df[feats])
            fits[mid] = (res_, feats)
            ref_id = CTRL_REFS[mid]
            row = model_row(res_, CTRL_FEATS, tf, mid, n_obs, n_ep, n_slots)
            row.update(fit_row(res_, feats, ref_id, fits.get(ref_id), llf0))
            R["t2"].append(row)

        # Table 1: logit(Y) = a + b*log_odds, H0: a=0 and H0: b=1
        # ("sig" is the beta=1 test; alpha_sig is the alpha=0 test)
        res = fits[1][0]
        b, b_se, _  = coef_se_p(res, "log_odds", null=0.0)
        a, a_se, p_a0 = coef_se_p(res, "const", null=0.0)
        _, _, p_b1  = coef_se_p(res, "log_odds", null=1.0)
        R["t1"].append({"tf": tf, "alpha": a, "alpha_se": a_se,
                        "alpha_p": p_a0, "alpha_sig": sig(p_a0),
                        "beta": b, "beta_se": b_se, "p_H0_beta_eq_1": p_b1,
                        "sig": sig(p_b1), "N": n_obs, "episodes": n_ep, "slots": n_slots})

        # Table 4
        csd_fits = {"price": fits[1]}
        for mid, feats in CSD_SPECS.items():
            res_ = fit(y, groups, df[feats])
            csd_fits[mid] = (res_, feats)
            ref_id = CSD_REFS[mid]
            row = model_row(res_, CSD_FEATS, tf, mid, n_obs, n_ep, n_slots)
            row.update(fit_row(res_, feats, ref_id, csd_fits[ref_id], llf0))
            R["t4"].append(row)

        print(f"  [{cname:7s}] Table1 alpha={a:.4f}{sig(p_a0)}  beta={b:.4f}{sig(p_b1)}"
              f" (H0:beta=1, p={p_b1:.4g})  | Table 2 (1)-(7), Table 4 (1)-(8) done")


# Paper-layout Table 2/4 (markdown): one panel per horizon, columns = models,
# coefficients with stars (vs. Null) and SE in parentheses, then the
# explanatory-power rows (pseudo R2, gain over the reference model, Wald).
VAR_LABEL = {"log_odds": "ln(p/(1-p))", "delta_logit": "Δ ln(p/(1-p))",
             "lo_x_ttm": "ln(p/(1-p)) × TTM", "lo_x_csd": "ln(p/(1-p)) × CSD",
             "csd_q_c": "CSD", "lo_x_delta": "ln(p/(1-p)) × Δ ln(p/(1-p))",
             "ttm_c": "TTM", "poly_vol_log": "ln(1+vol)",
             "lo_x_vol": "ln(p/(1-p)) × ln(1+vol)", "const": "Intercept"}


def model_label(m) -> str:
    if isinstance(m, float) and m.is_integer():  # ref column turns float when it holds None
        m = int(m)
    return f"({m})" if str(m).isdigit() else str(m)


def paper_table_md(tab: pd.DataFrame, feats: list[str], title: str, cname: str) -> str:
    _, glabel = CLUSTERS[cname]
    count_col = "slots" if cname == "slot" else "episodes"
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
        lines.append(f"| Clusters ({count_col}) | | " + " | ".join(
            f"{T[m][count_col]:,}" for m in models) + " |")
        lines.append("| Pseudo R² | | " + " | ".join(f"{T[m]['pseudo_r2']:.5f}" for m in models) + " |")
        lines.append("| ΔR² ×10⁴ | | " + " | ".join(
            f"{T[m]['d_pseudo_r2'] * 1e4:+.2f}" if has_ref[m] else "" for m in models) + " |")
        lines.append("| Wald χ² (df) | | " + " | ".join(
            f"{T[m]['wald_chi2']:.1f}{T[m]['wald_sig']} ({int(T[m]['wald_df'])})" if has_ref[m] else ""
            for m in models) + " |")
        lines.append("| vs. | | " + " | ".join(
            model_label(T[m]["ref"]) if has_ref[m] else "" for m in models) + " |")
    lines += ["", "Stars on coefficients test against the Null column (1 for ln(p/(1-p)), 0 otherwise); "
              f"SEs cluster-robust by {glabel}. Pseudo R² = McFadden. ΔR² and Wald χ² are relative to the "
              f"model in the \"vs.\" row; Wald χ² is the cluster-robust ({glabel}) joint test that the "
              "added terms are zero. *** p<0.001, ** p<0.01, * p<0.05, . p<0.1"]
    return "\n".join(lines) + "\n"


def save_csv(tab, path):
    pd.DataFrame(tab).to_csv(path, index=False, encoding="utf-8-sig")


save_csv(t3_rows, os.path.join(OUT_DIR, "table3_correlation.csv"))

for cname in CLUSTERS:
    out = os.path.join(OUT_DIR, cname)
    os.makedirs(out, exist_ok=True)
    R = rows[cname]
    save_csv(R["t1"], os.path.join(out, "table1_baseline.csv"))
    save_csv(R["t2"], os.path.join(out, "table2_control_candidates.csv"))
    save_csv(R["t4"], os.path.join(out, "table4_csd_models.csv"))
    for fname, rows_, feats_, title in [
            ("table2_control_candidates.md", R["t2"], CTRL_FEATS,
             "Table 2. Control-variable candidates"),
            ("table4_csd_models.md", R["t4"], ["log_odds", "delta_logit", "lo_x_ttm", "lo_x_csd", "csd_q_c"],
             "Table 4. CSD-augmented models")]:
        with open(os.path.join(out, fname), "w", encoding="utf-8") as f:
            f.write(paper_table_md(pd.DataFrame(rows_), feats_, title, cname))
    print(f"  [{cname}] tables saved → {out}")

print(f"\nall tables saved → {OUT_DIR}")
