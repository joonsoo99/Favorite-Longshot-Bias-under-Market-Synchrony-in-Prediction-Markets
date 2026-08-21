# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "download", "raw_output")
OUT_DIR = os.path.join(ROOT, "panels")
os.makedirs(OUT_DIR, exist_ok=True)

TF_CFG = {
    "5m":  {"raw": "5m_poly_raw.pkl",  "window_minutes": 5},
    "15m": {"raw": "15m_poly_raw.pkl", "window_minutes": 15},
    "60m": {"raw": "60m_poly_raw.pkl", "window_minutes": 60},
}

N_ASSETS = 4
EPS = 1e-6


# ttm == window_minutes (the open instant) is present in only ~0.01-0.05% of
# rows (structurally unreachable, not missing data), so "complete" means
# covering 0..window_minutes-1, checked as a superset (>=) rather than exact
# equality so rows with that rare bonus tick aren't wrongly flagged incomplete.
def is_complete(ph, window_minutes: int) -> bool:
    return isinstance(ph, dict) and set(ph.keys()) >= set(range(window_minutes))


def filter_complete_and_simultaneous(df_raw: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    complete = df_raw.apply(
        lambda r: is_complete(r["price_history_yes"], window_minutes)
                  and is_complete(r["price_history_no"], window_minutes),
        axis=1,
    )
    df = df_raw[complete]
    n_assets = df.groupby("slot_epoch")["asset"].transform("nunique")
    df = df[n_assets == N_ASSETS]
    return df.reset_index(drop=True)


# obs_epoch = slot_epoch (open) + window_minutes*60 (close) - ttm*60, i.e. the
# actual wall-clock time of that tick, not the naive slot_epoch - ttm*60.
def expand_price_history(df_raw: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    records = []
    for ep_id, row in enumerate(df_raw.itertuples(index=False)):
        tokens = [
            ("yes", row.price_history_yes, int(row.outcome)),
            ("no",  row.price_history_no,  1 - int(row.outcome)),
        ]
        for token, ph, token_outcome in tokens:
            for ttm, price in ph.items():
                if ttm < 1 or ttm >= window_minutes:
                    continue
                p = float(np.clip(price, EPS, 1 - EPS))
                records.append({
                    "episode_id": ep_id,
                    "asset":      row.asset,
                    "slot_epoch": row.slot_epoch,
                    "token":      token,
                    "ttm":        int(ttm),
                    "obs_epoch":  int(row.slot_epoch) + window_minutes * 60 - int(ttm) * 60,
                    "outcome":    token_outcome,
                    "log_odds":   float(np.log(p / (1 - p))),
                })
    return pd.DataFrame(records)


# First tick of each episode x token has no prior value, so delta_logit is
# left NaN (never filled with 0 or a synthetic baseline) and excluded
# downstream. The ttm-gap check is a belt-and-suspenders guard: the
# completeness filter above should already make every step exactly 1 minute.
def add_delta_logit(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["episode_id", "token", "ttm"],
                              ascending=[True, True, False]).copy()
    g = panel.groupby(["episode_id", "token"])
    panel["delta_logit"] = g["log_odds"].diff()
    panel.loc[g["ttm"].diff() != -1, "delta_logit"] = np.nan
    return panel


def add_poly_vol_prev(panel: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    vol = df_raw[["asset", "slot_epoch", "volume"]].sort_values(["asset", "slot_epoch"]).copy()
    vol["poly_vol_prev"] = vol.groupby("asset")["volume"].shift(1)
    return panel.merge(vol[["asset", "slot_epoch", "poly_vol_prev"]],
                        on=["asset", "slot_epoch"], how="left")


FINAL_COLS = ["episode_id", "asset", "slot_epoch", "token", "ttm", "obs_epoch",
              "outcome", "log_odds", "delta_logit", "poly_vol_prev"]

for tf, cfg in TF_CFG.items():
    print(f"\n{'='*60}\n  [{tf}] 패널 빌드\n{'='*60}")

    df_raw = pd.read_pickle(os.path.join(RAW_DIR, cfg["raw"]))
    print(f"  raw: {len(df_raw):,}행  assets: {sorted(df_raw['asset'].unique())}")

    n0 = len(df_raw)
    df_raw = filter_complete_and_simultaneous(df_raw, cfg["window_minutes"])
    print(f"  완결성+동시성 필터: {n0:,} → {len(df_raw):,}  "
          f"(제거 {n0-len(df_raw):,}, {(n0-len(df_raw))/n0*100:.2f}%)")

    panel = expand_price_history(df_raw, cfg["window_minutes"])
    panel = add_delta_logit(panel)
    panel = add_poly_vol_prev(panel, df_raw)
    panel = panel[FINAL_COLS].sort_values(["slot_epoch", "asset", "ttm"]).reset_index(drop=True)

    print(f"  최종: {panel.shape}  TTM 범위: {panel['ttm'].min()}~{panel['ttm'].max()}")
    print(f"  null 개수:\n{panel.isnull().sum().to_string()}")

    out_path = os.path.join(OUT_DIR, f"panel_{tf}.pkl")
    panel.to_pickle(out_path)
    print(f"  저장 완료 → {out_path}")

print(f"\n{'='*60}\n  모든 패널 빌드 완료\n{'='*60}")
