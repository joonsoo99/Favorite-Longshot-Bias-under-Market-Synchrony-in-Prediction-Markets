# -*- coding: utf-8 -*-
# Sample construction (paper Section 2). From the raw data collected by
# download/collect_polymarket_updown.py, builds the minute-level analysis
# sample used by every table, figure and test:
#   1. keep episodes whose minute-level price records are complete for both
#      the UP and DOWN contracts (and time windows with all 4 assets);
#   2. drop episodes whose previous episode (same asset and horizon) has no
#      volume -- this removes the first episode of each asset and the
#      March 2026 period without volume data;
#   3. keep only time windows in which all 4 assets remain;
#   4. within each episode, keep the minutes from the second minute after
#      the start through one minute before maturity (the first minute has
#      no previous price for the price-change control).
# CSD (cross-sectional dispersion of the 4 assets' UP log-odds) is computed
# per minute before steps 2-4, since it is a property of the time point.
# Output: panels/panel_{5m,15m,60m}.pkl
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


# CSD = sample standard deviation (ddof=1) of the 4 assets' UP log-odds at the
# same minute (obs_epoch); only minutes with all 4 assets get a value. Every
# row (both tokens, all 4 assets) at that minute gets the same value.
ASSET_STD = {"bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH",
             "solana": "SOL", "sol": "SOL", "xrp": "XRP"}
ASSETS = ["BTC", "ETH", "SOL", "XRP"]


def compute_csd(panel: pd.DataFrame) -> pd.DataFrame:
    yes = panel[panel["token"] == "yes"].copy()
    yes["asset_std"] = yes["asset"].map(ASSET_STD)
    cs = yes.groupby(["obs_epoch", "asset_std"])["log_odds"].mean().unstack("asset_std")
    cs = cs[[a for a in ASSETS if a in cs.columns]].dropna(how="any")
    out = cs.std(axis=1).reset_index()
    out.columns = ["obs_epoch", "csd_raw"]
    return out


# Steps 2-4: drop rows without the previous episode's volume (step 2), the
# first minute of each episode, whose delta_logit is undefined (step 4), and
# minutes without a CSD value; then keep only time windows in which all 4
# assets remain (step 3).
def final_sample(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.dropna(subset=["delta_logit", "poly_vol_prev", "csd_raw"])
    full_window = df.groupby("slot_epoch")["asset"].transform("nunique") == N_ASSETS
    return df[full_window].reset_index(drop=True)


FINAL_COLS = ["episode_id", "asset", "slot_epoch", "token", "ttm", "obs_epoch",
              "outcome", "log_odds", "delta_logit", "poly_vol_prev", "csd_raw"]

for tf, cfg in TF_CFG.items():
    print(f"\n{'='*60}\n  [{tf}] build panel\n{'='*60}")

    df_raw = pd.read_pickle(os.path.join(RAW_DIR, cfg["raw"]))
    print(f"  raw: {len(df_raw):,} rows  assets: {sorted(df_raw['asset'].unique())}")

    n0 = len(df_raw)
    df_raw = filter_complete_and_simultaneous(df_raw, cfg["window_minutes"])
    print(f"  completeness+simultaneity filter: {n0:,} → {len(df_raw):,}  "
          f"(dropped {n0-len(df_raw):,}, {(n0-len(df_raw))/n0*100:.2f}%)")

    panel = expand_price_history(df_raw, cfg["window_minutes"])
    panel = add_delta_logit(panel)
    panel = add_poly_vol_prev(panel, df_raw)
    panel = panel.sort_values(["slot_epoch", "asset", "ttm"]).reset_index(drop=True)
    panel = panel.merge(compute_csd(panel), on="obs_epoch", how="left")

    n1 = len(panel)
    panel = final_sample(panel)[FINAL_COLS]
    print(f"  steps 2-4 (volume, 4-asset windows, first minute): {n1:,} → {len(panel):,} rows")
    print(f"  final sample: {len(panel):,} rows  {panel['episode_id'].nunique():,} episodes  "
          f"{panel['slot_epoch'].nunique():,} time windows  TTM range: {panel['ttm'].min()}~{panel['ttm'].max()}")

    out_path = os.path.join(OUT_DIR, f"panel_{tf}.pkl")
    panel.to_pickle(out_path)
    print(f"  saved → {out_path}")

print(f"\n{'='*60}\n  all panels built\n{'='*60}")
