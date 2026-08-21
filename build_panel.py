# -*- coding: utf-8 -*-
"""
build_panel.py — 5m / 15m / 60m 패널 구축

입력: download/raw_output/{5m,15m,60m}_poly_raw.pkl
출력: panels/panel_{5m,15m,60m}.pkl

컬럼:
  episode_id, asset, slot_epoch, token, ttm, obs_epoch, outcome,
  log_odds, delta_logit, poly_vol_prev

완결성 필터링: episode(asset×slot) 단위로 price_history가 ttm=0..
  window_minutes-1을 빈틈없이 다 채우지 못하면 통째로 제거한다.
  (ttm=window_minutes, 즉 윈도우 시작 시점은 실측상 0.01~0.05%에서만
  존재해 API가 구조적으로 못 주는 값이다 — 개장 직후 1분 내엔 체결/
  호가가 거의 없다. 결측이 아니라 애초에 달성 불가능한 범위이므로
  "완결"의 기준에서 제외한다.) 그 뒤, 4개 자산(BTC/ETH/SOL/XRP)이 같은
  slot_epoch에서 전부 살아남지 못하면 그 slot_epoch 자체를 제거한다 —
  CSD(cross-sectional dispersion)가 4자산 동시 관측을 전제로 하므로
  한 자산만 빠져도 그 시점 전체가 분석에 쓰일 수 없기 때문이다.

delta_logit: 직전 1분 대비 log_odds 변화. 각 에피소드×토큰의 첫 시점(직전
  관측 없음)은 0으로 채우지 않고 NaN으로 남긴다 — 분석 시 그 행을 제외하는
  것이 확정된 방식이다 (0으로 채우면 "몰라서 0"과 "안 변해서 0"이 섞이고,
  logit(0.5) 대비로 채우면 5m에서 log_odds와 완전공선성이 생겨 계수가
  왜곡됨을 확인했다). 위 완결성 필터링 덕분에 매 episode가 ttm 간격 1인
  연속 시퀀스이므로 첫 시점 외에는 NaN이 생기지 않는다.
"""
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


def is_complete(ph, window_minutes: int) -> bool:
    """ph가 achievable한 전체 범위 ttm=0..window_minutes-1을 다 포함하는지.

    포함 관계(>=)로 검사한다 — 극히 드물게(전체의 0.01~0.05%) ttm=
    window_minutes(보너스 시점)까지 얹혀 있는 행도 있는데, 이건 필요한
    범위를 다 가지고 있고 더 가진 것이므로 완결로 봐야 한다(==로 검사하면
    이런 행을 "불완전"으로 잘못 분류하게 된다). 이 보너스 시점 자체는
    expand_price_history에서 잘라내, 모든 episode의 ttm 범위가 항상
    1..window_minutes-1로 균일하게 유지되도록 한다.
    """
    return isinstance(ph, dict) and set(ph.keys()) >= set(range(window_minutes))


def filter_complete_and_simultaneous(df_raw: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    """episode 완결성 → 4자산 동시성 순으로 필터링."""
    complete = df_raw.apply(
        lambda r: is_complete(r["price_history_yes"], window_minutes)
                  and is_complete(r["price_history_no"], window_minutes),
        axis=1,
    )
    df = df_raw[complete]
    n_assets = df.groupby("slot_epoch")["asset"].transform("nunique")
    df = df[n_assets == N_ASSETS]
    return df.reset_index(drop=True)


def expand_price_history(df_raw: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    """YES/NO 각 토큰의 price_history → 슬롯 레벨 패널 (ttm 1..window_minutes-1만 포함).
    token='yes': log_odds=logit(price_yes), outcome=row.outcome
    token='no' : log_odds=logit(price_no),  outcome=1-row.outcome

    obs_epoch(실제 관측 wall-clock epoch) = slot_epoch(개장) + window_minutes*60
    (마감) - ttm*60. slot_epoch는 개장 시각이므로 여기에 윈도우 길이를
    더해야 마감 시각이 나오고, 거기서 ttm분을 빼야 그 관측이 실제로
    일어난 시각이 된다.
    """
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


def add_delta_logit(panel: pd.DataFrame) -> pd.DataFrame:
    """에피소드 × 토큰별 모멘텀: TTM 내림차순(시간순) diff.
    첫 시점은 NaN으로 남김 (모듈 docstring 참고). 완결성 필터링을 거쳤으므로
    정상적으로는 이게 유일한 NaN 원인이지만, ttm 간격이 1이 아닌 행도
    안전장치로 동일하게 NaN 처리한다."""
    panel = panel.sort_values(["episode_id", "token", "ttm"],
                              ascending=[True, True, False]).copy()
    g = panel.groupby(["episode_id", "token"])
    panel["delta_logit"] = g["log_odds"].diff()
    panel.loc[g["ttm"].diff() != -1, "delta_logit"] = np.nan
    return panel


def add_poly_vol_prev(panel: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """직전 슬롯 Polymarket 거래대금."""
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
