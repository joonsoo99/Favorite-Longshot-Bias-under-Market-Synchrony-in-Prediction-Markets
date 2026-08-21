# -*- coding: utf-8 -*-
"""
common.py — 분석 스크립트 공통 유틸리티

panels/panel_{5m,15m,60m}.pkl을 읽어 CSD·통제변수까지 붙은 분석용 표본을
만드는 로직과, GLM 적합/유의성 표시 등 5개 분석 스크립트가 공통으로 쓰는
함수를 모아둔다. 각 스크립트에 흩어져 있던 중복 코드(ASSET_STD, compute_csd,
fit 등)를 한 곳으로 모은 것으로, 로직 자체는 바뀌지 않았다.

delta_logit 컨벤션 (확정, build_panel.py에서 계산·저장): 각 에피소드×토큰의
첫 관측치는 직전 값이 없어 Δlogit을 정의할 수 없으므로, 0이나 logit(0.5)
같은 임의값으로 채우지 않고 NaN으로 남겨 표본에서 제외한다. (0 채움은
"몰라서 0"과 "안 변해서 0"을 섞고, logit(0.5) 대비 계산은 5m에서 log_odds와
완전공선성을 일으켜 계수를 왜곡시킨다는 것을 실증적으로 확인했다 — 두
대안 모두 기각.) 같은 원칙으로, price_history 중간에 누락된 분(minute)이
있어 바로 이전 행과 ttm 간격이 1이 아닌 경우도 NaN 처리한다.
load_analysis_panel()은 이 컬럼을 재계산하지 않고 panel pkl에 저장된
값을 그대로 쓴다.
"""
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

# 시간대별 플롯 색상 (dataviz 팔레트, 카테고리 순서 고정)
TF_COLORS  = {"5m": "#2a78d6", "15m": "#eb6834", "60m": "#1baf7a"}
TF_MARKERS = {"5m": "o", "15m": "s", "60m": "^"}

# equity_curve.py에서 비교할 3개 모형
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

# equity_curve.py(백테스트+위험조정 통합 표·그림)에서 쓰는 전략 파라미터·필요 컬럼
BACKTEST_FEE         = 0.015
BACKTEST_EDGE_THRESH = 0.015
BACKTEST_NEED = ["log_odds", "delta_logit", "ttm_c", "lo_x_ttm", "csd_raw", "csd_q_c", "lo_x_csd"]


def sig(p) -> str:
    """p값 → 유의성 별표."""
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.1 else "")))


def compute_csd(panel: pd.DataFrame) -> pd.DataFrame:
    """obs_epoch별 canonical CSD = std(log_odds) over 4 assets (YES 토큰 기준).

    4자산이 동시에 관측된 obs_epoch만 대상으로 하며, 반환 컬럼명은
    obs_epoch, csd_raw (원값, rank 변환 전) 이다.
    """
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
    """Cluster-robust(episode_id) 로지스틱 GLM."""
    Xc = sm.add_constant(Xdf.astype(float), has_constant="add")
    m = sm.GLM(y, Xc, family=sm.families.Binomial())
    return m.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(ep_ids)}, disp=False)


def coef_se_p(res, name: str, null: float = 0.0):
    """계수, SE, H0=null 대비 양측 p값. 모형에 없는 변수면 (None, None, None)."""
    if name not in res.params.index:
        return None, None, None
    c = float(res.params[name])
    se = float(res.bse[name])
    z = (c - null) / se if se > 0 else np.nan
    p = float(2 * sp_stats.norm.sf(abs(z)))
    return c, se, p


def load_analysis_panel(tf: str, extra_need: list[str] | None = None) -> pd.DataFrame:
    """panel_{tf}.pkl을 읽어 분석에 필요한 파생변수를 전부 붙이고,
    결측 없는 행만 남긴 최종 표본을 반환한다.

    붙는 컬럼:
      ttm_c, lo_x_ttm            (ttm 중앙값 중심화 및 log_odds와의 상호작용)
      lo_x_delta                 (log_odds × delta_logit, delta_logit은 panel pkl 원본 그대로)
      poly_vol_log, lo_x_vol     (거래량 로그, log_odds와의 상호작용)
      csd_raw, csd_q_c, lo_x_csd (CSD 원값 / 만기구간별 순위중심화 / 상호작용)
      price                      (=sigmoid(log_odds), 백테스트용 시장가)

    extra_need로 지정한 컬럼까지 포함해 dropna하며, 지정 안 하면 전체
    파생변수 기준으로 dropna한다(=Table류 스크립트의 기본 동작).
    """
    panel = pd.read_pickle(os.path.join(PNL_DIR, TF_CFG[tf]))
    # delta_logit은 build_panel.py가 이미 계산해 저장했으므로 그대로 쓴다
    # (여기서 다시 계산하면 두 곳의 로직을 항상 동기화해야 하는 중복이 생긴다).

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


def trade_mask(res, price, edge_thresh: float = BACKTEST_EDGE_THRESH):
    """모형 적합확률(res.fittedvalues)이 시장가보다 edge_thresh 이상 높으면 매수."""
    return (res.fittedvalues - price) >= edge_thresh


def trade_cost_pnl(trade, price, outcome, fee: float = BACKTEST_FEE):
    """거래된 행만의 매수비용(cost)과 손익(pnl). 매수가 = 시장가 + fee."""
    cost = price[trade] + fee
    pnl = outcome[trade] - cost
    return cost, pnl


FIG_DPI = 600  # PNG 미리보기용 해상도 (제출용은 PDF 벡터라 DPI 무관)


def setup_plot_rc() -> None:
    """Elsevier 아트웍 가이드 충족: PDF에 폰트를 TrueType으로 실제 임베드(Type 3
    비트맵 폰트 대신)하고, 승인 폰트 목록(Arial/Helvetica/Times/Courier/Symbol)
    중 하나를 쓴다. 플로팅 스크립트가 matplotlib.pyplot을 임포트한 직후,
    figure를 만들기 전에 한 번 호출한다."""
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42  # Type 42 = TrueType 임베드
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Arial"


def save_fig(fig, path_no_ext: str) -> None:
    """PNG(미리보기용, FIG_DPI)와 PDF(제출용 벡터, Elsevier 승인 포맷) 둘 다 저장."""
    fig.savefig(f"{path_no_ext}.png", facecolor=fig.get_facecolor(), dpi=FIG_DPI)
    fig.savefig(f"{path_no_ext}.pdf", facecolor=fig.get_facecolor())


def style_axes(ax) -> None:
    """플롯 공통 스타일(테두리/눈금 색/격자) 적용."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
