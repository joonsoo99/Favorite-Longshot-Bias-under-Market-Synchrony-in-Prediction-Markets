# -*- coding: utf-8 -*-
"""
equity_curve.py — 백테스트 성과·위험조정 지표 표 + 누적손익(equity curve) 그림

이전엔 backtest.py(수익성 표: trades/ROI/win_rate)와 equity_curve.py(위험조정
표: MDD/Calmar)가 따로 있었는데, 논문 Table 5·6이 사실상 같은 표(모형×TF당
한 행)라 컬럼만 나눠서 중복이었다. 모형 적합도 tf×model마다 각자 한 번씩
따로 하고 있어 계산도 중복이었다. 이제 fit을 한 번만 해서 두 표의 모든
컬럼을 한 번에 계산·저장한다 (backtest.py는 폐기).

전략: 각 행(관측치)마다 모형 적합확률 q_hat과 시장가 P(=sigmoid(log_odds))를
비교해 q_hat - P >= EDGE_THRESH면 그 토큰을 1주 매수(매수가 = P + FEE).
손익 = 실현 outcome(0/1) - 매수가. in-sample 백테스트다(모형을 전체 데이터로
학습하고 같은 데이터로 평가).

위험조정 지표(MDD, Calmar)는 거래 시점(obs_epoch) 순 누적손익($) 곡선
기준으로 계산한다(단리 가정 — 재투자·복리 없음). %로 정규화하면 초반
누적매수비용이 작을 때 변동폭이 과장되는 문제가 있어, MDD는 $ 절대 낙폭
(peak-to-trough)으로 정의한다. 연환산 손익 = (최종 누적손익$) ×
(365/거래기간일수)의 단순 연율화, Calmar = 연환산 손익$ / |MDD$|.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common import (TF_CFG, OUT_DIR, BACKTEST_MODEL_SPECS,
                     BACKTEST_MODEL_COLORS, BACKTEST_FEE, BACKTEST_EDGE_THRESH, BACKTEST_NEED, FIG_DPI,
                     fit, load_analysis_panel, trade_mask, trade_cost_pnl, style_axes, save_fig,
                     setup_plot_rc)

setup_plot_rc()

FEE, EDGE_THRESH, NEED = BACKTEST_FEE, BACKTEST_EDGE_THRESH, BACKTEST_NEED

EMPTY_ROW = dict(n_trades=0, total_cost=0.0, total_pnl=0.0, avg_pnl=np.nan, roi_pct=np.nan,
                  win_rate=np.nan, mdd_dollar=np.nan, n_days=np.nan, ann_pnl=np.nan, calmar=np.nan)


def backtest_row(trade, cost, pnl, outcome, t_dates):
    """거래(1건 이상) → 수익성 지표(trades/ROI/win_rate) + 위험조정 지표(MDD/Calmar) 한 행.
    trade.sum()==0인 경우는 호출 전에 EMPTY_ROW로 따로 처리한다."""
    n_trades = int(trade.sum())
    order = np.argsort(t_dates)
    cum_pnl = np.cumsum(pnl[order])
    mdd = float((cum_pnl - np.maximum.accumulate(cum_pnl)).min())
    n_days = max((t_dates[order][-1] - t_dates[order][0]).total_seconds() / 86400, 1)
    total_pnl = float(cum_pnl[-1])
    total_cost = float(cost.sum())
    ann_pnl = total_pnl * 365 / n_days

    return dict(
        n_trades=n_trades, total_cost=total_cost, total_pnl=total_pnl,
        avg_pnl=float(pnl.mean()), roi_pct=total_pnl / total_cost * 100,
        win_rate=float((outcome[trade] == 1).mean()),
        mdd_dollar=mdd, n_days=n_days, ann_pnl=ann_pnl,
        calmar=ann_pnl / abs(mdd) if mdd != 0 else np.nan,
    ), cum_pnl, order


# figsize 폭 7.5in ≈ 190mm = Elsevier 2단(전체 폭) 규격
fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.1), dpi=FIG_DPI, sharey=False)
fig.patch.set_facecolor("#fcfcfb")
rows = []

for ax, tf in zip(axes, TF_CFG):
    print(f"\n{'='*70}\n  [{tf}]\n{'='*70}")
    df = load_analysis_panel(tf, extra_need=NEED)
    y  = df["outcome"].values.astype(float)
    ep = df["episode_id"].values
    price = df["price"].values
    t_dates_all = pd.to_datetime(df["obs_epoch"].values, unit="s", utc=True)

    style_axes(ax)
    ax.axhline(0, color="#c3c2b7", linewidth=1.2, linestyle="--", zorder=1)

    for mname, feats in BACKTEST_MODEL_SPECS.items():
        res = fit(y, ep, df[feats])
        trade = trade_mask(res, price, EDGE_THRESH)
        if trade.sum() == 0:
            print(f"  [{mname}] 거래 없음")
            rows.append({"tf": tf, "model": mname, **EMPTY_ROW})
            continue
        cost, pnl = trade_cost_pnl(trade, price, y, FEE)
        row, cum_pnl, order = backtest_row(trade, cost, pnl, y, t_dates_all[trade])
        rows.append({"tf": tf, "model": mname, **row})
        print(f"  [{mname}]  trades={row['n_trades']:,}  total_pnl=${row['total_pnl']:,.1f}  "
              f"roi={row['roi_pct']:.2f}%  win_rate={row['win_rate']*100:.1f}%  "
              f"mdd=${row['mdd_dollar']:,.1f}  calmar={row['calmar']:.2f}")

        # 범례는 모형명만 짧게 (n/final 값은 표(backtest_results.csv)에 이미 있음) —
        # 패널 폭이 190mm 규격에 맞춰 좁아져서 긴 라벨은 겹침/잘림이 생긴다.
        ax.plot(t_dates_all[trade].values[order], cum_pnl, color=BACKTEST_MODEL_COLORS[mname],
                linewidth=1.3, label=mname.split("_")[0])

    ax.set_title(tf, fontsize=9, fontweight="bold")
    ax.set_xlabel("Time", fontsize=7.5)
    if ax is axes[0]:
        ax.set_ylabel("Cumulative PnL ($)", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.2, borderaxespad=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(labelsize=7)

fig.tight_layout()
base_path = os.path.join(OUT_DIR, "FIG3")  # 논문 Figure 3: 백테스트 equity curve
save_fig(fig, base_path)
print(f"\n플롯 저장 → {base_path}.png / .pdf")

# ── 통합 표 (Table 5+6: 수익성 + 위험조정) ──────────────────────
out = pd.DataFrame(rows)[["tf", "model", "n_trades", "total_cost", "total_pnl", "avg_pnl",
                           "roi_pct", "win_rate", "mdd_dollar", "ann_pnl", "calmar"]]
csv_path = os.path.join(OUT_DIR, "backtest_results.csv")
out.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\n{'='*96}\n  Table 5+6 통합: 백테스트 성과 및 위험조정 지표\n{'='*96}")
header = (f"{'tf':>4} {'model':<14} {'trades':>9} {'ROI%':>7} {'win%':>6} "
          f"{'총손익$':>10} {'MDD$':>10} {'연환산$':>10} {'Calmar':>7}")
print(header)
for _, r in out.iterrows():
    print(f"{r['tf']:>4} {r['model']:<14} {r['n_trades']:>9,} {r['roi_pct']:>7.2f} "
          f"{r['win_rate']*100:>6.1f} {r['total_pnl']:>10,.1f} {r['mdd_dollar']:>10,.1f} "
          f"{r['ann_pnl']:>10,.1f} {r['calmar']:>7.2f}")
print(f"\n표 저장 완료 → {csv_path}")
