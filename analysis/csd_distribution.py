# -*- coding: utf-8 -*-
"""
csd_distribution.py — CSD 원본(raw) 분포 히스토그램

목적: CSD가 오른쪽으로 긴 꼬리를 갖는 분포임을 보여, 회귀분석에서
백분위 순위(rank) 변환을 사용하는 이유를 시각적으로 뒷받침한다.

CSD는 obs_epoch 단위 지표라 delta_logit/ttm 등 관측(행) 단위 파생변수가
필요 없으므로, load_analysis_panel을 거치지 않고 compute_csd만 바로 쓴다.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import matplotlib.pyplot as plt

from common import (TF_CFG, PNL_DIR, OUT_DIR, TF_COLORS, FIG_DPI, compute_csd, style_axes,
                     save_fig, setup_plot_rc)

setup_plot_rc()

# figsize 폭 7.5in ≈ 190mm = Elsevier 2단(전체 폭) 규격
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9), dpi=FIG_DPI, sharey=False)
fig.patch.set_facecolor("#fcfcfb")

for ax, (tf, fname) in zip(axes, TF_CFG.items()):
    panel = pd.read_pickle(os.path.join(PNL_DIR, fname))
    csd = compute_csd(panel)["csd_raw"]
    mean_v, med_v = csd.mean(), csd.median()
    p95 = csd.quantile(0.95)

    style_axes(ax)
    ax.hist(csd, bins=80, range=(0, p95 * 1.6), color=TF_COLORS[tf], alpha=0.85, edgecolor="none")
    ax.axvline(med_v, color="#0b0b0b", linewidth=1.8, linestyle="-", label=f"median={med_v:.2f}")
    ax.axvline(mean_v, color="#0b0b0b", linewidth=1.8, linestyle="--", label=f"mean={mean_v:.2f}")
    ax.set_title(f"{tf}  (skew={csd.skew():.2f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("CSD (raw std of log-odds)", fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel("Frequency (obs_epoch)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    print(f"{tf}  mean={mean_v:.3f}  median={med_v:.3f}  skew={csd.skew():.3f}  "
          f"max={csd.max():.3f}  p99={csd.quantile(0.99):.3f}")

fig.tight_layout()
base_path = os.path.join(OUT_DIR, "FIG1")  # 논문 Figure 1: CSD 분포(rank 변환 근거)
save_fig(fig, base_path)
print(f"저장 완료 → {base_path}.png / .pdf")
