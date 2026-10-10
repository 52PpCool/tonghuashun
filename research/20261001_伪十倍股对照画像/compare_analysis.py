#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_analysis.py —— 伪十倍股对照画像 第四步：十倍股 vs 伪十倍股 画像区分度检验

研究问题：上一轮画像（行业集中/次新/小市值）到底是"十倍股的特征"，
         还是"所有大涨股都这样"（即画像没有区分度）？

对照组：同一智能选股初筛 137 只中，核验涨幅 ∈ [500%, 1000%) 的 28 只。
        两组共享"通过初筛"这一选择，唯一差异是核验涨幅档位 —— 条件对照组。
        （诚实声明：对照不是"全体A股"，结论适用范围是"初筛选出来的大涨股内部"。）

检验设计：
  H1 行业：十倍股的"硬科技三兄弟"（电子/通信/机械设备）占比是否显著更高 → Fisher 精确检验
  H2 上市批次：窗口起点（2016-09-01）前已上市（老股）占比对比 → Fisher 精确检验
  H3 起点市值：分布对比 → Mann-Whitney U + 中位数 + <50亿占比 Fisher
  效应量：比值比 OR 与组间差，不只报 p 值（课23贝叶斯精神的频率版补充）。

多重比较声明：3 个检验，Bonferroni 校正阈值 0.05/3≈0.0167（课36精神）。
"""
import json
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK HK",
                                   "Noto Sans Mono CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DIR = os.path.dirname(os.path.abspath(__file__))
TENX_FULL = "/home/ppku/zcodespace/tonghuashun/research/20261001_十倍股画像/data/profile_full.csv"
OUT_FULL = os.path.join(DIR, "data", "control_full.csv")
OUT_SUM = os.path.join(DIR, "data", "comparison.json")
CHARTS = os.path.join(DIR, "charts")
os.makedirs(CHARTS, exist_ok=True)

HARD_TECH = {"电子", "通信", "机械设备"}
CAP_YI = 50.0  # "小市值"阈值（上一轮画像：70% 十倍股 <50亿）


def parse_cap_json(fpath):
    """与上一轮同款：直接在原始文件文本上正则（Node 写出，中文未转义）。"""
    txt = open(fpath, encoding="utf-8").read()
    m = re.search(r"\|(\d{6}\.(?:SZ|SH))\|[^|]*\|([^|]+)\|", txt)
    if not m:
        return None
    mm = re.match(r"([\d.]+)(亿|万|元)?", m.group(2))
    if not mm:
        return None
    val, unit = float(mm.group(1)), mm.group(2) or "元"
    return val if unit == "亿" else (val / 1e4 if unit == "万" else val / 1e8)


def main():
    tenx = pd.read_csv(TENX_FULL, dtype={"list_date": str})
    ctl = pd.read_csv(os.path.join(DIR, "data", "profile_control.csv"), dtype={"list_date": str})

    caps = {}
    for f in glob.glob(os.path.join(DIR, "raw", "03_cap_at_listday", "*.json")):
        yi = parse_cap_json(f)
        if yi:
            caps[os.path.basename(f).replace("_", ".").replace(".json", "")] = yi
    ctl["start_mktcap_yi"] = ctl.apply(
        lambda r: r["mktcap_20160901_yi"] if r["listed_before_window"]
        else caps.get(r["code"], np.nan), axis=1)
    ctl["start_cap_anchor"] = np.where(ctl["listed_before_window"], "2016-09-01市值", "上市日市值")
    ctl["group"] = "control_500_1000"
    tenx["group"] = "tenx_1000plus"
    ctl.to_csv(OUT_FULL, index=False, encoding="utf-8-sig")
    print(f"[info] 对照组起点市值补齐 {ctl['start_mktcap_yi'].notna().sum()}/{len(ctl)}")
    print(f"[info] 十倍组起点市值补齐 {tenx['start_mktcap_yi'].notna().sum()}/{len(tenx)}")

    S = {"n_tenx": len(tenx), "n_control": len(ctl)}

    # ---------- H1 行业 ----------
    t_hard = int(tenx["ths_industry"].isin(HARD_TECH).sum())
    c_hard = int(ctl["ths_industry"].isin(HARD_TECH).sum())
    table = [[t_hard, len(tenx) - t_hard], [c_hard, len(ctl) - c_hard]]
    or_h1, p_h1 = stats.fisher_exact(table)
    S["H1_industry"] = {
        "hard_tech_trio": sorted(HARD_TECH), "tenx_share": t_hard / len(tenx),
        "control_share": c_hard / len(ctl), "table": table,
        "odds_ratio": None if np.isinf(or_h1) else float(or_h1), "p_fisher": float(p_h1)}

    # ---------- H2 上市批次（窗口起点前/后） ----------
    # 教训留痕（2026-10-01 复查修正）：此前按 list_year<=2015 判"老股"，会把 2016 年内
    # 早于窗口起点上市的股票错归为"窗口内上市"（十倍组：新易盛 2016-03-03、兆易创新
    # 2016-08-18），与本管道市值锚定口径（listed_before_window，见上一轮
    # profile_analysis.py）不一致。现统一为严格窗口口径：list_date < 2016-09-01 即老股。
    w_start = pd.Timestamp("2016-09-01")
    t_dates = pd.to_datetime(tenx["list_date"], format="%Y%m%d")
    c_dates = pd.to_datetime(ctl["list_date"], format="%Y%m%d")
    t_old = int((t_dates < w_start).sum())
    c_old = int((c_dates < w_start).sum())
    t_inwin = len(tenx) - t_old
    c_inwin = len(ctl) - c_old
    or_h2, p_h2 = stats.fisher_exact([[t_inwin, t_old], [c_inwin, c_old]])
    S["H2_cohort"] = {
        "definition": "老股 = 上市日 < 2016-09-01（严格窗口口径，与市值锚定一致）",
        "tenx_old_before_window": t_old, "control_old_before_window": c_old,
        "tenx_inwindow_share": t_inwin / len(tenx), "control_inwindow_share": c_inwin / len(ctl),
        "odds_ratio": float(or_h2), "p_fisher": float(p_h2)}

    # ---------- H3 起点市值 ----------
    a = tenx["start_mktcap_yi"].dropna()
    b = ctl["start_mktcap_yi"].dropna()
    u, p_h3 = stats.mannwhitneyu(a, b, alternative="two-sided")
    t_small = int((a < CAP_YI).sum()); c_small = int((b < CAP_YI).sum())
    or_h3, p_h3b = stats.fisher_exact([[t_small, len(a) - t_small], [c_small, len(b) - c_small]])
    S["H3_start_cap"] = {
        "tenx_median_yi": float(a.median()), "control_median_yi": float(b.median()),
        "tenx_under50_share": t_small / len(a), "control_under50_share": c_small / len(b),
        "mw_u": float(u), "p_mannwhitney": float(p_h3),
        "under50_odds_ratio": None if np.isinf(or_h3) else float(or_h3), "p_fisher_under50": float(p_h3b)}

    # 多重比较（3 个家族成员）
    S["bonferroni_alpha"] = 0.05 / 3
    for k in ("H1_industry", "H2_cohort", "H3_start_cap"):
        pvals = []
        for kk, vv in S[k].items():
            if kk.startswith("p_"):
                pvals.append(vv)
        S[k]["min_p"] = min(pvals)
        S[k]["significant_after_bonferroni"] = bool(min(pvals) < S["bonferroni_alpha"])

    # 行业分布全表（对照 vs 十倍）
    ind_t = tenx["ths_industry"].value_counts()
    ind_c = ctl["ths_industry"].value_counts()
    S["industry_counts_tenx"] = ind_t.to_dict()
    S["industry_counts_control"] = ind_c.to_dict()

    json.dump(S, open(OUT_SUM, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in S.items() if k.startswith("H") or "bonferroni" in k},
                     ensure_ascii=False, indent=2))

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ten-baggers (n=88) vs pseudo ten-baggers 500~1000% (n=28): does the profile discriminate?",
                 fontsize=12)

    ax = axes[0]
    cats = ["硬科技三兄弟\n(电子/通信/机械)", "其他行业"]
    tv = [t_hard, len(tenx) - t_hard]; cv = [c_hard, len(ctl) - c_hard]
    x = np.arange(2); w = 0.38
    ax.bar(x - w / 2, [v / len(tenx) * 100 for v in tv], w, label=f"十倍股 n={len(tenx)}", color="#C00000")
    ax.bar(x + w / 2, [v / len(ctl) * 100 for v in cv], w, label=f"伪十倍 n={len(ctl)}", color="#4472C4")
    for i in range(2):
        ax.text(i - w / 2, tv[i] / len(tenx) * 100 + 1, f"{tv[i]}只", ha="center")
        ax.text(i + w / 2, cv[i] / len(ctl) * 100 + 1, f"{cv[i]}只", ha="center")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("组内占比 %"); ax.set_title(f"H1 行业集中度  Fisher p={p_h1:.2e}")
    ax.legend()

    ax = axes[1]
    labels = ["窗口内上市\n(≥2016-09-01)", "老股\n(<2016-09-01)"]
    tv2 = [t_inwin, t_old]; cv2 = [c_inwin, c_old]
    ax.bar(x - w / 2, [v / len(tenx) * 100 for v in tv2], w, label="十倍股", color="#C00000")
    ax.bar(x + w / 2, [v / len(ctl) * 100 for v in cv2], w, label="伪十倍", color="#4472C4")
    for i in range(2):
        ax.text(i - w / 2, tv2[i] / len(tenx) * 100 + 1, f"{tv2[i]}只", ha="center")
        ax.text(i + w / 2, cv2[i] / len(ctl) * 100 + 1, f"{cv2[i]}只", ha="center")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("组内占比 %"); ax.set_title(f"H2 上市批次  Fisher p={p_h2:.2e}")
    ax.legend()

    ax = axes[2]
    bp = ax.boxplot([a, b], tick_labels=[f"十倍股 n={len(a)}", f"伪十倍 n={len(b)}"],
                    showfliers=True, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#C00000", "#4472C4"]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_ylabel("起点市值（亿元，log）")
    ax.set_title(f"H3 起点市值  MW-U p={p_h3:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "comparison_3panel.png"), dpi=150)
    print(f"[done] {OUT_SUM} + {CHARTS}/comparison_3panel.png")


if __name__ == "__main__":
    main()
