#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
profile_analysis.py —— 十倍股画像 第四步：核心分析

研究问题（在第 5 轮研究 20260905_最近10年A股十倍股 的 88 只已核验十倍股上做画像）：
  Q1 行业分布：十倍股集中在哪些行业？
  Q2 上市时间：窗口起点前已是老股，还是窗口内上市的次新？
  Q3 起点市值：十倍从多大的市值起步？（老股=2016-09-01 市值；次新=上市日市值）
  Q4 时间-10倍：从窗口起点到"首次达到10倍"用了多久？
  Q5 拿得住吗：起点→首次10倍这段路上的最大回撤有多深？

口径与诚实声明：
  - 价格为新浪源前复权收盘价（窗口 2016-09-01~2026-08-31，起点取窗口内首个交易日，
    对次新股即上市初期）。与 iFinD 核验涨幅存在复权基差异，逐股对比见 profile_full.csv，
    差异中位数在 README 报告。
  - "首次10倍"以收盘价 ≥ 10×起点收盘 定义（不含盘中），会系统性偏晚。
  - 幸存者偏差：88 只是"今天回看仍活着且涨满10倍"的股票，退市/归零者不在池内，
    所以本画像≠"买入并持有的期望"，这是教学上必须声明的。
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 中文字体：本机有 Noto CJK（fc-list 可查），否则图表中文会缺字
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK HK",
                                   "Noto Sans Mono CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data")
PRICE_DIR = os.path.join(DATA, "prices")
CHARTS = os.path.join(DIR, "charts")
OUT_FULL = os.path.join(DATA, "profile_full.csv")
TENX_CSV = "/home/ppku/zcodespace/tonghuashun/research/20260905_最近10年A股十倍股/数据/results_verified.csv"
os.makedirs(CHARTS, exist_ok=True)

W_START = pd.Timestamp("2016-09-01")
W_END = pd.Timestamp("2026-08-31")


def parse_cap_json(fpath):
    """直接在原始文件文本上正则（文件由 Node JSON.stringify 写出，中文未被转义）。
    教训留痕：此前用 json.dumps(loaded_dict) 再正则，Python 默认 ensure_ascii=True
    把"亿"转成 \\u4ebf，单位识别失败按"元"/1e8，市值全部缩小1e8倍。"""
    txt = open(fpath, encoding="utf-8").read()
    m = re.search(r"\|(\d{6}\.(?:SZ|SH))\|[^|]*\|([^|]+)\|", txt)
    if not m:
        return None
    raw = m.group(2)
    mm = re.match(r"([\d.]+)(亿|万|元)?", raw)
    if not mm:
        return None
    val, unit = float(mm.group(1)), mm.group(2) or "元"
    yi = val if unit == "亿" else (val / 1e4 if unit == "万" else val / 1e8)
    return yi


def max_drawdown(close):
    peak = close.cummax()
    dd = close / peak - 1.0
    return float(dd.min())


def main():
    df = pd.read_csv(os.path.join(DATA, "profile_basic.csv"), dtype={"list_date": str})
    verified = pd.read_csv(TENX_CSV)
    verified["code"] = verified["股票代码"]
    df = df.merge(verified[["code", "核验区间涨幅%(前复权)"]], on="code", how="left")

    # 补上 75 只次新股的上市日市值
    caps = {}
    for f in glob.glob(os.path.join(DIR, "raw", "03_cap_at_listday", "*.json")):
        yi = parse_cap_json(f)
        code = os.path.basename(f).replace("_", ".").replace(".json", "")
        if yi:
            caps[code] = yi
    df["start_mktcap_yi"] = df.apply(
        lambda r: r["mktcap_20160901_yi"] if r["listed_before_window"]
        else caps.get(r["code"], np.nan), axis=1)
    df["start_cap_anchor"] = np.where(df["listed_before_window"], "2016-09-01市值", "上市日市值")
    print(f"[info] 上市日市值补齐：{df['start_mktcap_yi'].notna().sum()}/88")

    rows = []
    for _, r in df.iterrows():
        fpath = os.path.join(PRICE_DIR, f"{r['code'].replace('.', '_')}.csv")
        if not os.path.exists(fpath):
            fpath = os.path.join(PRICE_DIR, f"{r['code'].split('.')[0]}.csv")
        rec = dict(r)
        if not os.path.exists(fpath):
            rec.update({"n_days": 0, "gain_calc_pct": np.nan, "days_to_10x": np.nan,
                        "years_to_10x": np.nan, "mdd_to_10x_pct": np.nan, "hit10x": False})
            rows.append(rec)
            continue
        px = pd.read_csv(fpath, parse_dates=["date"])
        px = px[(px["date"] >= W_START) & (px["date"] <= W_END)].reset_index(drop=True)
        close = px["close"].astype(float)
        base = close.iloc[0]
        base_date = px["date"].iloc[0]
        gain = close.iloc[-1] / base - 1.0
        # 首次10倍：收盘价 ≥ 10×起点收盘
        target = 10.0 * base
        hit = close >= target
        if hit.any():
            i10 = int(np.argmax(hit.values))
            d10 = px["date"].iloc[i10]
            days = (d10 - base_date).days
            mdd = max_drawdown(close.iloc[: i10 + 1])
            hit10x = True
        else:
            d10, days, mdd, hit10x = pd.NaT, np.nan, max_drawdown(close), False
        rec.update({
            "base_date": base_date.date().isoformat(), "base_close": base,
            "n_days": len(px), "gain_calc_pct": gain * 100,
            "days_to_10x": days, "years_to_10x": days / 365.25 if hit10x else np.nan,
            "date_to_10x": d10.date().isoformat() if hit10x else "",
            "mdd_to_10x_pct": mdd * 100, "hit10x": hit10x,
        })
        rows.append(rec)
    full = pd.DataFrame(rows)
    full.to_csv(OUT_FULL, index=False, encoding="utf-8-sig")
    print(f"[done] 写出 {OUT_FULL}")

    # ============ 汇总统计 ============
    S = {}
    S["n"] = len(full)
    S["hit10x_by_close"] = int(full["hit10x"].sum())
    d = full["gain_calc_pct"]
    diff = (full["gain_calc_pct"] - full["核验区间涨幅%(前复权)"]).abs()
    S["gain_calc_median_pct"] = float(d.median())
    S["gain_vs_ifind_median_absdiff_pct"] = float(diff.median())
    S["gain_vs_ifind_corr"] = float(full[["gain_calc_pct", "核验区间涨幅%(前复权)"]].corr().iloc[0, 1])

    # Q1 行业
    ind = full["ths_industry"].value_counts()
    S["industry_top"] = ind.head(10).to_dict()
    S["industry_top2_share"] = float(ind.head(2).sum()) / len(full)

    # Q2 上市时间
    # 口径说明（2026-10-01 复查补记）：listed_before_2016 是"年份口径"（上市年 ≤2015）；
    # 严格窗口口径以 start_cap 锚定为准（listed_before_window 列）——2016 年内早于 09-01
    # 上市的新易盛(2016-03-03)、兆易创新(2016-08-18)按严格口径也是老股，故 13 ≠ 11。
    full["list_year"] = full["list_date"].str[:4].astype(int)
    S["listed_before_2016"] = int((full["list_year"] <= 2015).sum())
    S["listed_before_window"] = int(full["listed_before_window"].sum())
    S["listed_inwindow_strict"] = int((~full["listed_before_window"].astype(bool)).sum())
    S["listed_2016_2020"] = int(full["list_year"].between(2016, 2020).sum())
    S["listed_2021_plus"] = int((full["list_year"] >= 2021).sum())
    S["list_year_counts"] = full["list_year"].value_counts().sort_index().to_dict()

    # Q3 起点市值
    cap = full["start_mktcap_yi"].dropna()
    bins = [0, 30, 50, 100, 300, 1e9]
    labels = ["<30亿", "30-50亿", "50-100亿", "100-300亿", ">300亿"]
    capbucket = pd.cut(cap, bins=bins, labels=labels, right=False)
    S["start_cap_buckets"] = capbucket.value_counts().reindex(labels).to_dict()
    S["start_cap_median_yi"] = float(cap.median())
    S["under_50yi_share"] = float((cap < 50).mean())

    # Q4 时间-10倍
    t = full.loc[full["hit10x"], "days_to_10x"]
    S["days_to_10x_median"] = float(t.median()) if len(t) else None
    S["days_to_10x_q25"] = float(t.quantile(0.25)) if len(t) else None
    S["days_to_10x_q75"] = float(t.quantile(0.75)) if len(t) else None
    S["days_to_10x_min"] = float(t.min()) if len(t) else None
    S["days_to_10x_max"] = float(t.max()) if len(t) else None

    # Q5 途中最大回撤
    m = full["mdd_to_10x_pct"].dropna()
    S["mdd_median_pct"] = float(m.median())
    S["mdd_worse_than_50pct_share"] = float((m <= -50).mean())
    S["mdd_worst"] = float(m.min())

    json.dump(S, open(os.path.join(DATA, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(json.dumps(S, ensure_ascii=False, indent=2, default=str))

    # ============ 图表（英文标签避免中文字体缺失） ============
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Profile of 88 A-share ten-baggers (2016-09-01 ~ 2026-08-31, survivorship-biased)",
                 fontsize=13)

    ax = axes[0][0]
    ind.head(12).iloc[::-1].plot.barh(ax=ax, color="#4472C4")
    ax.set_title("Q1 Industry (THS level-1) top 12")
    ax.set_xlabel("number of stocks")

    ax = axes[0][1]
    ly = full["list_year"].value_counts().sort_index()
    ax.bar(ly.index.astype(str), ly.values, color="#ED7D31")
    ax.set_title("Q2 Listing year cohort")
    ax.set_xlabel("year")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=60)

    ax = axes[1][0]
    t10 = (full.loc[full["hit10x"], "days_to_10x"] / 365.25).dropna()
    ax.hist(t10, bins=20, color="#70AD47", edgecolor="white")
    ax.axvline(t10.median(), color="red", ls="--",
               label=f"median {t10.median():.1f}y")
    ax.set_title("Q4 Time from window start to first 10x (close-based)")
    ax.set_xlabel("years")
    ax.set_ylabel("count")
    ax.legend()

    ax = axes[1][1]
    mdd = full["mdd_to_10x_pct"].dropna()
    ax.hist(mdd, bins=24, color="#C00000", edgecolor="white")
    ax.axvline(mdd.median(), color="black", ls="--",
               label=f"median {mdd.median():.1f}%")
    ax.set_title("Q5 Max drawdown suffered before reaching 10x")
    ax.set_xlabel("max drawdown (%)")
    ax.set_ylabel("count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "profile_4panel.png"), dpi=150)

    fig2, ax = plt.subplots(figsize=(8, 5))
    order = ["<30亿", "30-50亿", "50-100亿", "100-300亿", ">300亿"]
    cnt = capbucket.value_counts().reindex(order).fillna(0)
    ax.bar(order, cnt.values, color="#7030A0")
    for i, v in enumerate(cnt.values):
        ax.text(i, v + 0.5, str(int(v)), ha="center")
    ax.set_title("Q3 Starting market cap (old: 2016-09-01; new: listing day)")
    ax.set_ylabel("count")
    fig2.tight_layout()
    fig2.savefig(os.path.join(CHARTS, "start_mktcap.png"), dpi=150)
    print(f"[done] 图表已存 {CHARTS}/")


if __name__ == "__main__":
    main()
