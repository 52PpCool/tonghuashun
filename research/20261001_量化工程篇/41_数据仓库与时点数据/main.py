#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py —— 第41课：数据仓库的三场实验

实验1【时点股票池与幸存者偏差的规模】用 instrument(上市日,退市日) 做 PIT 查询：
    "2016-09-01 / 2018-12-28 当天可买的股票池" vs "用今天的在市名单重建历史池"——
    差值就是被退市"吃掉"的股票数（课05 幸存者偏差的定量版）。
实验2【复权因子表】adj_factor = 不复权收盘/前复权收盘（同日拉取）。
    展示：①因子在分红除权日跳变；②同一股票"不复权涨幅"与"前复权涨幅"差多少——
    这正是第37课"10倍判定对复权口径敏感"（86/88 触线、华盛昌/高争民爆两口径分叉）的机理。
实验3【行业快照的债】对照 137 只候选的"同花顺一级"（课37查询时点快照）与
    深交所官方行业（今日快照）——分类既依赖时点也依赖来源；38课遗留的
    "前视分类偏差"无法用免费数据还清，本实验量化暴露它。

运行： python main.py --build   （首次：建仓+联网拉30只不复权/前复权，约2分钟）
       python main.py --report  （三实验+图表，纯本地）
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(HERE)
sys.path.insert(0, ARCH)
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import warehouse as wh
from quantlab.data import CACHE_DIR

CACHE_STOCKS = sorted({f.split("_")[1] for f in os.listdir(CACHE_DIR) if f.startswith("stock_")})


def cmd_build(con):
    stats = wh.build_instruments(con)
    print(f"[build] instrument: 在市 {stats['在市']} 只 + 退市 {stats['退市']} 只"
          f"（沪市退市口径={stats['沪退市口径']}）")
    n = wh.ingest_cache_stocks(con)
    print(f"[build] daily_price: 缓存 {n} 只个股（前复权，2015 起）")
    n = wh.build_adj_factor(con, CACHE_STOCKS)
    print(f"[build] raw_price + adj_factor: {n}/{len(CACHE_STOCKS)} 只（同日双口径拉取）")
    n = wh.load_industry_snapshot(con)
    print(f"[build] industry_snapshot: {n} 只（课37/38候选，同花顺一级）")


# ---------------------------------------------------------------------------
def exp1_pool(con):
    print("\n" + "=" * 72)
    print("实验1【时点股票池与幸存者偏差的规模】")
    rows = []
    for d in ["2016-09-01", "2018-12-28", "2021-12-31"]:
        pit = con.execute("SELECT COUNT(*) FROM instrument WHERE list_date<=? "
                          "AND (delist_date IS NULL OR delist_date>?)", [d, d]).fetchone()[0]
        naive = con.execute("SELECT COUNT(*) FROM instrument WHERE list_date<=? "
                            "AND delist_date IS NULL", [d,]).fetchone()[0]
        rows.append({"时点": d, "PIT可选池(含后来退市)": pit,
                     "用今天的名单重建": naive, "被退市吃掉": pit - naive,
                     "偏差比例%": round((pit - naive) / pit * 100, 2)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "exp1_时点股票池.csv"), index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False))

    # 池子规模按月演化 + 期间退市数（图）
    months = pd.date_range("2016-01-31", "2026-09-30", freq="ME")
    sizes, gone = [], []
    for m in months:
        s = con.execute("SELECT COUNT(*) FROM instrument WHERE list_date<=? "
                        "AND (delist_date IS NULL OR delist_date>?)", [m, m]).fetchone()[0]
        g = con.execute("SELECT COUNT(*) FROM instrument WHERE delist_date IS NOT NULL "
                        "AND delist_date>? AND list_date<=?", [m, m]).fetchone()[0]
        sizes.append(s); gone.append(g)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(months, sizes, label="PIT 可选股票池（月末）", lw=1.5)
    ax.fill_between(months, [s - g for s, g in zip(sizes, gone)], label="其中：若用今天的名单会漏掉的退市股", alpha=0.35)
    ax.set_title("时点股票池与退市股：幸存者偏差是'看不见的分母'（数据源：沪深交易所上市/退市清单）")
    ax.set_ylabel("股票数"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_PIT股票池.png"), dpi=150)
    return tab


def exp2_factor(con):
    print("\n" + "=" * 72)
    print("实验2【复权因子表：两口径涨幅的机理】")
    rows = []
    # 用 pandas 逐股算：不复权累计涨幅 vs 前复权累计涨幅（同一窗口）
    for code in CACHE_STOCKS:
        raw = con.execute("SELECT date, close FROM raw_price WHERE code=? ORDER BY date",
                          [code]).fetchdf()
        qfq = con.execute("SELECT date, close FROM daily_price WHERE code=? ORDER BY date",
                          [code]).fetchdf()
        if len(raw) < 100 or len(qfq) < 100:
            continue
        g_raw = raw["close"].iloc[-1] / raw["close"].iloc[0] - 1
        g_qfq = qfq["close"].iloc[-1] / qfq["close"].iloc[0] - 1
        name = con.execute("SELECT name FROM instrument WHERE code=?", [code]).fetchone()
        rows.append({"code": code, "name": name[0] if name else "",
                     "不复权涨幅%": round(g_raw * 100, 1), "前复权涨幅%": round(g_qfq * 100, 1),
                     "差(分红再投效应)pp": round((g_qfq - g_raw) * 100, 1)})
    tab = pd.DataFrame(rows).sort_values("差(分红再投效应)pp", ascending=False)
    tab.to_csv(os.path.join(OUT, "exp2_两口径涨幅.csv"), index=False, encoding="utf-8-sig")
    print(tab.head(8).to_string(index=False))
    print(f"... 共 {len(tab)} 只；两口径差的中位数 {tab['差(分红再投效应)pp'].median():.1f}pp，"
          f"最大 {tab['差(分红再投效应)pp'].max():.1f}pp")
    # 因子跳变示例（茅台）
    f = con.execute("SELECT date, factor FROM adj_factor WHERE code='600519' ORDER BY date").fetchdf()
    f["chg"] = f["factor"].pct_change()
    jumps = f[f["chg"].abs() > 0.01]
    print(f"\n600519 复权因子跳变日（|变动|>1%，即分红/除权日）共 {len(jumps)} 次，"
          f"最近 3 次：")
    print(jumps.tail(3).to_string(index=False))
    return tab


def exp3_industry(con):
    print("\n" + "=" * 72)
    print("实验3【行业快照的债：分类既依赖时点也依赖来源】")
    # 同花顺一级（课37快照） vs 深交所行业（今日快照）：137候选中的深市股票
    alt = con.execute("""
        SELECT split_part(i.code, '.', 1) AS code6, i.ths_industry, c.csrc_industry
        FROM industry_snapshot i
        JOIN (SELECT DISTINCT code, csrc_industry FROM industry_alt) c
          ON split_part(i.code, '.', 1) = c.code
    """).fetchdf()
    if len(alt):
        alt["一致"] = alt["ths_industry"] == alt["csrc_industry"]
        agree = alt["一致"].mean() * 100
        alt.to_csv(os.path.join(OUT, "exp3_行业两来源对照.csv"), index=False, encoding="utf-8-sig")
        print(f"137候选中的深市股票 {len(alt)} 只：同花顺一级 vs 深交所行业 完全一致仅 {agree:.1f}%")
        print("  （行业口径天然多义：'电子' vs '计算机、通信和其他电子设备制造业'是两套语言；")
        print("    38课的前视分类偏差（2016年的行业归法）需历史行业变更数据，免费源无解，债务保留）")
    else:
        print("  （industry_alt 为空：请先 --build）")
    return alt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="建仓（联网）")
    ap.add_argument("--report", action="store_true", help="三实验（本地）")
    args = ap.parse_args()
    con = wh.connect()
    if args.build:
        cmd_build(con)
    if args.report:
        exp1_pool(con)
        exp2_factor(con)
        exp3_industry(con)
        print("\n[done] 产出已写入 outputs/")
