#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py —— 第42课：画像变策略（收官课）

⚠️ 先读这段（循环论证警告，本课的灵魂）：
  本课把第37/38课的"十倍股画像"（硬科技行业/次新/小市值三特征）变成一个可回测的
  选股策略。但 137 只候选本身是 2026 年回看"近10年涨幅"筛出来的——**用结果样本
  回测选股策略是循环论证**：策略"选中"的股票本来就是按"后来涨了多少"挑出来的。
  所以本实验回答的不是"这策略能不能赚钱"，而是一个更干净的问题：
  **"区分度"（38课已验证）变成"组合持仓"后还剩多少？**——即画像分层在
  组合层面（等权、含成本、含涨跌停/手数约束）是否仍然单调。
  真正可投资的版本需要：PIT 股票池（第41课仓库）+ PIT 行业分类（免费数据无解，
  38/41 课遗留债）+ 当期可计算的滚动特征。这是整个工程篇的闭环。

实验设计：
  - 母体：137 只初选候选中**有完整特征与价格的 116 只**（88十倍 + 28对照）；
    其余 21 只（核验涨幅<500% 的误选股）缺行业/市值特征，不纳入——注意这会
    让低分层的构成偏乐观，诚实记录。
  - 打分（2016-09-01 锚点，特征为慢变量，故一次性买入持有10年，不做再平衡）：
    score = [行业∈{电子,通信,机械设备}] + [2016-09-01后上市] + [起点市值<50亿] ∈ {0,1,2,3}
  - 分层组合：score=3/2/1/0 各成一个等权组合（锚点日起每日对未建仓股票挂等权买单，
    次新上市首日起自动追单——一字板拒单由引擎撮合台处理，第39课规则全部生效）。
  - 对照：全部116只等权（含成本 / 无成本）。
  - 引擎：quantlab.engine 多标的回测（第39课），现金500万（避免116等权时高价股
    一手买不起的手数截断扭曲权重）。

运行： python main.py   （约1-2分钟）
"""
import os
import sys

import numpy as np
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

from quantlab.engine import Backtester, BrokerConfig, Strategy
from quantlab.data import load_price_csv

P37 = os.path.join(os.path.dirname(ARCH), "20261001_十倍股画像")
P38 = os.path.join(os.path.dirname(ARCH), "20261001_伪十倍股对照画像")
HARD_TECH = {"电子", "通信", "机械设备"}
ANCHOR = pd.Timestamp("2016-09-01")
INIT_CASH = 5_000_000.0


def build_universe() -> pd.DataFrame:
    """合并两轮画像特征 → 116 只宇宙 + 画像打分。"""
    a = pd.read_csv(os.path.join(P37, "data", "profile_full.csv"), dtype={"code": str})
    b = pd.read_csv(os.path.join(P38, "data", "control_full.csv"), dtype={"code": str})
    u = pd.concat([a, b], ignore_index=True)
    u["code6"] = u["code"].str.split(".").str[0]
    u["list_date"] = u["list_date"].astype(str)
    u["hard_tech"] = u["ths_industry"].isin(HARD_TECH).astype(int)
    u["new_stock"] = (u["list_date"] >= "20160901").astype(int)
    u["small_cap"] = (u["start_mktcap_yi"] < 50).astype(int)
    u["score"] = u["hard_tech"] + u["new_stock"] + u["small_cap"]
    return u


def load_prices(codes: list) -> dict:
    """从两轮画像的留档价格目录加载日K（新浪源前复权）。"""
    data = {}
    dirs = [os.path.join(P37, "data", "prices"), os.path.join(P38, "data", "prices")]
    for c in codes:
        for d in dirs:
            f = os.path.join(d, f"{c}.csv")
            if os.path.exists(f):
                data[c] = load_price_csv(f)
                break
    return data


class EqualWeightAtAnchor(Strategy):
    """锚点日起，对未建仓的标的每日挂等权买单（剩余现金均摊）。
    次新股上市首日进入日历后自动开始追单；一字板拒单后次日继续（引擎撤旧挂新）。"""

    def __init__(self, codes, slippage=1e-3):
        self.codes = list(codes)
        self.slippage = slippage

    def on_bar(self, ctx):
        remaining = [c for c in self.codes
                     if ctx.position(c) == 0 and not ctx.has_pending(c, "BUY")]
        if not remaining:
            return
        per = ctx.cash / len(remaining)
        for c in remaining:
            bar = ctx.bar(c)
            if bar is None:
                continue                     # 未上市/停牌
            px = float(bar["close"]) * (1 + self.slippage)
            sh = int(per / px / 100) * 100
            if sh > 0:
                ctx.buy(c, shares=sh)


def run_portfolio(data: dict, codes: list, name: str, costs=True) -> "Result":
    cfg = BrokerConfig(cash=INIT_CASH)
    if not costs:
        cfg.commission, cfg.min_commission, cfg.stamp_tax, cfg.slippage = 0.0, 0.0, 0.0, 0.0
    bt = Backtester({c: data[c] for c in codes}, cfg, exec_model="next_open", name=name)
    return bt.run(EqualWeightAtAnchor(codes))


def main():
    u = build_universe()
    print(f"宇宙：{len(u)} 只（score分布: {u['score'].value_counts().sort_index().to_dict()}）")
    data = load_prices(u["code6"].tolist())
    u = u[u["code6"].isin(data)].reset_index(drop=True)
    print(f"有价格者：{len(u)} 只（score分布: {u['score'].value_counts().sort_index().to_dict()}）")

    rows, curves = [], {}
    # 分层组合
    for s in [3, 2, 1, 0]:
        codes = u.loc[u["score"] == s, "code6"].tolist()
        if not codes:
            continue
        r = run_portfolio(data, codes, f"score={s}")
        m = r.metrics()
        rows.append({"组合": f"画像分层 score={s}", "n": len(codes), "年化%": round(m["ann_ret_pct"], 2),
                     "最大回撤%": round(m["max_dd_pct"], 2), "成交笔数": m["n_filled"],
                     "拒单数": m["n_rejected"]})
        curves[f"score={s} (n={len(codes)})"] = r.equity / INIT_CASH
    # 对照：全部等权（含成本 / 无成本）
    for name, costs in [("全部116等权(含成本)", True), ("全部116等权(无成本)", False)]:
        codes = u["code6"].tolist()
        r = run_portfolio(data, codes, name, costs=costs)
        m = r.metrics()
        rows.append({"组合": name, "n": len(codes), "年化%": round(m["ann_ret_pct"], 2),
                     "最大回撤%": round(m["max_dd_pct"], 2), "成交笔数": m["n_filled"],
                     "拒单数": m["n_rejected"]})
        curves[name] = r.equity / INIT_CASH

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "exp1_画像分层组合.csv"), index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False))

    # 分层单调性（Spearman：score vs 年化）
    tiers = tab[tab["组合"].str.startswith("画像分层")]
    from scipy import stats as sps
    rho, p = sps.spearmanr(tiers["组合"].str.split("=").str[1].astype(int), tiers["年化%"])
    print(f"\n分层单调性：Spearman ρ={rho:.2f}（p={p:.3f}，n=4 组，仅示意不做推断主打）")

    fig, ax = plt.subplots(figsize=(11.5, 5))
    for k, v in curves.items():
        ax.plot(v.index, v.values, lw=1.4, label=k)
    ax.set_title("画像分数 → 组合净值（2016-09-01锚点等权买入持有10年，含A股约束成本）\n"
                 "循环论证警告：母体按'后来涨幅'筛出，本图演示区分度→组合的衰减，不构成可投资策略")
    ax.set_ylabel("净值（初始500万）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_画像分层净值.png"), dpi=150)
    print(f"\n[done] 产出已写入 {OUT}/")


if __name__ == "__main__":
    main()
