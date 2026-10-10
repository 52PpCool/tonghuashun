#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第39课 main.py —— 自写引擎的三场考试

实验1【对账】同一套双均线（SMA20/60，平安银行，成本口径与课02/21完全一致），
    分别用本引擎的 close 模式（信号日收盘成交）与 next_open 模式（次日开盘成交）
    复跑，与课02手写向量化版（净值1.229/年化1.84%）、课21 backtrader coc 版
    （1.2286）、bt 默认次日开盘版（1.0907）多方对表。验收：年化差 < 0.1pp。
实验2【约束消融】T+1 / 涨跌停拒单 / 手数取整 / 滑点 逐个关掉，
    双均线（平安）与海龟（茅台，移植课03状态机）各跑 6 组配置——
    回答"每一条A股撮合规则值多少年化收益"。
实验3【不可实现收益】用课37留档的次新股价格：
    海天精工（上市后连续一字板）与中际旭创（停牌+复牌连板）——
    挂单在封板日被逐日拒单、穿过停牌冻结，量化"纸面收益里有多少根本买不到"。

运行：`python main.py`（约 20 秒），产出在 outputs/。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans Mono CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(HERE)
sys.path.insert(0, ARCH)
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

from quantlab.engine import Backtester, BrokerConfig, Strategy
from quantlab.data import load_cache, load_profile_price, QUANT_ROOT

M02 = os.path.join(QUANT_ROOT, "02_双均线策略与向量化回测", "outputs", "metrics.csv")
M21 = os.path.join(QUANT_ROOT, "21_backtrader事件驱动回测", "outputs", "bt_metrics.csv")


# ===========================================================================
# 策略定义（on_bar 里只写"决策"，撮合与账户全部交给引擎）
# ===========================================================================
class SmaCross(Strategy):
    """SMA(20)>SMA(60) 全仓、否则空仓。信号序列与课02完全同款（NaN→0，预热期空仓）。"""

    def __init__(self, code, sig):
        self.code, self.sig, self.prev = code, sig, 0.0

    def on_bar(self, ctx):
        s = float(self.sig.get(ctx.date, 0.0))
        if s != self.prev:
            if s == 1.0:
                ctx.buy(self.code)
            else:
                ctx.sell(self.code)
            self.prev = s


class Turtle(Strategy):
    """课03海龟状态机的引擎移植：20日突破入场 / 2N止损 / 10日通道离场 / 0.5N金字塔加仓。
    与课03一致：收盘出信号、次日开盘成交（= 本引擎 next_open 模式的天然语义）。"""

    def __init__(self, code, risk_pct=0.01, entry_win=20, exit_win=10,
                 add_step=0.5, max_units=4, stop_mult=2.0):
        self.code, self.risk_pct = code, risk_pct
        self.entry_win, self.exit_win = entry_win, exit_win
        self.add_step, self.max_units, self.stop_mult = add_step, max_units, stop_mult
        self.units, self.n0, self.stop_price, self.last_entry = 0, 0.0, 0.0, 0.0

    def _unit_shares(self, equity, n0, px):
        # 海龟仓位公式的股票版（课03同款）：股数 = 权益×1% / N
        # （每股波动 1N = 每股盈亏 N 元，预算/N 即"亏 1N 恰好亏 1% 权益"的股数）
        # 现金不足时由引擎在撮合时自动缩到手数上限，策略侧无需关心
        return int(equity * self.risk_pct / n0 // 100) * 100

    @staticmethod
    def _shrink_to_cash(sh, cash, px):
        """现金上限：最多买到现金允许的手数（课03同款 affordability 截断）。"""
        affordable = int(cash / (px * (1 + 2.5e-4)) // 100) * 100
        return max(min(sh, affordable), 0)

    def on_bar(self, ctx):
        h = ctx.history(self.code, 25)
        if len(h) < 22:
            return
        pc = h["close"].shift(1)
        tr = pd.concat([h["high"] - h["low"], (h["high"] - pc).abs(),
                        (h["low"] - pc).abs()], axis=1).max(axis=1)
        n0 = float(tr.tail(20).mean())          # N = ATR(20)（含今日，与课03一致）
        if not np.isfinite(n0):
            return
        don_up = float(h["high"].iloc[-(self.entry_win + 1):-1].max())   # 前20日最高（不含今天）
        don_low = float(h["low"].iloc[-(self.exit_win + 1):-1].min())    # 前10日最低
        c = float(h["close"].iloc[-1])
        if self.units == 0:
            if c > don_up:
                self.n0 = n0
                sh = self._shrink_to_cash(self._unit_shares(ctx.equity, n0, c), ctx.cash, c)
                if sh > 0:
                    ctx.buy(self.code, shares=sh)
        elif c < self.stop_price:
            ctx.sell(self.code)                  # 2N止损，次日开盘全出
        elif c < don_low:
            ctx.sell(self.code)                  # 10日通道下破
        elif self.units < self.max_units and c >= self.last_entry + self.add_step * self.n0:
            sh = self._shrink_to_cash(self._unit_shares(ctx.equity, self.n0, c), ctx.cash, c)
            if sh > 0:
                ctx.buy(self.code, shares=sh)

    def on_fill(self, tr):
        if tr["side"] == "BUY":                  # 按实际成交价（含滑点）上移止损锚
            self.units += 1
            self.last_entry = tr["price"]
            self.stop_price = tr["price"] - self.stop_mult * self.n0
        else:
            self.units, self.stop_price, self.last_entry = 0, 0.0, 0.0


class ChaseBuy(Strategy):
    """从第 start_i 根bar起，只要空仓就每天挂市价买单（引擎自动撤旧挂新）。
    用于实验3：一字板期间逐日拒单，第一个没封死的日子才上得了车。"""

    def __init__(self, code, start_i=0):
        self.code, self.start_i = code, start_i

    def on_bar(self, ctx):
        if ctx.i >= self.start_i and ctx.position(self.code) == 0:
            ctx.buy(self.code)


def sma_signal(close, fast=20, slow=60):
    return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)


# ===========================================================================
# 实验1：对账
# ===========================================================================
def exp1():
    print("=" * 72)
    print("实验1【对账】双均线 SMA20/60 平安银行，成本口径 = 课02/21（佣金万2.5双边、"
          "印花税卖千0.5、滑点单边0.1%）")
    px = load_cache("000001_平安银行")
    sig = sma_signal(px["close"])

    # 对齐课02口径：课02未建模涨跌停/T+1 → 关闭这两条，只比"撮合框架"本身
    cfg02 = BrokerConfig(t_plus_1=False, limit_rules=False)
    r_close = Backtester({"000001": px}, cfg02, exec_model="close",
                         name="本引擎·收盘成交").run(SmaCross("000001", sig))
    # 对齐 bt 默认口径：次日开盘成交 + 滑点钳制在当日高低价内（bt slip_match 行为）
    cfg_bt = BrokerConfig(t_plus_1=False, limit_rules=False, slip_clamp=True)
    r_open = Backtester({"000001": px}, cfg_bt, exec_model="next_open",
                        name="本引擎·次日开盘").run(SmaCross("000001", sig))

    m02 = pd.read_csv(M02).set_index("版本").loc["正确版(信号次日执行)"]
    mbt = pd.read_csv(M21).set_index("版本")
    bt_coc = mbt.loc["bt对齐版(coc收盘成交,恒定滑点)"]
    bt_def = mbt.loc["bt默认市价单(次日开盘成交)"]

    mc, mo = r_close.metrics(), r_open.metrics()
    rows = [
        {"版本": "① 课02 手写向量化(信号次日收盘)", "来源": "课02 metrics.csv",
         "期末净值": m02["期末净值"], "年化%": m02["年化收益率(%)"], "最大回撤%": m02["最大回撤(%)"]},
        {"版本": "② bt coc 收盘成交(恒定滑点)", "来源": "课21 bt_metrics.csv",
         "期末净值": bt_coc["期末净值"], "年化%": bt_coc["年化收益率(%)"], "最大回撤%": bt_coc["最大回撤(%)"]},
        {"版本": "③ 本引擎 close 模式", "来源": "本课",
         "期末净值": mc["final_nav"], "年化%": mc["ann_ret_pct"], "最大回撤%": mc["max_dd_pct"]},
        {"版本": "④ bt 默认次日开盘成交", "来源": "课21 bt_metrics.csv",
         "期末净值": bt_def["期末净值"], "年化%": bt_def["年化收益率(%)"], "最大回撤%": bt_def["最大回撤(%)"]},
        {"版本": "⑤ 本引擎 next_open 模式", "来源": "本课",
         "期末净值": mo["final_nav"], "年化%": mo["ann_ret_pct"], "最大回撤%": mo["max_dd_pct"]},
    ]
    tab = pd.DataFrame(rows)
    tab["年化%"] = tab["年化%"].astype(float).round(2)
    tab["vs①差(pp)"] = (tab["年化%"] - float(m02["年化收益率(%)"])).round(2)
    tab.loc[tab.index >= 3, "vs①差(pp)"] = np.nan
    tab.to_csv(os.path.join(OUT, "exp1_对账.csv"), index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False))
    d = abs(mc["ann_ret_pct"] - float(m02["年化收益率(%)"]))
    print(f"\n[验收] ③vs① 年化差 = {d:.3f} pp（{'通过' if d < 0.1 else '未通过'}，阈值0.1pp）；"
          f"⑤vs④ 年化差 = {abs(mo['ann_ret_pct'] - float(bt_def['年化收益率(%)'])):.2f} pp；"
          f"成交时点本身带来的年化差（③-⑤）= {mc['ann_ret_pct'] - mo['ann_ret_pct']:.2f} pp")

    fig, ax = plt.subplots(figsize=(11, 5))
    (r_close.equity / 1e6).plot(ax=ax, label=f"本引擎·收盘成交（年化 {mc['ann_ret_pct']:.2f}%）", lw=1.4)
    (r_open.equity / 1e6).plot(ax=ax, label=f"本引擎·次日开盘成交（年化 {mo['ann_ret_pct']:.2f}%）", lw=1.4)
    ax.set_title("同一个双均线，两种成交时点：撮合规则不是实现细节（平安银行，成本同课02）")
    ax.set_ylabel("净值")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_对账_成交时点对比.png"), dpi=150)
    return r_close, r_open


# ===========================================================================
# 实验2：约束消融
# ===========================================================================
def exp2():
    print("\n" + "=" * 72)
    print("实验2【消融】逐条关掉撮合规则，看每条规则'值'多少年化（exec=next_open）")
    px_sma = load_cache("000001_平安银行")
    px_tur = load_cache("600519_贵州茅台")
    cases = [
        ("双均线@平安银行", "000001", px_sma, lambda: SmaCross("000001", sma_signal(px_sma["close"])),
         dict(slippage=1.0e-3)),
        ("海龟@贵州茅台", "600519", px_tur, lambda: Turtle("600519"),
         dict(slippage=5.0e-4, min_commission=0.0)),   # 课03口径（滑点万5、无最低佣金）
    ]
    toggles = [
        ("基线(全部规则开)", {}),
        ("关T+1", dict(t_plus_1=False)),
        ("关涨跌停拒单", dict(limit_rules=False)),
        ("关手数(允许1股)", dict(board_lot=1)),
        ("关滑点", dict(slippage=0.0)),
        ("全关(裸价差)", dict(t_plus_1=False, limit_rules=False, board_lot=1, slippage=0.0)),
    ]
    rows = []
    for name, code, px, mk, base_kwargs in cases:
        base = None
        for tname, kw in toggles:
            cfgkw = dict(base_kwargs)
            cfgkw.update(kw)
            r = Backtester({code: px}, BrokerConfig(**cfgkw),
                           exec_model="next_open", name=f"{name}-{tname}").run(mk())
            m = r.metrics()
            if tname.startswith("基线"):
                base = m
            rows.append({"策略": name, "配置": tname, "年化%": round(m["ann_ret_pct"], 2),
                         "最大回撤%": round(m["max_dd_pct"], 2), "成交笔数": m["n_filled"],
                         "拒单数": m["n_rejected"],
                         "Δ年化 vs 基线(pp)": round(m["ann_ret_pct"] - base["ann_ret_pct"], 2)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "exp2_消融.csv"), index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False))
    tur_base = tab[(tab["策略"] == "海龟@贵州茅台") & (tab["配置"].str.startswith("基线"))]
    m03 = pd.read_csv(os.path.join(QUANT_ROOT, "03_海龟交易法则", "outputs", "metrics.csv"))
    a03 = float(m03.set_index("组合").loc["海龟策略(系统一)", "年化收益"]) * 100
    print(f"\n[参照] 本引擎海龟基线年化 {float(tur_base['年化%'].iloc[0]):.2f}% "
          f"vs 课03手写状态机 {a03:.2f}%（差异来自估值时点/最低佣金等微小口径）")
    # 拒单原因分布（基线）：确认拒单不是策略侧现金bug
    for name, code, px, mk, base_kwargs in cases:
        r = Backtester({code: px}, BrokerConfig(**base_kwargs),
                       exec_model="next_open", name="diag").run(mk())
        rc = r.orders[r.orders["status"] == "REJECTED"]["reason"].value_counts()
        print(f"[拒单原因] {name}: {rc.to_dict() if len(rc) else '无'}")
    return tab


# ===========================================================================
# 实验3：不可实现收益（次新连板与停牌）
# ===========================================================================
def _gap_report(px, code):
    ds = px.index.to_series()
    gap = ds.diff().dt.days
    big = gap[gap > 15]
    lines = [f"  {code} 价格序列 {ds.iloc[0].date()} ~ {ds.iloc[-1].date()}，共 {len(ds)} 根bar"]
    for d, g in big.items():
        i = ds.index.get_loc(d)
        lines.append(f"  停牌式断档：{ds.iloc[i-1].date()} → {d.date()}（{int(g)} 个自然日无行情）")
    if not len(big):
        lines.append("  无 >15 日的断档")
    return lines


def _chase_case(px, code, start_i, rules, tag, exec_model="close"):
    r = Backtester({code: px}, BrokerConfig(limit_rules=rules, t_plus_1=False),
                   exec_model=exec_model, name=f"{code}-{tag}").run(ChaseBuy(code, start_i=start_i))
    m = r.metrics()
    fills = r.orders[r.orders["status"] == "FILLED"]
    rej = r.orders[r.orders["status"] == "REJECTED"]
    return r, fills, rej, m


def exp3():
    print("\n" + "=" * 72)
    print("实验3【不可实现收益】次新连板（海天精工601882）与 停牌/复牌连板（中际旭创300308）")
    log_lines = []
    results = {}

    # ---- 案例1：海天精工，第二个交易日起每天挂买单（打新未中想上车）----
    haitian = load_profile_price("601882")
    log_lines.append("【案例1】海天精工 601882（2016-11-07 上市，其后连续一字板）")
    log_lines += _gap_report(haitian, "601882")
    for rules, tag in [(True, "有涨跌停拒单"), (False, "无涨跌停拒单(纸面)")]:
        r, fills, rej, m = _chase_case(haitian, "601882", 1, rules, tag)
        results[("海天精工", tag)] = r
        log_lines.append(f"  [{tag}] 首次成交：{fills['fill_date'].iloc[0] if len(fills) else '从未'}"
                         + (f" @ {fills['fill_price'].iloc[0]:.3f}（拒单 {len(rej)} 次）" if len(fills) else ""))
        if len(fills):
            day1_close = float(haitian["close"].iloc[0])
            fp = float(fills["fill_price"].iloc[0])
            log_lines.append(f"      上市首日收盘 {day1_close:.2f} → 实际上车价 {fp:.3f}，"
                             f"多付 {(fp / day1_close - 1) * 100:.1f}% 的'连板溢价'；"
                             f"期末净值 {m['final_nav']:.3f}")

    # ---- 案例2：中际旭创，长期停牌后复牌 + 节后连续一字板 ----
    # 数据证据：课37取数窗口从 2016-08-15 开始，但 300308 首根bar是 2016-09-30
    # → 2016-08-15~09-29 无行情（长期停牌），09-30 为复牌首日；随后国庆休市，
    #   10-10 起开盘即一字板（o=h=l=c）连拉多日。
    xc = load_profile_price("300308")
    log_lines.append("\n【案例2】中际旭创 300308（长期停牌→09-30复牌→节后连续一字板）")
    log_lines.append("  [数据证据] 取数窗口起点 2016-08-15，价格首根bar 2016-09-30"
                     " → 停牌发生在窗口内前段（08-15~09-29 无行情）")
    log_lines += _gap_report(xc, "300308")
    for rules, tag in [(True, "有涨跌停拒单"), (False, "无涨跌停拒单(纸面)")]:
        r, fills, rej, m = _chase_case(xc, "300308", 1, rules, tag)
        results[("旭创", tag)] = r
        log_lines.append(f"  [{tag}] 首次成交：{fills['fill_date'].iloc[0] if len(fills) else '从未'}"
                         + (f" @ {fills['fill_price'].iloc[0]:.3f}（拒单 {len(rej)} 次，期末净值 {m['final_nav']:.2f}）"
                            if len(fills) else ""))
        if len(rej):
            log_lines.append("      被拒日期：" + ", ".join(rej["created"].dt.date.astype(str)))

    # ---- 案例2b：挂单穿过长假+连板（next_open：复牌首日 09-30 收盘挂单）----
    r_hold, fills_h, rej_h, _ = _chase_case(xc, "300308", 0, True, "挂单穿长假", exec_model="next_open")
    log_lines.append("\n  [挂单穿长假演示] next_open 模式，09-30（复牌首日）收盘后挂单："
                     "订单穿过国庆 7 天长假，10-10 起连续被一字板拒单，直到开板日才成交：")
    log_lines.append(r_hold.orders.head(10).to_string(index=False))

    with open(os.path.join(OUT, "exp3_订单簿.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print("\n".join(log_lines))

    # ---- 图：海天精工连板期 vs 上车点 ----
    r_on = results[("海天精工", "有涨跌停拒单")]
    r_off = results[("海天精工", "无涨跌停拒单(纸面)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    n_show = 60
    ax.plot(haitian.index[:n_show], haitian["close"].iloc[:n_show], lw=1.2,
            label="海天精工收盘价(前复权)")
    rej_days = r_on.orders[r_on.orders["status"] == "REJECTED"]["created"]
    rej_days = [d for d in rej_days if d in haitian.index]
    ax.scatter(rej_days, haitian.loc[rej_days, "close"], marker="x", color="red",
               s=45, zorder=5, label="挂单被拒(涨停封板)")
    f_on = r_on.orders[r_on.orders["status"] == "FILLED"]
    if len(f_on):
        fd = f_on["fill_date"].iloc[0]
        ax.scatter([fd], [haitian.loc[fd, "close"]], marker="*", color="green",
                   s=160, zorder=6, label="首次成交(有规则)")
    f_off = r_off.orders[r_off.orders["status"] == "FILLED"]
    if len(f_off):
        fd = f_off["fill_date"].iloc[0]
        ax.scatter([fd], [haitian.loc[fd, "close"]], marker="o", facecolor="none",
                   edgecolor="orange", s=150, zorder=6, label="纸面成交(无规则)")
    ax.set_title("海天精工：连板期的挂单全部被拒")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    (r_on.equity / 1e6).plot(ax=ax, label="有涨跌停规则(可实现)")
    (r_off.equity / 1e6).plot(ax=ax, label="无规则(纸面)", ls="--")
    ax.set_title("两条净值线的差距 = 不可实现收益")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_案例_次新连板.png"), dpi=150)
    return results


if __name__ == "__main__":
    exp1()
    exp2()
    exp3()
    print("\n[done] 产出已写入 outputs/")
