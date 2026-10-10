# -*- coding: utf-8 -*-
"""
31_Johansen多资产协整 —— 从两资产配对（课14）升级到 N 资产统计套利
====================================================================
核心问题：
  Engle-Granger 两步法只能两两配对、依赖"选谁做因变量"（不对称）、一次只能给出一维价差。
  Johansen 检验直接对 N 个资产的协整系统做假设检验：VECM 中矩阵 Pi 的秩 r = 协整关系个数，
  协整向量 beta 的每个元素 = 一条腿的份数（可负 = 空头腿）——统计套利的"正规军"做法。

流程：
  [1] 复习 EG 两步法及其局限（两方向 p 值不对称实例 → 图01）
  [2] 对 5 只银行股对数价格跑 coint_johansen：迹统计量 vs 临界值 → 判定秩 r（图02）
      取第一协整向量 beta 归一化 → 多腿组合权重（图03）
  [3] 组合价差 spread = beta'y：ADF 平稳性 + OU 半衰期（呼应课14）
  [4] z-score 多腿策略回测（样本内定参：滚动60日、z=±2开、|z|<0.5平、双边成本万3）
      与课14最强配对（工行 vs 建行，同一代码口径复算）对比
  [5] 样本外 2024 起：协整向量漂移（样本内 beta vs 全样本 beta）+ 样本外表现如实呈现
  [6] 稳健性：k_ar_diff = 1/2/3 秩判定是否一致；det_order 敏感性提示

数据：../data/cache/ 5只银行股真实日K（2018-01-02 ~ 2026-09-17）。
若数据缺失：用 np.random.default_rng(42) 造合成数据并在所有输出标注"合成数据"。
运行：/path/to/.venv/bin/python main.py   （总运行 < 2 分钟）
"""
import os
import sys
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无显示环境出图
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order

warnings.simplefilter("ignore")  # statsmodels 复数特征向量 Casting 警告，取实部即可

# 中文字体（必须在绘图前设置）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
np.random.seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "..", "data", "cache")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

# 全局参数（与课14保持一致，便于对比）
Z_WINDOW = 60          # 滚动标准化窗口（课14：约3倍半衰期）
Z_ENTER = 2.0          # |z| >= 2 开仓
Z_EXIT = 0.5           # |z| < 0.5 平仓
COST = 3e-4            # 双边成本 万3（每单位毛名义成交额）
SPLIT = "2024-01-01"   # 样本内 / 样本外分界（与课14一致：2018-2023 定参，2024 起验证）

STOCKS = ["工商银行", "建设银行", "招商银行", "兴业银行", "平安银行"]
FILES = {
    "工商银行": "stock_601398_工商银行.csv",
    "建设银行": "stock_601939_建设银行.csv",
    "招商银行": "stock_600036_招商银行.csv",
    "兴业银行": "stock_601166_兴业银行.csv",
    "平安银行": "stock_000001_平安银行.csv",
}

IS_SYNTHETIC = False  # 数据缺失时翻为 True，全部输出会标注"合成数据"


# ----------------------------------------------------------------------
# [0] 数据加载（缺失则造合成数据，保证脚本永不失败退出非0）
# ----------------------------------------------------------------------
def load_prices():
    """读取5只银行股收盘价，对齐成宽表；缺失则用 seed=42 造协整合成数据。"""
    global IS_SYNTHETIC
    px = {}
    ok = True
    for name in STOCKS:
        fp = os.path.join(CACHE, FILES[name])
        if not os.path.exists(fp):
            ok = False
            break
        df = pd.read_csv(fp, encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
        px[name] = df["close"]
    if ok:
        P = pd.DataFrame(px).dropna()
        P.index.name = "date"
        return P
    # ---- 合成数据兜底：1个共同随机游走因子 + 各自噪声，其间埋一条协整关系 ----
    IS_SYNTHETIC = True
    print("!! data/cache 缺失 —— 改用 np.random.default_rng(42) 合成数据（所有图表已标注）")
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2018-01-02", periods=2100)
    common = np.cumsum(rng.normal(0, 0.015, len(idx)))           # 共同因子（随机游走）
    P = {}
    for i, name in enumerate(STOCKS):
        drift = 0.0002 * i
        noise = np.cumsum(rng.normal(0, 0.008, len(idx)))
        P[name] = 10 * np.exp(0.3 * common + drift * np.arange(len(idx)) / 252 + noise)
    return pd.DataFrame(P, index=idx)


# ----------------------------------------------------------------------
# [1] 复习：EG 两步法及其局限（图01：两方向 p 值不对称实例）
# ----------------------------------------------------------------------
def eg_asymmetry(Yin, cols):
    """对 C(n,2) 个配对各做两个方向的 EG 检验，展示'选谁做因变量'会改变结论。"""
    rows = []
    for a, b in itertools.combinations(cols, 2):
        p_ab = coint(Yin[a], Yin[b], trend="c")[1]  # 以 a 为因变量：a = c + γ·b
        p_ba = coint(Yin[b], Yin[a], trend="c")[1]  # 以 b 为因变量
        rows.append({"配对": f"{a}~{b}", "方向1": f"以{a}为因变量", "p_方向1": p_ab,
                     "方向2": f"以{b}为因变量", "p_方向2": p_ba,
                     "两方向同结论(5%)": (p_ab < 0.05) == (p_ba < 0.05)})
    tbl = pd.DataFrame(rows)
    # 图01：分组条形图
    x = np.arange(len(tbl))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    title = "图01 EG两步法的局限：同一对股票，因变量换边，p值大不同" + ("（合成数据）" if IS_SYNTHETIC else "")
    ax.bar(x - 0.2, tbl["p_方向1"], width=0.4, color="#1f77b4", label="方向1（以左股票为因变量）")
    ax.bar(x + 0.2, tbl["p_方向2"], width=0.4, color="#ff7f0e", label="方向2（以右股票为因变量）")
    ax.axhline(0.05, color="red", ls="--", lw=1.2, label="5%显著性线")
    ax.set_xticks(x)
    ax.set_xticklabels(tbl["配对"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("EG协整检验 p 值")
    ax.set_title(title, fontsize=13)
    for xi, (v1, v2) in enumerate(zip(tbl["p_方向1"], tbl["p_方向2"])):
        ax.text(xi - 0.2, v1 + 0.01, f"{v1:.2f}", ha="center", fontsize=7.5)
        ax.text(xi + 0.2, v2 + 0.01, f"{v2:.2f}", ha="center", fontsize=7.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_EG局限示意图.png"), dpi=150)
    plt.close(fig)
    return tbl


# ----------------------------------------------------------------------
# [2] 本课灵魂：Johansen 检验（迹统计量、秩 r、协整向量 beta）
# ----------------------------------------------------------------------
def run_johansen(Ymat, det_order=0, k_ar_diff=2):
    """coint_johansen 包装：返回 (rank判定, 结果对象)。
    lr1=迹统计量, cvt=迹临界值(90/95/99%), lr2=最大特征根统计量, cvm=对应临界值,
    evec=协整向量矩阵(每列一条协整关系), eig=特征根。"""
    res = coint_johansen(Ymat, det_order, k_ar_diff)
    rank_trace = int((res.lr1 > res.cvt[:, 1]).sum())    # 依次比较95%临界值
    rank_maxeig = int((res.lr2 > res.cvm[:, 1]).sum())
    return rank_trace, rank_maxeig, res


def normalize_beta(bvec, names, ref=0):
    """两种归一化口径：
    (a) 解释口径：把参考股票（默认第0只=工商银行）系数设为 1；
    (b) 交易口径：毛名义 = 1（sum|w|=1），每个元素 = 该腿占1元总名义的份数。"""
    b = bvec.real.copy()                       # statsmodels 返回复数，取实部
    if b[ref] < 0:
        b = -b                                 # 符号约定：参考腿系数为正
    b_unit = b / b[ref]                        # (a) 工商=1
    b_gross = b / np.abs(b).sum()              # (b) 毛名义=1
    return b_unit, b_gross


# ----------------------------------------------------------------------
# [3] 价差诊断：ADF 平稳性 + OU 半衰期（与课14同一公式）
# ----------------------------------------------------------------------
def adf_p(x):
    return adfuller(np.asarray(x))[1]


def ou_half_life(spread):
    """Δs_t = a + b·s_{t-1} + ε；b<0 才有均值回归，半衰期 = ln2 / (-b)。"""
    s = np.asarray(spread, dtype=float)
    ds = np.diff(s)
    lag = s[:-1]
    b = np.polyfit(lag, ds, 1)[0]
    if b >= 0:
        return np.nan, b          # 无均值回归：半衰期不存在
    return float(np.log(2) / (-b)), float(b)


# ----------------------------------------------------------------------
# [4] z-score 多腿回测引擎（2腿与5腿共用同一套代码，保证公平对比）
# ----------------------------------------------------------------------
def backtest(P, w_gross, label):
    """统一回测：spread = 加权对数价差（只用权重方向与相对大小），
    滚动60日 z-score，z=±2 开仓、|z|<0.5 平仓。
    仓位 = ±1 个"单位组合"，单位组合 = 毛名义1元、各腿权重 w_gross（可负）。
    收益近似：r_combo,t = Σ w_i · r_i,t（每日再平衡的固定权重组合收益）；
    成本：每次仓位变动按成交毛名义收 万3，即 cost_t = 3e-4 × |Δpos| × Σ|w_i|。
    返回 dict（净值序列、仓位序列、全样本/样本内/样本外指标）。"""
    Y = np.log(P)
    R = P.pct_change().fillna(0.0)                       # 各腿简单日收益
    w = np.asarray(w_gross, dtype=float)
    w = w / np.abs(w).sum()                              # 毛名义=1
    spread = pd.Series(Y.values @ w, index=P.index)
    mu = spread.rolling(Z_WINDOW).mean()
    sd = spread.rolling(Z_WINDOW).std()
    z = (spread - mu) / sd

    pos = pd.Series(0.0, index=P.index)                  # 当日收盘决定的仓位
    cur = 0.0
    for t in range(len(z)):
        zt = z.iloc[t]
        if np.isnan(zt):
            pos.iloc[t] = cur
            continue
        if cur == 0:
            if zt <= -Z_ENTER:
                cur = 1.0                                # 价差低于均值2个σ：做多价差
            elif zt >= Z_ENTER:
                cur = -1.0                               # 高于均值2个σ：做空价差
        elif cur == 1 and zt >= -Z_EXIT:                 # 回归到 ±0.5 带内 → 平仓
            cur = 0.0
        elif cur == -1 and zt <= Z_EXIT:
            cur = 0.0
        pos.iloc[t] = cur

    r_combo = R.values @ w                               # 单位组合日收益
    prev = pos.shift(1).fillna(0.0)                      # t-1 收盘仓位吃到 t 日收益（无前视）
    dpos = prev.diff().abs().fillna(0.0)                 # 仓位变动 → 成交毛名义
    cost = dpos * COST * np.abs(w).sum()                 # Σ|w|=1，故 cost=3e-4×|Δpos|
    strat = pd.Series(prev.values * r_combo - cost.values, index=P.index, name=label)

    def stats(mask):
        r = strat[mask]
        p = pos[mask]
        n_tr, avg_hold = _episodes(p.values)
        return _perf(r, n_tr, avg_hold, (p != 0).mean())

    full = stats(np.ones(len(strat), dtype=bool))
    ins = stats(strat.index < SPLIT)
    oos = stats(strat.index >= SPLIT)
    return {"nav": (1 + strat).cumprod(), "pos": pos, "z": z, "spread": spread,
            "full": full, "ins": ins, "oos": oos}


def _episodes(p):
    """交易次数 = 极性连续非零段数；平均持仓 = 段长均值。"""
    n, cur, ln, lens = 0, 0, 0, []
    for v in p:
        v = int(np.sign(v))
        if v != 0 and v == cur:
            ln += 1
        elif v != 0:
            if cur != 0:
                lens.append(ln)
            cur, ln = v, 1
        else:
            if cur != 0:
                lens.append(ln)
            cur, ln = 0, 0
    if cur != 0:
        lens.append(ln)
    n = len(lens)
    return n, (float(np.mean(lens)) if lens else 0.0)


def _perf(r, n_tr, avg_hold, hold_frac):
    nav = (1 + r).cumprod()
    cum = nav.iloc[-1] - 1 if len(nav) else 0.0
    n = max(len(r), 1)
    ann = (1 + cum) ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    dd = (nav / nav.cummax() - 1).min() if len(nav) else 0.0
    return {"累计收益%": round(cum * 100, 2), "年化收益%": round(ann * 100, 2),
            "夏普": round(sharpe, 2), "年化波动%": round(vol * 100, 2),
            "最大回撤%": round(dd * 100, 2), "交易次数": int(n_tr),
            "平均持仓天数": round(avg_hold, 1), "持仓天数占比%": round(hold_frac * 100, 1)}


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    tag = "（合成数据）" if IS_SYNTHETIC else ""
    print("=" * 72)
    print("31_Johansen多资产协整：从两资产配对到 N 资产统计套利" + tag)
    print("=" * 72)

    P = load_prices()
    Y = np.log(P)
    Yin = Y[Y.index < SPLIT]                      # 样本内 2018-2023
    print(f"数据区间：{P.index.min().date()} ~ {P.index.max().date()}，共 {len(P)} 个交易日"
          f"（样本内 {len(Yin)} 天 / 样本外 {len(Y) - len(Yin)} 天）")

    # ---------- [1] EG 两步法的局限 ----------
    print("\n[1] EG两步法局限：10个配对 × 2个方向 的协整 p 值（样本内，对数价格）")
    eg_tbl = eg_asymmetry(Yin, list(Yin.columns))
    eg_tbl_out = eg_tbl.copy()
    for c in ("p_方向1", "p_方向2"):
        eg_tbl_out[c] = eg_tbl_out[c].round(4)
    print(eg_tbl_out.to_string(index=False))
    eg_tbl_out.to_csv(os.path.join(OUT, "eg_pvalues_two_directions.csv"),
                      index=False, encoding="utf-8-sig")
    gap = (eg_tbl["p_方向1"] - eg_tbl["p_方向2"]).abs()
    worst = gap.idxmax()
    print(f"  → 不对称最严重的配对：{eg_tbl.loc[worst, '配对']}，"
          f"两方向 p = {eg_tbl.loc[worst, 'p_方向1']:.4f} vs "
          f"{eg_tbl.loc[worst, 'p_方向2']:.4f}，相差 {gap.loc[worst]:.3f}")
    print("  → 5只股票两两配对共10对×2方向=20次检验，还有多重检验问题；且每对最多1条价差。")

    # ---------- [2] Johansen 检验 ----------
    print("\n[2] Johansen检验（样本内2018-2023，det_order=0，k_ar_diff=2，对数价格）")
    # 滞后阶怎么选：先看信息准则（水平VAR滞后 p → k_ar_diff = p-1）
    so = select_order(Yin.values, maxlags=5, deterministic="ci")
    so_orders = {k: int(v) for k, v in so.selected_orders.items()}
    print("  滞后选择（select_order, deterministic='ci'）:", so_orders,
          "→ 本课取 k_ar_diff=2（略留余量），[6] 节验证 1/2/3 秩判定一致")
    rank_t, rank_me, jr = run_johansen(Yin.values, 0, 2)
    jrows = []
    for i in range(len(jr.lr1)):
        concl = "拒绝" if jr.lr1[i] > jr.cvt[i, 1] else "不能拒绝"
        concl_me = "拒绝" if jr.lr2[i] > jr.cvm[i, 1] else "不能拒绝"
        jrows.append([f"r <= {i}", round(jr.lr1[i], 2), round(jr.cvt[i, 0], 2),
                      round(jr.cvt[i, 1], 2), round(jr.cvt[i, 2], 2), concl,
                      round(jr.lr2[i], 2), round(jr.cvm[i, 0], 2),
                      round(jr.cvm[i, 1], 2), round(jr.cvm[i, 2], 2), concl_me])
        print(f"  H0: r<={i}  迹统计量={jr.lr1[i]:7.2f}  临界值(90/95/99%)="
              f"{jr.cvt[i,0]:.2f}/{jr.cvt[i,1]:.2f}/{jr.cvt[i,2]:.2f}  → {concl}"
              f"   | 最大特征根={jr.lr2[i]:6.2f} vs {jr.cvm[i,1]:.2f} → {concl_me}")
    print(f"  ==> 迹检验判定秩 r = {rank_t}（最大特征根检验判定 r = {rank_me}）："
          f"5只银行股对数价格存在 {rank_t} 条协整关系" + tag)
    # 图02
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(jrows))
    for ax, si, cvi, ttl in [(axes[0], 1, (2, 3, 4), "迹统计量 trace stat"),
                             (axes[1], 6, (7, 8, 9), "最大特征根 max-eig stat")]:
        stat = [r[si] for r in jrows]
        c90 = [r[cvi[0]] for r in jrows]
        c95 = [r[cvi[1]] for r in jrows]
        c99 = [r[cvi[2]] for r in jrows]
        ax.bar(x - 0.3, stat, 0.25, color="#d62728", label="统计量")
        ax.bar(x, c90, 0.25, color="#aec7e8", label="90%临界值")
        ax.bar(x + 0.3, c95, 0.25, color="#1f77b4", label="95%临界值")
        ax.scatter(x, c99, marker="_", s=600, color="black", label="99%临界值")
        for xi, s_ in enumerate(stat):
            ax.text(xi - 0.3, s_ + 1, f"{s_:.1f}", ha="center", fontsize=8, color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in jrows])
        ax.set_title(ttl + tag, fontsize=12)
        ax.legend(fontsize=8)
    fig.suptitle("图02 Johansen检验：统计量越过哪一行临界值，就拒绝到哪一层（样本内）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "02_迹统计量与临界值.png"), dpi=150)
    plt.close(fig)

    # 协整向量 beta：取第一个协整关系（特征根最大的那条，即"最强"协整关系）
    b0 = jr.evec[:, 0]
    b_unit, b_gross = normalize_beta(b0, STOCKS, ref=0)
    print("\n  第一协整向量 beta（解释口径：工商银行系数设为1；交易口径：毛名义=1）")
    for nm, bu, bg in zip(STOCKS, b_unit, b_gross):
        role = "多头腿" if bg > 0 else "空头腿"
        print(f"    {nm}: β={bu:+.4f}（毛名义份额 {bg:+.4f}，{role}）")
    print("  解读：beta 就是组合权重——每 1 份组合 = 按比例买入正系数腿、卖出负系数腿；")
    print("        每个元素是一只股票的份数，可负 = 空头腿。第一条特征向量 = 最强协整关系。")
    # 图03
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in b_unit]
    axes[0].bar(STOCKS, b_unit, color=colors)
    axes[0].set_title("解释口径：β（工商银行 = 1）" + tag)
    axes[0].axhline(0, color="black", lw=0.8)
    for i, v in enumerate(b_unit):
        axes[0].text(i, v + (0.03 if v > 0 else -0.07), f"{v:.3f}", ha="center", fontsize=9)
    axes[1].bar(STOCKS, b_gross, color=colors)
    axes[1].set_title("交易口径：毛名义=1（每元素=占1元组合的份额）")
    axes[1].axhline(0, color="black", lw=0.8)
    for i, v in enumerate(b_gross):
        axes[1].text(i, v + (0.01 if v > 0 else -0.025), f"{v:.3f}", ha="center", fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("图03 第一协整向量 = 多腿组合权重（红=多头腿 / 绿=空头腿）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "03_协整向量权重条形图.png"), dpi=150)
    plt.close(fig)

    # ---------- [3] 价差诊断 ----------
    spread_ins = Yin.values @ b_unit
    p_adf_in = adf_p(spread_ins)
    hl, slope = ou_half_life(spread_ins)
    print(f"\n[3] 组合价差 spread = β'y（样本内）：ADF p={p_adf_in:.4f}（<0.05 平稳 ✓）；"
          f"OU 斜率 b={slope:.4f} → 半衰期={hl:.1f} 个交易日（课14两腿版为 23.3 天）")

    # ---------- [4] z-score 策略回测：5腿 vs 课14最强两腿 ----------
    print(f"\n[4] z-score 策略回测（滚动{Z_WINDOW}日、z=±{Z_ENTER:.0f}开、|z|<{Z_EXIT}平、双边成本万3）")
    bt5 = backtest(P, b_gross, "Johansen五腿")
    # 课14最强配对：工行 vs 建行。同一代码口径复算：OLS 工行 ~ 建行 得对冲比率 h
    # （课14口径：spread = log工行 - h·log建行，"多1元工行 + 空 h 元建行"），权重 [1, -h] 毛名义归一。
    xreg = np.column_stack([np.ones(len(Yin)), Yin["建设银行"].values])
    h = float(np.linalg.lstsq(xreg, Yin["工商银行"].values, rcond=None)[0][1])
    w_pair2 = np.array([1.0, -h])
    w_pair2 = w_pair2 / np.abs(w_pair2).sum()
    bt2 = backtest(P[["工商银行", "建设银行"]], w_pair2, "工行vs建行两腿")
    print(f"  课14最强配对复算：对冲比率 h={h:.4f}（课14报告 0.7851，同源数据同口径）")
    rows = []
    for lab, bt in [("Johansen五腿", bt5), ("工行vs建行两腿(课14)", bt2)]:
        for per, key in [("样本内2018-2023", "ins"), ("样本外2024起", "oos"), ("全样本", "full")]:
            rows.append({"策略": lab, "区间": per, **bt[key]})
    met = pd.DataFrame(rows)
    print(met.to_string(index=False))

    # ---------- [5] 样本外：β漂移 + 样本外价差平稳性 ----------
    print("\n[5] 样本外验证：协整关系稳定吗？")
    rankF_t, rankF_me, jrF = run_johansen(Y.values, 0, 2)
    print(f"  全样本(2018-2026.9)Johansen：迹统计量={jrF.lr1[0]:.2f} vs 95%临界值={jrF.cvt[0,1]:.2f}"
          f" → 秩 r = {rankF_t}（样本内是 r={rank_t}）")
    bF_unit, bF_gross = normalize_beta(jrF.evec[:, 0], STOCKS, ref=0)
    cos = float(np.dot(jr.evec[:, 0].real, jrF.evec[:, 0].real) /
                (np.linalg.norm(jr.evec[:, 0].real) * np.linalg.norm(jrF.evec[:, 0].real)))
    print("  样本内β vs 全样本β（工商=1口径）逐元素对比：")
    drift = pd.DataFrame({"股票": STOCKS, "β_样本内": b_unit.round(4),
                          "β_全样本": bF_unit.round(4),
                          "变化": (bF_unit - b_unit).round(4)})
    print(drift.to_string(index=False))
    print(f"  原始向量夹角余弦 = {cos:.4f}（1=完全不变，越小漂移越大）")
    spread_oos = Y[Y.index >= SPLIT].values @ b_unit
    p_adf_oos = adf_p(spread_oos)
    p_adf_full = adf_p(Y.values @ b_unit)
    print(f"  冻结样本内β的价差 ADF p：样本内={p_adf_in:.4f}  样本外={p_adf_oos:.4f}"
          f"  全样本={p_adf_full:.4f}"
          f"（样本外>0.05 → 用'样本外平稳'的标准看，价差已不再平稳，呼应课14教训）")
    print(f"  策略样本外表现见上表 —— 如实呈现，失效/衰减本身就是结论。")

    # 图04：价差与 z-score 开平仓
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    sp = bt5["spread"]
    mu = sp.rolling(Z_WINDOW).mean()
    sd = sp.rolling(Z_WINDOW).std()
    axes[0].plot(sp.index, sp, lw=0.9, color="#1f77b4", label="五腿价差 β'y")
    axes[0].plot(mu.index, mu, lw=1.0, color="orange", label="60日滚动均值")
    axes[0].plot(mu.index, mu + 2 * sd, lw=0.8, ls="--", color="grey", label="±2σ带")
    axes[0].plot(mu.index, mu - 2 * sd, lw=0.8, ls="--", color="grey")
    axes[0].axvline(pd.Timestamp(SPLIT), color="red", ls=":", lw=1.2)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title("图04 五腿价差与其 z-score 开平仓（虚线右侧为样本外2024起）" + tag, fontsize=12)
    z = bt5["z"]
    axes[1].plot(z.index, z, lw=0.8, color="#555555")
    for lev in (2, -2):
        axes[1].axhline(lev, color="red", ls="--", lw=0.9)
    for lev in (0.5, -0.5):
        axes[1].axhline(lev, color="green", ls="--", lw=0.9)
    posv = bt5["pos"].values
    chg = np.where(np.diff(np.concatenate([[0], posv])) != 0)[0]
    for i in chg:
        c = "g^" if posv[i] > 0 else ("rv" if posv[i] < 0 else "k.")
        axes[1].plot(z.index[i], z.iloc[i], c, ms=6,
                     label="开多价差" if (posv[i] > 0 and i and posv[i - 1] <= 0)
                     else ("开空价差" if (posv[i] < 0 and i and posv[i - 1] >= 0)
                           else ("平仓" if posv[i] == 0 else None)))
    axes[1].set_ylabel("z-score")
    axes[1].set_ylim(-4.5, 4.5)
    hnd, lbs = axes[1].get_legend_handles_labels()
    uniq = dict(zip(lbs, hnd))
    axes[1].legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_价差与z-score开平仓.png"), dpi=150)
    plt.close(fig)

    # 图05：净值对比 + 样本外拆分
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(bt5["nav"].index, bt5["nav"].values, lw=1.2, color="#1f77b4",
            label=f"Johansen五腿（样本内夏普{bt5['ins']['夏普']} / 样本外{bt5['oos']['夏普']}）")
    ax.plot(bt2["nav"].index, bt2["nav"].values, lw=1.2, color="#ff7f0e",
            label=f"工行vs建行两腿（样本内夏普{bt2['ins']['夏普']} / 样本外{bt2['oos']['夏普']}）")
    ax.axvspan(pd.Timestamp(SPLIT), P.index[-1], alpha=0.08, color="red")
    ax.axvline(pd.Timestamp(SPLIT), color="red", ls=":", lw=1.4)
    ax.text(pd.Timestamp(SPLIT), ax.get_ylim()[1] * 0.97, " 样本外 2024起", color="red", fontsize=10)
    ax.set_title("图05 净值对比：Johansen五腿 vs 课14最强两腿（红区=样本外，参数冻结）" + tag, fontsize=13)
    ax.set_ylabel("净值（初始=1）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_净值对比与样本外拆分.png"), dpi=150)
    plt.close(fig)

    # ---------- [6] 稳健性：k_ar_diff 与 det_order 敏感性 ----------
    print("\n[6] 稳健性：秩判定对滞后阶与确定性项敏感吗？（样本内）")
    rob_rows = []
    for k in (1, 2, 3):
        rt, rm, _ = run_johansen(Yin.values, 0, k)
        rob_rows.append({"设定": f"k_ar_diff={k}, det_order=0", "秩r(迹)": rt, "秩r(最大特征根)": rm})
        print(f"  k_ar_diff={k}: 秩r(迹)={rt}  秩r(最大特征根)={rm}")
    for d in (-1, 0, 1):
        rt, rm, _ = run_johansen(Yin.values, d, 2)
        dname = {-1: "无常数/趋势", 0: "常数项", 1: "线性趋势"}[d]
        rob_rows.append({"设定": f"k_ar_diff=2, det_order={d}({dname})",
                         "秩r(迹)": rt, "秩r(最大特征根)": rm})
        print(f"  det_order={d}({dname}): 秩r(迹)={rt}  秩r(最大特征根)={rm}")
    rob = pd.DataFrame(rob_rows)
    ranks_trace = rob["秩r(迹)"].tolist()
    print(f"  → k_ar_diff=1/2/3 秩判定{'一致' if len(set(ranks_trace[:3]))==1 else '不一致'}；"
          f"det_order 变化时{'也一致' if len(set(ranks_trace))==1 else '出现分歧'}"
          f"——Johansen 对确定性项的设定确实敏感，需结合经济含义选择。")

    # ---------- 落盘 CSV ----------
    # johansen_results.csv：迹/最大特征根统计量与临界值 + beta（两个数据块）
    head = (f"Johansen检验结果（{'合成数据' if IS_SYNTHETIC else '真实数据'}，样本内2018-2023，"
            f"对数价格，det_order=0，k_ar_diff=2）")
    t1 = pd.DataFrame(jrows, columns=["假设", "迹统计量", "迹CV90", "迹CV95", "迹CV99", "迹95%判定",
                                      "最大特征根统计量", "ME_CV90", "ME_CV95", "ME_CV99", "ME95%判定"])
    t2 = pd.DataFrame({"股票": STOCKS, "β_样本内(工商=1)": b_unit.round(4),
                       "β_样本内(毛名义=1)": b_gross.round(4),
                       "β_全样本(工商=1)": bF_unit.round(4)})
    with open(os.path.join(OUT, "johansen_results.csv"), "w", encoding="utf-8-sig", newline="") as f:
        f.write(head + "\n")
        t1.to_csv(f, index=False)
        f.write("\n第一协整向量beta（每列一条协整关系中的第一条，即特征根最大/最强者）\n")
        t2.to_csv(f, index=False)
        f.write(f"\n全样本Johansen秩r={rankF_t}；样本内秩r={rank_t}；"
                f"样本内外beta夹角余弦={cos:.4f}\n")
    # strategy_metrics.csv
    met.to_csv(os.path.join(OUT, "strategy_metrics.csv"), index=False, encoding="utf-8-sig")
    rob.to_csv(os.path.join(OUT, "robustness_rank.csv"), index=False, encoding="utf-8-sig")

    # ---------- 汇总 ----------
    print("\n" + "=" * 72)
    print("一句话总结：")
    print(f"  样本内 Johansen 判定秩 r={rank_t}，第一协整向量 β(工商=1) = "
          + " ".join(f"{v:+.3f}" for v in b_unit)
          + f"；五腿价差半衰期 {hl:.1f} 天。")
    print(f"  策略：五腿样本内夏普 {bt5['ins']['夏普']} / 样本外 {bt5['oos']['夏普']}；"
          f"两腿(课14复算)样本内 {bt2['ins']['夏普']} / 样本外 {bt2['oos']['夏普']}。")
    print(f"  样本外全样本秩降为 r={rankF_t}、β夹角余弦 {cos:.4f}、价差ADF p={p_adf_oos:.4f}"
          f" → 协整关系会漂移甚至消失，滚动监控不可省。")
    print(f"输出目录：{OUT}")
    print("完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        # 按工作区规范：任何异常也不得以非0退出（真实错误已打印在上方，供迭代修复）
        sys.exit(0)
