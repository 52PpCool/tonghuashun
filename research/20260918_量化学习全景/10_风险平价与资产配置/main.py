# -*- coding: utf-8 -*-
"""
10_风险平价与资产配置
======================
"不要把鸡蛋放在一个篮子里"的工程化实现。

本脚本用 10 个申万/同花顺行业板块指数的真实日线数据，构建并回测四种
月度再平衡的资产配置策略，与沪深300买入持有作对比：

  1. 等权组合       —— 每月把 10 个行业各配 10%，最朴素的"分散"。
  2. 最小方差组合   —— 滚动 24 个月窗口估计协方差，最小化组合波动。
  3. 风险平价组合   —— 迭代求解"每个资产的风险贡献相等"的组合。
  4. 波动率目标组合 —— 等权基础上加现金腿，把组合波动压到年化 10%。

教学主线：等权 ≠ 等风险。谁波动大、谁与其他资产相关性强，谁就在等权
组合里贡献了不成比例的风险。风险平价不预测收益，只"摆平"风险贡献。

运行：.venv/bin/python main.py   （在工作目录 10_风险平价与资产配置/ 下）
输出：outputs/ 下 5 张图 + portfolios.csv；控制台打印全部关键数字。
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")  # 无界面环境出图
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 中文字体与负号显示（必须在绘图前设置）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
SEED = 42                       # 随机种子（合成数据兜底时使用）
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "cache")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

N_COV = 24                      # 协方差滚动估计窗口（月）
RF_ANNUAL = 0.02                # 无风险利率（年化），用于夏普比率与现金腿
TARGET_VOL = 0.10               # 波动率目标的年化目标波动
DESIGNED_START = "2018-01-01"   # 课程设计上希望 2018 年起回测（受数据可得性约束，见 README）
SPLICE_THRESH = 0.20            # 日收益超过 ±20% 判定为"基期跳变"毛刺（真实板块指数不可能）
COST_PER_SIDE = 0.001           # 单边交易成本假设 0.1%，仅用于估算成本拖累（回测本身未扣费）


# ----------------------------------------------------------------------
# [1] 数据加载与清洗
# ----------------------------------------------------------------------
def splice_scale_jumps(close: pd.Series, thresh: float = SPLICE_THRESH) -> pd.Series:
    """清洗板块指数中的"基期跳变"毛刺。

    原始数据里，部分行业指数会在个别交易日突然跳到另一个尺度
    （例如单日 -64%，次日 +170% 回到原位）。真实行业指数不可能单日
    涨跌超过 ±20%，这些跳变是数据源的基期重置伪影，不是市场行为。

    处理办法（指数"复权"式拼接）：一旦某天的收益率绝对值超过阈值，
    就把该天之前的全部历史序列乘以跳变比例，使该天收益归零——
    等价于把新旧两段指数"拼接"成一条连续曲线。

    副作用：跳变当天的真实涨跌被抹为 0（本来也不可考），其余收益不变。
    """
    v = close.to_numpy(dtype=float).copy()
    n_splice = 0
    for t in range(1, len(v)):
        r = v[t] / v[t - 1] - 1.0
        if abs(r) > thresh:            # 判定为尺度跳变
            v[:t] *= v[t] / v[t - 1]   # 把历史拉到新尺度，收益归零
            n_splice += 1
    return pd.Series(v, index=close.index), n_splice


def load_boards() -> tuple[pd.DataFrame, dict]:
    """读取 10 个行业板块 CSV，清洗后取收盘价，返回(价格面板, 清洗报告)。"""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "board_*.csv")))
    closes, report = {}, {}
    for f in files:
        name = os.path.basename(f)[6:-4]          # board_白酒.csv -> 白酒
        df = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
        df = df.dropna(subset=["close"]).sort_index()  # 末行可能只有成交量、无收盘价
        c, n_splice = splice_scale_jumps(df["close"])
        closes[name] = c
        report[name] = {"start": df.index.min(), "n": len(df), "spliced": n_splice}
    px = pd.DataFrame(closes).dropna()            # dropna=取10个行业的共同重叠交易日
    return px, report


def to_monthly_returns(px: pd.DataFrame) -> pd.DataFrame:
    """日线收盘 -> 月末收盘 -> 月频简单收益。剔除不完整的最后一个月。"""
    m = px.resample("ME").last()
    if (m.index[-1] - px.index.max()).days > 7:   # 数据没到月末 -> 该月不完整
        m = m.iloc[:-1]
    return m.pct_change().dropna()


def load_benchmark() -> pd.Series:
    """沪深300 收盘价（指数文件干净，无需拼接清洗）。"""
    f = os.path.join(DATA_DIR, "index_000300_沪深300.csv")
    df = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
    return df["close"].dropna().sort_index()


def make_synthetic() -> pd.DataFrame:
    """兜底：若真实数据缺失，用 rng(42) 造带因子结构的合成月收益（标注为合成数据）。"""
    print("!! 未找到有效真实数据，改用合成数据（rng=42），结果仅作演示 !!")
    n_month, n_asset = 56, 10
    factor = RNG.normal(0, 0.04, (n_month, 1))              # 共同市场因子
    beta = RNG.uniform(0.6, 1.4, (1, n_asset))
    idio = RNG.normal(0, 0.05, (n_month, n_asset))          # 特质波动
    idx = pd.date_range("2022-01-31", periods=n_month, freq="ME")
    return pd.DataFrame(factor * beta + idio, index=idx,
                        columns=[f"行业{i+1}" for i in range(n_asset)])


# ----------------------------------------------------------------------
# [2] 风险贡献：本课的核心公式
# ----------------------------------------------------------------------
def risk_contributions(w: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, float]:
    """组合风险分解。

    组合波动  sigma_p = sqrt(w' Σ w)
    第 i 个资产的风险贡献（以波动率单位计）：
        RC_i = w_i * (Σ w)_i / sqrt(w' Σ w)
    满足极佳的加总性质：Σ_i RC_i = sigma_p。
    即"组合的总波动可以不多不少地拆成 N 份，每份归因到一个资产"。

    返回 (rc 向量, 组合年化前的月频波动 sigma_p)。
    """
    x = S @ w
    sigma = float(np.sqrt(w @ x))
    rc = w * x / sigma
    return rc, sigma


# ----------------------------------------------------------------------
# [3] 三种配置权重的求解器（只用协方差，不用任何收益预测）
# ----------------------------------------------------------------------
def ew_weights(n: int) -> np.ndarray:
    """等权：每个资产 1/N。"""
    return np.ones(n) / n


def min_var_weights(S: np.ndarray) -> np.ndarray:
    """最小方差组合：min w'Σw，s.t. Σw=1, w>=0（凸优化，SLSQP 求解）。

    注意它只关心"波动"，完全不看预期收益——低波资产（如银行、电力）
    往往被配到很高的权重，甚至出现 0 权重的角点解。
    """
    n = S.shape[0]
    res = minimize(lambda w: w @ S @ w, np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 1000, "ftol": 1e-14})
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def risk_parity_weights(S: np.ndarray) -> np.ndarray:
    """风险平价组合：让每个资产的风险贡献占比都等于 1/N。

    求解目标（scipy.optimize 最小化）：
        min_w  Σ_i ( RC_i/sigma_p - 1/N )^2
        s.t.   Σw = 1, w >= 0
    其中 RC_i/sigma_p = w_i(Σw)_i / (w'Σw) 是归一化风险贡献，理想解下
    每一项都恰好等于 1/N，目标函数值降为 0。这是非线性方程组，没有
    闭式解，只能迭代求解；初值用"波动率倒数"加权（很接近风险平价）。
    """
    n = S.shape[0]

    def objective(w):
        x = S @ w
        sig2 = w @ x                       # w'Σw
        rc_norm = w * x / sig2             # 归一化风险贡献，和为 1
        return np.sum((rc_norm - 1.0 / n) ** 2) * 1e4   # 放大 1e4 便于求解器收敛

    # 初值：波动率倒数加权（对角协方差下的近似风险平价）
    iv = 1.0 / np.sqrt(np.diag(S))
    w0 = iv / iv.sum()
    res = minimize(objective, w0, method="SLSQP",
                   bounds=[(1e-6, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 1000, "ftol": 1e-18})
    w = res.x if res.success else w0       # 兜底：求解失败退回波动率倒数
    w = np.clip(w, 1e-6, None)
    w /= w.sum()
    return w


# ----------------------------------------------------------------------
# [4] 月度再平衡回测引擎
# ----------------------------------------------------------------------
def backtest(monthly_ret: pd.DataFrame, bench_ret: pd.Series) -> dict:
    """四种策略统一走同一条回测流水线，保证可比。

    时序约定（避免未来函数）：
      在每月末 t，用截至 t 的最近 24 个月收益估计协方差 Σ_t，
      求出目标权重；该权重"持有"下一个月，赚到 t+1 月的收益。
      月内权重随价格漂移，月末再平衡。

    波动率目标策略的账本里有第 N+1 个"资产"——现金腿：
      杠杆 L = 目标波动 / 等权组合已实现波动（滚动24个月，年化）
      风险资产权重 = L × 等权，现金权重 = 1 - L。
      L>1 表示借钱加杠杆（付融资利率 rf）；L<1 表示部分资金吃利息。
      本数据窗口内已实现波动 16%~25% 始终高于 10%，L 始终小于 1。
    """
    names = list(monthly_ret.columns)
    n = len(names)
    ret = monthly_ret.to_numpy()
    rf_m = RF_ANNUAL / 12.0                     # 月度无风险利率

    def solve_targets(window: np.ndarray):
        """给定最近 N_COV 个月收益窗口，求解四策略的目标权重（含现金腿）。"""
        S = np.cov(window, rowvar=False, ddof=1)                 # 样本协方差（月频）
        w_ew = ew_weights(n)
        w_mv = min_var_weights(S)
        w_rp = risk_parity_weights(S)
        ew_vol_ann = float(np.std(window @ w_ew, ddof=1) * np.sqrt(12))  # 等权组合已实现波动
        leverage = TARGET_VOL / ew_vol_ann                                # 波动率目标杠杆
        targets = {"等权": np.append(w_ew, 0.0),                   # 账本 = [10个行业..., 现金]
                   "最小方差": np.append(w_mv, 0.0),
                   "风险平价": np.append(w_rp, 0.0),
                   "波动率目标": np.append(leverage * w_ew, 1.0 - leverage)}
        return S, targets, leverage

    # 估计时点：月末 t 需要至少 24 个月历史 -> 从第 24 个月末（0-based 23）开始
    est_points = list(range(N_COV - 1, len(ret)))

    strategies = ["等权", "最小方差", "风险平价", "波动率目标"]
    state = {s: None for s in strategies}       # 各策略当前(漂移后)权重 [风险资产..., 现金]
    nav = {s: 1.0 for s in strategies}
    nav_series = {s: [] for s in strategies}
    turn_over = {s: [] for s in strategies}     # 每次再平衡的单边换手
    turn_dates = []                             # 对应的再平衡日期
    weight_hist = {s: [] for s in strategies}   # 每次再平衡的目标权重（画堆叠图用）
    weight_dates = []
    lev_hist = []

    for k in range(N_COV, len(ret)):            # 策略月：k=24..55（2024-01 ~ 2026-08）
        est = k - 1                             # 估计窗口截止的月份（0-based）
        S, targets, leverage = solve_targets(ret[est - N_COV + 1: est + 1])  # 最近24个月
        lev_hist.append((monthly_ret.index[est], leverage))

        if k > N_COV:                           # 建仓月（k=24）不计换手
            turn_dates.append(monthly_ret.index[est])
        for s in strategies:
            tgt = targets[s]
            if state[s] is not None:            # 单边换手 = 0.5 × Σ|目标权重 - 漂移后权重|
                turn_over[s].append(0.5 * float(np.abs(tgt - state[s]).sum()))
            state[s] = tgt.copy()
            weight_hist[s].append(tgt.copy())
        weight_dates.append(monthly_ret.index[est])

        # —— 用目标权重持有下一个月 ——
        r_vec = ret[k]
        for s in strategies:
            w = state[s]
            r_p = float(w[:n] @ r_vec + w[n] * rf_m)      # 风险资产收益 + 现金利息
            nav[s] *= (1.0 + r_p)
            nav_series[s].append((monthly_ret.index[k], nav[s], r_p))
            # 权重漂移：涨得多的资产占比自然变大，直到下月末再平衡
            state[s] = np.append(w[:n] * (1.0 + r_vec) / (1.0 + r_p),
                                 w[n] * (1.0 + rf_m) / (1.0 + r_p))

    # 最后一个估计时点（2026-08-31）：只求解不交易，供风险贡献图/权重图使用
    S_last, targets_last, lev_last = solve_targets(ret[-N_COV:])
    lev_hist.append((monthly_ret.index[-1], lev_last))
    for s in strategies:
        tgt = targets_last[s]
        turn_over[s].append(0.5 * float(np.abs(tgt - state[s]).sum()))
        weight_hist[s].append(tgt.copy())
    turn_dates.append(monthly_ret.index[-1])
    weight_dates.append(monthly_ret.index[-1])
    ew_est = targets_last["等权"][:n]
    mv_est = targets_last["最小方差"][:n]
    rp_est = targets_last["风险平价"][:n]

    out = {"names": names, "nav": {}, "ret": {}, "turnover": {}, "weights": {},
           "ew_est": ew_est, "mv_est": mv_est, "rp_est": rp_est,
           "lev_hist": lev_hist, "S_last": S_last, "ew_est_date": monthly_ret.index[-1]}
    for s in strategies:
        df = pd.DataFrame(nav_series[s], columns=["date", "nav", "ret"]).set_index("date")
        out["nav"][s] = df["nav"]
        out["ret"][s] = df["ret"]
        out["turnover"][s] = pd.Series(turn_over[s], index=turn_dates)
        out["weights"][s] = pd.DataFrame(weight_hist[s], index=weight_dates,
                                         columns=names + ["现金"])
    return out


def perf_stats(ret_m: pd.Series, turnover_ann: float) -> dict:
    """单一策略的绩效指标（月频收益口径）。"""
    n = len(ret_m)
    total = float((1.0 + ret_m).prod())
    ann = total ** (12.0 / n) - 1.0                      # 几何年化
    vol = float(ret_m.std(ddof=1) * np.sqrt(12))         # 年化波动
    sharpe = (ann - RF_ANNUAL) / vol if vol > 0 else np.nan
    nav = (1.0 + ret_m).cumprod()
    dd = nav / nav.cummax() - 1.0
    return {"年化收益率(%)": ann * 100, "年化波动率(%)": vol * 100,
            "夏普比率(rf=2%)": sharpe, "最大回撤(%)": float(dd.min()) * 100,
            "月度胜率(%)": float((ret_m > 0).mean()) * 100,
            "年化换手率(%)": turnover_ann * 100}


# ----------------------------------------------------------------------
# [5] 主流程
# ----------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("10_风险平价与资产配置：等权 / 最小方差 / 风险平价 / 波动率目标 对比回测")
    print("=" * 78)

    # ---- 5.1 数据 ----
    px, report = load_boards()
    if len(px) < 400:                        # 真实数据不足 -> 合成数据兜底
        monthly = make_synthetic()
        is_synth = True
    else:
        monthly = to_monthly_returns(px)
        is_synth = False
    print(f"\n[数据] {'合成数据' if is_synth else '真实数据'}  10个行业共同重叠期: "
          f"{px.index.min().date()} ~ {px.index.max().date()}（{len(px)} 个交易日）")
    print("[数据] 各行业起始日与基期跳变清洗次数（>±20%的单日跳变视为数据源基期重置）:")
    for name, info in report.items():
        print(f"    {name:6s} 起始 {info['start'].date()}  样本{info['n']:5d}天  清洗跳变 {info['spliced']} 次")
    print(f"[说明] 课程设计希望 {DESIGNED_START[:4]} 年起回测，但 10 个行业的共同重叠期从 "
          f"{px.index.min().date()} 开始（电力行业指数发布最晚），\n"
          f"       且滚动 {N_COV} 个月协方差需要 {N_COV} 个月预热，故四策略统一从 "
          f"{monthly.index[N_COV].date()} 起评估。")

    bench = load_benchmark()
    bench_m = to_monthly_returns(bench.to_frame("close"))

    # ---- 5.2 回测 ----
    bt = backtest(monthly, bench_m)
    names = bt["names"]
    strat_months = bt["ret"]["等权"].index
    bench_strat = bench_m.reindex(strat_months)["close"]
    print(f"\n[回测] 统一评估窗口: {strat_months.min().date()} ~ {strat_months.max().date()} "
          f"（{len(strat_months)} 个月）")

    # ---- 5.3 风险贡献直觉：等权 != 等风险 ----
    S_last = bt["S_last"]
    est_date = bt["ew_est_date"]
    rc_ew, sig_ew = risk_contributions(bt["ew_est"], S_last)
    rc_rp, sig_rp = risk_contributions(bt["rp_est"], S_last)
    rc_ew_share, rc_rp_share = rc_ew / sig_ew, rc_rp / sig_rp   # 归一化：占比，和为1
    print(f"\n[风险贡献] 估计时点 {est_date.date()}（滚动24个月协方差），占比口径（占组合总波动的比例）")
    hdr = f"    {'行业':6s} {'年化波动':>8s} {'等权-权重':>9s} {'等权-风险贡献':>13s} {'平价-权重':>9s} {'平价-风险贡献':>13s}"
    print(hdr)
    vol_ann = np.sqrt(np.diag(S_last) * 12)
    order = np.argsort(-vol_ann)             # 按波动从大到小排
    for i in order:
        print(f"    {names[i]:6s} {vol_ann[i]*100:7.1f}% {bt['ew_est'][i]*100:8.1f}% "
              f"{rc_ew_share[i]:12.1%} {bt['rp_est'][i]*100:8.1f}% {rc_rp_share[i]:12.1%}")
    i_max, i_min = int(np.argmax(rc_ew_share)), int(np.argmin(rc_ew_share))
    print(f"    等权组合月波动 {sig_ew:.2%}（年化 {sig_ew*np.sqrt(12):.1%}）：每个行业权重都是10%，"
          f"但风险贡献从 {names[i_min]} 的 {rc_ew_share[i_min]:.1%}"
          f" 到 {names[i_max]} 的 {rc_ew_share[i_max]:.1%}——谁（与其他资产）共同波动大，谁主导风险。")
    dev = float(np.abs(rc_rp_share - 1.0 / len(names)).max())
    print(f"    风险平价求解质量：最大 |风险贡献占比 - 1/N| = {dev:.2e}（≈0 表示每个资产风险贡献确实相等）")

    # ---- 5.4 绩效表 ----
    rows = {}
    for s in ["等权", "最小方差", "风险平价", "波动率目标"]:
        t_ann = float(bt["turnover"][s].mean() * 12)     # 月均单边换手 -> 年化
        rows[s] = perf_stats(bt["ret"][s], t_ann)
    rows["沪深300(基准)"] = perf_stats(bench_strat, 0.0)  # 买入持有：零换手
    perf = pd.DataFrame(rows).T
    print("\n[绩效表] （月频口径；换手为单边、未含成本）")
    print(perf.round(2).to_string())
    perf.to_csv(os.path.join(OUT_DIR, "portfolios.csv"), encoding="utf-8-sig")

    print("\n[成本提示] 若单边成本 0.1%，各策略年化成本拖累约为：")
    for s in ["等权", "最小方差", "风险平价", "波动率目标"]:
        t_ann = float(bt["turnover"][s].mean() * 12)
        print(f"    {s}: 年化换手 {t_ann*100:.1f}% × 0.1% ≈ {t_ann*COST_PER_SIDE*100:.2f}%/年")

    # ---- 5.5 相关性：分散化的天花板 ----
    corr = monthly.corr()
    iu = np.triu_indices(len(names), 1)
    pairs = [(names[i], names[j], corr.values[i, j]) for i, j in zip(*iu)]
    pairs.sort(key=lambda x: x[2])
    print(f"\n[相关性] 10行业月收益两两相关系数：均值 {np.mean([p[2] for p in pairs]):.2f}，"
          f"最低 {pairs[0][0]}-{pairs[0][1]} {pairs[0][2]:.2f}，最高 {pairs[-1][0]}-{pairs[-1][1]} {pairs[-1][2]:.2f}")

    # ---- 5.6 年度收益（如实呈现跑输的年份） ----
    yearly = {}
    for s in ["等权", "最小方差", "风险平价", "波动率目标"]:
        yearly[s] = bt["ret"][s].groupby(lambda d: d.year).apply(lambda x: (1 + x).prod() - 1)
    yearly["沪深300(基准)"] = bench_strat.groupby(lambda d: d.year).apply(lambda x: (1 + x).prod() - 1)
    yearly = pd.DataFrame(yearly)
    print("\n[年度收益] （2026 为 1-8 月）")
    print((yearly * 100).round(2).to_string())

    lev = pd.Series({d: l for d, l in bt["lev_hist"]})
    print(f"\n[波动率目标] 杠杆 L=10%/已实现波动 的区间: {lev.min():.2f} ~ {lev.max():.2f}"
          f"（始终<1：无需借钱，现金腿规模 {((1-lev).min()*100):.0f}%~{((1-lev).max()*100):.0f}%，按 2% 吃利息）")

    # ---- 5.7 绘图 ----
    colors = plt.get_cmap("tab10").colors
    x_ord = [names[i] for i in np.argsort(-vol_ann)]     # 按波动从大到小

    # 图1：等权组合的风险贡献
    fig, ax = plt.subplots(figsize=(10, 5.2))
    xx = np.arange(len(names))
    idx_ord = [names.index(nm) for nm in x_ord]
    ax.bar(xx - 0.2, bt["ew_est"][idx_ord] * 100, 0.4, label="权重占比", color="#7f9fc4")
    ax.bar(xx + 0.2, rc_ew_share[idx_ord] * 100, 0.4, label="风险贡献占比", color="#c46a6a")
    ax.axhline(10, color="gray", ls="--", lw=1, label="等风险基准 10%")
    ax.set_xticks(xx, x_ord, rotation=30, ha="right")
    ax.set_ylabel("占组合总波动的比例 (%)")
    ax.set_title(f"图1 等权组合的风险贡献（时点 {est_date.date()}，滚动24个月协方差）\n"
                 "权重都是10%，但风险贡献严重不均——谁与组合共同波动大，谁主导风险")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "01_等权组合的风险贡献.png"), dpi=150); plt.close(fig)

    # 图2：风险平价组合的风险贡献
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(xx - 0.2, bt["rp_est"][idx_ord] * 100, 0.4, label="权重占比", color="#7f9fc4")
    ax.bar(xx + 0.2, rc_rp_share[idx_ord] * 100, 0.4, label="风险贡献占比", color="#c46a6a")
    ax.axhline(10, color="gray", ls="--", lw=1, label="等风险基准 10%")
    ax.set_xticks(xx, x_ord, rotation=30, ha="right")
    ax.set_ylabel("占组合总波动的比例 (%)")
    ax.set_title(f"图2 风险平价组合的风险贡献（同一时点 {est_date.date()}）\n"
                 "低波资产（银行/电力）拿更高权重，十个行业的风险贡献被摆平到各 10%")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "02_风险平价组合的风险贡献.png"), dpi=150); plt.close(fig)

    # 图3：净值对比
    fig, ax = plt.subplots(figsize=(10, 5.6))
    style = {"等权": ("#4c78a8", "-"), "最小方差": ("#f58518", "-"),
             "风险平价": ("#54a24b", "-"), "波动率目标": ("#b279a2", "-"),
             "沪深300(基准)": ("#555555", "--")}
    for s in ["等权", "最小方差", "风险平价", "波动率目标"]:
        ax.plot(bt["nav"][s].index, bt["nav"][s].values, color=style[s][0], ls=style[s][1],
                label=f"{s}（{bt['nav'][s].iloc[-1]:.3f}）")
    bnav = (1 + bench_strat).cumprod()
    ax.plot(bnav.index, bnav.values, color=style["沪深300(基准)"][0], ls=style["沪深300(基准)"][1],
            label=f"沪深300买入持有（{bnav.iloc[-1]:.3f}）")
    ax.set_title(f"图3 四策略净值对比（{strat_months.min().date()} ~ {strat_months.max().date()}，月末，未扣交易成本）")
    ax.set_ylabel("净值（起点=1）"); ax.grid(alpha=0.3); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "03_四策略净值对比.png"), dpi=150); plt.close(fig)

    # 图4：回撤对比
    fig, ax = plt.subplots(figsize=(10, 5.0))
    for s in ["等权", "最小方差", "风险平价", "波动率目标"]:
        nav = bt["nav"][s]
        dd = nav / nav.cummax() - 1
        ax.plot(dd.index, dd.values * 100, color=style[s][0], label=s)
    bdd = bnav / bnav.cummax() - 1
    ax.plot(bdd.index, bdd.values * 100, color=style["沪深300(基准)"][0], ls="--", label="沪深300(基准)")
    ax.set_title("图4 回撤对比（月末净值口径）")
    ax.set_ylabel("回撤 (%)"); ax.grid(alpha=0.3); ax.legend(ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "04_回撤对比.png"), dpi=150); plt.close(fig)

    # 图5：权重演变（最小方差 vs 风险平价）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    for ax, s in zip(axes, ["最小方差", "风险平价"]):
        wh = bt["weights"][s][names]                     # 不画现金腿（前三者现金为0）
        ax.stackplot(wh.index, *[wh[c].values * 100 for c in names],
                     labels=names, colors=colors[:len(names)], alpha=0.9)
        ax.set_title(f"{s}组合权重演变（月度再平衡）")
        ax.set_ylim(0, 100); ax.set_ylabel("权重 (%)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.margins(x=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10, fontsize=8, frameon=False)
    fig.suptitle("图5 权重演变堆叠图：最小方差偏爱低波资产（角点解），风险平价权重更均匀", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(os.path.join(OUT_DIR, "05_权重演变堆叠图.png"), dpi=150); plt.close(fig)

    print(f"\n[输出] 图与绩效表已写入: {OUT_DIR}")
    print("完成，退出码 0。")


if __name__ == "__main__":
    main()
