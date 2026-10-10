# -*- coding: utf-8 -*-
"""
09 马科维茨有效前沿（Modern Portfolio Theory, Markowitz 1952）
=============================================================
一节课讲清楚三件事：
  1) 均值-方差框架怎么把"配钱"变成一个数学优化问题（GMV / 目标收益最小方差 / 最大夏普）；
  2) 有效前沿长什么样，随机组合为什么全在前沿内侧（分散化的几何直觉）；
  3) 【本课灵魂】均值/协方差的"估计误差"如何让样本内最优组合在样本外现出原形
     —— 优化的"样本内幻觉"。

数据：data/cache/board_*.csv 共 10 个真实行业板块日线（不同行业起始日期不同，
     取共同重叠区间保证协方差矩阵完整），重采样为周频收益。
随机种子 42；单文件自包含；总运行时间 < 2 分钟。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示环境也能出图
import matplotlib.pyplot as plt
import cvxpy as cp

# ---------------------------------------------------------------- 全局配置
SEED = 42
rng = np.random.default_rng(SEED)          # 本课唯一的随机数发生器（随机组合云用）
RF = 0.02                                  # 年化无风险利率（夏普比率分母用）
WEEKS_PER_YEAR = 52                        # 周频年化因子：均值×52，协方差×52，波动×√52
EST_WEEKS = 52                             # 滚动实验：用过去 52 周估计
HOLD_WEEKS = 13                            # 滚动实验：持有未来 13 周（一个季度）

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data', 'cache')
OUT_DIR = os.path.join(BASE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


def log(msg):
    print(msg, flush=True)


# ================================================================
# 第 1 步：读取数据 → 周频收益（处理不同行业起始日期不同的问题）
# ================================================================
def load_weekly_returns():
    """读取 10 个行业 CSV，取共同重叠区间的周收益。

    关键点：不同行业起始日期不同（半导体 2019 起、煤炭 2020 起、电力 2021-12 起），
    若直接拼接会有 NaN，协方差矩阵算不出来。做法：转成周收盘价宽表后 dropna()，
    只保留所有行业都有报价的"共同重叠区间"。
    """
    files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith('board_') and f.endswith('.csv'))
    closes = {}
    for f in files:
        name = f.replace('board_', '').replace('.csv', '')
        df = pd.read_csv(os.path.join(DATA_DIR, f), encoding='utf-8-sig', parse_dates=['date'])
        df = df.set_index('date').sort_index()
        closes[name] = df['close'].resample('W-FRI').last()   # 每周最后一个交易日收盘价
    wide = pd.DataFrame(closes).dropna()                       # 取共同重叠区间
    R = wide.pct_change().dropna()                             # 周简单收益率
    log(f'【数据】行业数={R.shape[1]}，共同重叠周收益样本={R.shape[0]} 周，'
        f'区间 {R.index[0].date()} ~ {R.index[-1].date()}')
    return R


# ================================================================
# 第 2 步：均值-方差优化的三类问题（cvxpy，凸优化）
# ================================================================
_SOLVERS = [cp.CLARABEL, cp.SCS, cp.OSQP, cp.ECOS]   # 依次尝试，保证求解稳


def _solve(prob):
    for s in _SOLVERS:
        try:
            prob.solve(solver=s)
        except Exception:
            continue
        if prob.status in ('optimal', 'optimal_inaccurate'):
            return True
    return False


def min_var_weights(mu, Sigma, target=None, allow_short=False):
    """最小方差优化。

    问题：min w'Σw   s.t.  sum(w)=1, [μ'w = target], [w>=0]
    target=None 时即全局最小方差组合 GMV。
    allow_short=False 时不允许卖空（本课默认）。
    返回权重向量；无解返回 None。
    """
    n = len(mu)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if not allow_short:
        cons.append(w >= 0)
    if target is not None:
        cons.append(mu @ w == target)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), cons)
    if _solve(prob) and w.value is not None:
        v = np.asarray(w.value).ravel()
        v = np.clip(v, 0, None) if not allow_short else v   # 清掉 1e-9 级数值噪声
        return v / v.sum()
    # 极端 target（正好压在可行域边界）等式约束可能数值失败 → 退化为不等式 μ'w >= target
    if target is not None:
        cons2 = [cp.sum(w) == 1] + ([w >= 0] if not allow_short else []) + [mu @ w >= target]
        prob2 = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), cons2)
        if _solve(prob2) and w.value is not None:
            v = np.asarray(w.value).ravel()
            v = np.clip(v, 0, None) if not allow_short else v
            return v / v.sum()
    return None


def max_sharpe_weights(mu, Sigma, rf=RF):
    """最大夏普组合（切点组合），只做不允许卖空版本。

    max (μ'w-rf)/sqrt(w'Σw) 不是凸问题，标准凸化技巧（Cornuejols & Tütüncü）：
      令 y = w/κ（κ>0 的缩放），等价于
        min y'Σy   s.t.  (μ-rf)'y = 1,  y >= 0
      解出后 w = y / sum(y)。
    若估计窗内所有行业年化均值都低于 rf，问题不可行 → 返回 None（调用方回退到 GMV）。
    """
    mu = np.asarray(mu, dtype=float)
    if mu.max() <= rf:
        return None
    n = len(mu)
    y = cp.Variable(n)
    cons = [(mu - rf) @ y == 1, y >= 0]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma))), cons)
    if _solve(prob) and y.value is not None:
        yv = np.asarray(y.value).ravel()
        s = yv.sum()
        if s > 1e-12:
            w = yv / s
            return np.clip(w, 0, None)
    return None


def sharpe_of(mu, Sigma, w, rf=RF):
    """给定年化 mu/Sigma 与权重，算组合的（样本内）年化夏普。"""
    w = np.asarray(w, dtype=float)
    ret = float(mu @ w)
    vol = float(np.sqrt(max(w @ Sigma @ w, 1e-18)))
    return (ret - rf) / vol


def perf_stats(weekly_returns, name):
    """从一条周收益序列算年化收益/波动/夏普/最大回撤（样本外实绩统计）。"""
    r = pd.Series(weekly_returns).dropna()
    ann = r.mean() * WEEKS_PER_YEAR
    vol = r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    shp = (ann - RF) / vol if vol > 1e-12 else np.nan
    wealth = (1 + r).cumprod()
    maxdd = float((wealth / wealth.cummax() - 1).min())
    return {'组合': name, '年化收益': ann, '年化波动': vol, '夏普比率': shp,
            '最大回撤': maxdd, '区间起点': r.index[0].date(), '区间终点': r.index[-1].date()}


# ================================================================
# 主流程
# ================================================================
def main():
    # ---------- 1. 数据与年化统计 ----------
    R = load_weekly_returns()
    names = list(R.columns)
    n = len(names)
    mu = R.mean().values * WEEKS_PER_YEAR            # 年化均值向量（周均值×52）
    Sigma = R.cov().values * WEEKS_PER_YEAR          # 年化协方差矩阵（周协方差×52）
    Corr = R.corr().values                           # 相关系数矩阵（无量纲，不年化）
    log('【统计】年化收益：' + ', '.join(f'{k}={v:.1%}' for k, v in zip(names, mu)))
    log(f'【统计】行业间平均相关系数={Corr[np.triu_indices(n, 1)].mean():.2f}，'
        f'最小配对相关={Corr[np.triu_indices(n, 1)].min():.2f}')

    # 图01：相关系数热力图 —— 分散化机会藏在哪里
    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    im = ax.imshow(Corr, cmap='RdYlGn_r', vmin=0.2, vmax=1.0)
    ax.set_xticks(range(n), names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n), names, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{Corr[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                    color='black')
    fig.colorbar(im, ax=ax, label='周收益相关系数', shrink=0.85)
    ax.set_title('图01  10个行业周收益相关系数热力图（共同重叠区间，颜色越绿越适合分散化）')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_相关系数热力图.png'), dpi=150)
    plt.close(fig)

    # ---------- 2. 三类优化：GMV / 最大夏普（全样本，用于教学展示） ----------
    w_gmv = min_var_weights(mu, Sigma)                       # 全局最小方差
    w_ms = max_sharpe_weights(mu, Sigma)                     # 最大夏普（切点组合）
    if w_ms is None:                                         # 理论上不会发生，防御性兜底
        log('【警告】全样本最大夏普不可行，回退 GMV')
        w_ms = w_gmv
    log('【优化】全样本 GMV      年化收益={:.2%} 波动={:.2%} 夏普={:.2f}'.format(
        mu @ w_gmv, np.sqrt(w_gmv @ Sigma @ w_gmv), sharpe_of(mu, Sigma, w_gmv)))
    log('【优化】全样本 最大夏普  年化收益={:.2%} 波动={:.2%} 夏普={:.2f}'.format(
        mu @ w_ms, np.sqrt(w_ms @ Sigma @ w_ms), sharpe_of(mu, Sigma, w_ms)))

    # ---------- 3. 有效前沿（50 档目标收益，允许/不允许卖空两版） ----------
    tgts = np.linspace(mu.min(), mu.max(), 50)
    vol_long, vol_short = [], []
    for t in tgts:
        wl = min_var_weights(mu, Sigma, target=t, allow_short=False)
        ws = min_var_weights(mu, Sigma, target=t, allow_short=True)
        vol_long.append(np.sqrt(wl @ Sigma @ wl) if wl is not None else np.nan)
        vol_short.append(np.sqrt(max(ws @ Sigma @ ws, 0)) if ws is not None else np.nan)
    vol_long = np.array(vol_long)
    vol_short = np.array(vol_short)
    frontier = pd.DataFrame({'目标年化收益': tgts, '多头限制_年化波动': vol_long,
                             '允许卖空_年化波动': vol_short})
    frontier.to_csv(os.path.join(OUT_DIR, 'frontier_points.csv'),
                    index=False, encoding='utf-8-sig')

    # 图02：多头限制版前沿 + 2000 个随机权重组合"云"
    cloud_w = rng.dirichlet(np.ones(n), size=2000)           # 随机长多权重（和=1，w>=0）
    cloud_ret = cloud_w @ mu
    cloud_vol = np.sqrt(np.einsum('ij,jk,ik->i', cloud_w, Sigma, cloud_w))
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    ax.scatter(cloud_vol, cloud_ret, s=6, c='lightgray', alpha=0.5,
               label='2000个随机权重组合（云）', zorder=1)
    ax.plot(vol_long, tgts, 'b-', lw=2.2, label='有效前沿（不允许卖空）', zorder=3)
    ax.scatter([np.sqrt(Sigma[i, i]) for i in range(n)], mu, marker='D', s=48,
               c='tab:orange', edgecolor='k', zorder=4, label='单一行业')
    for i in range(n):
        ax.annotate(names[i], (np.sqrt(Sigma[i, i]), mu[i]), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    ax.scatter(np.sqrt(w_gmv @ Sigma @ w_gmv), mu @ w_gmv, marker='*', s=320,
               c='green', edgecolor='k', zorder=5, label='GMV 全局最小方差')
    ax.scatter(np.sqrt(w_ms @ Sigma @ w_ms), mu @ w_ms, marker='*', s=320,
               c='red', edgecolor='k', zorder=5, label='最大夏普组合')
    # 资本市场线：从无风险利率出发穿过切点组合的射线
    ms_vol = np.sqrt(w_ms @ Sigma @ w_ms)
    xs = np.linspace(0, max(vol_long) * 1.05, 10)
    ax.plot(xs, RF + (mu @ w_ms - RF) / ms_vol * xs, 'r--', lw=1.2, alpha=0.7,
            label='资本市场线 CML')
    ax.set_xlabel('年化波动率 σ')
    ax.set_ylabel('年化收益 μ')
    ax.set_title('图02  马科维茨有效前沿与随机组合云（全部随机组合都在前沿内侧）')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_有效前沿与随机组合云.png'), dpi=150)
    plt.close(fig)

    # 图03：允许卖空 vs 不允许卖空的前沿对比
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    ax.plot(vol_short, tgts, 'r-', lw=2.2, label='允许卖空前沿')
    ax.plot(vol_long, tgts, 'b-', lw=2.2, label='不允许卖空前沿')
    ax.scatter([np.sqrt(Sigma[i, i]) for i in range(n)], mu, marker='D', s=42,
               c='tab:orange', edgecolor='k', zorder=4, label='单一行业')
    for i in range(n):
        ax.annotate(names[i], (np.sqrt(Sigma[i, i]), mu[i]), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('年化波动率 σ')
    ax.set_ylabel('年化收益 μ')
    ax.set_title('图03  允许卖空 vs 不允许卖空：约束把前沿"压"向右上方\n'
                 '（卖空能借力，但低估协方差时杠杆会把估计误差也放大）')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_允许卖空对比.png'), dpi=150)
    plt.close(fig)

    # ---------- 4.【本课灵魂】滚动样本外实验：估计误差的代价 ----------
    # 规则：每 13 周为一个调仓点。调仓时只用"过去 52 周"数据估计 μ/Σ，
    #       优化出最大夏普组合 → 持有未来 13 周 → 滚动推进。
    #       对比同一优化问题"样本内最优夏普"与"样本外实际表现"。
    N = len(R)
    rebal_points = list(range(EST_WEEKS, N, HOLD_WEEKS))     # 第一个样本外周 = 第52周
    log(f'【实验】样本外区间 {R.index[EST_WEEKS].date()} ~ {R.index[-1].date()}，'
        f'共 {N - EST_WEEKS} 周，滚动调仓 {len(rebal_points)} 次')

    ms_oos_weekly = []      # 滚动最大夏普组合的样本外周收益（拼接）
    gmv_oos_weekly = []     # 滚动 GMV 的样本外周收益
    ew_oos_weekly = []      # 等权组合的样本外周收益
    roll_ws = []            # 每期滚动优化出的最大夏普权重（第6步画权重图用）
    rec_is_sharpe, rec_oos_ms_sharpe, rec_oos_ew_sharpe, rec_dates = [], [], [], []
    n_fallback = 0          # 某些熊市窗口"所有行业均值都低于 rf"导致最大夏普不可行的次数

    for t0 in rebal_points:
        R_tr = R.iloc[t0 - EST_WEEKS:t0]                     # 过去52周（仅用历史，无未来信息）
        mu_a = R_tr.mean().values * WEEKS_PER_YEAR
        Sig_a = R_tr.cov().values * WEEKS_PER_YEAR
        w = max_sharpe_weights(mu_a, Sig_a)
        if w is None:                                        # 不可行 → 回退 GMV（并如实计数）
            w = min_var_weights(mu_a, Sig_a)
            n_fallback += 1
        w_g = min_var_weights(mu_a, Sig_a)                   # 同一窗口重新优化 GMV（滚动版）
        if w_g is None:
            w_g = np.full(n, 1.0 / n)
        roll_ws.append(w)
        seg = R.iloc[t0:t0 + HOLD_WEEKS]                     # 未来13周（真正"考卷"）
        r_ms = pd.Series(seg.values @ w, index=seg.index)    # 组合周收益 = 权重·资产周收益
        r_ew = seg.mean(axis=1)
        ms_oos_weekly.append(r_ms)
        gmv_oos_weekly.append(pd.Series(seg.values @ w_g, index=seg.index))
        ew_oos_weekly.append(r_ew)
        a = r_ms.mean() * WEEKS_PER_YEAR
        v = r_ms.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
        rec_is_sharpe.append(sharpe_of(mu_a, Sig_a, w))      # 同一权重在估计窗内的"最优"夏普
        rec_oos_ms_sharpe.append((a - RF) / v if v > 1e-12 else np.nan)
        ae = r_ew.mean() * WEEKS_PER_YEAR
        ve = r_ew.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
        rec_oos_ew_sharpe.append((ae - RF) / ve if ve > 1e-12 else np.nan)
        rec_dates.append(seg.index[0])

    ms_oos = pd.concat(ms_oos_weekly)
    gmv_oos = pd.concat(gmv_oos_weekly)
    ew_oos = pd.concat(ew_oos_weekly)
    is_mean = float(np.nanmean(rec_is_sharpe))               # 平均"样本内最优夏普"
    oos_ms = perf_stats(ms_oos, '')['夏普比率']              # 滚动优化组合的真实样本外夏普
    oos_ew = perf_stats(ew_oos, '')['夏普比率']              # 等权对照的样本外夏普
    log(f'【实验】最大夏普回退GMV次数={n_fallback}/{len(rebal_points)}')
    log(f'【实验】平均样本内最优夏普 = {is_mean:.2f}')
    log(f'【实验】滚动最大夏普 样本外实际夏普 = {oos_ms:.2f}  → 夏普落差 = {is_mean - oos_ms:.2f}')
    log(f'【实验】等权组合 样本外实际夏普 = {oos_ew:.2f}'
        f'（{"等权打赢了优化" if oos_ew > oos_ms else "优化样本外仍胜等权"}）')

    # 图04：样本内最优 vs 样本外实际（左：逐调仓期；右：全程汇总）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6))
    x = np.arange(len(rec_dates))
    ax1.plot(x, rec_is_sharpe, 'o-', c='red', label='样本内最优夏普（估计窗内，幻觉）')
    ax1.plot(x, rec_oos_ms_sharpe, 's-', c='blue', label='样本外实际夏普（未来13周，现实）')
    ax1.plot(x, rec_oos_ew_sharpe, '^--', c='gray', label='等权对照（未来13周）')
    ax1.axhline(0, c='k', lw=0.6)
    ax1.set_xticks(x, [d.date().strftime('%y-%m') for d in rec_dates],
                   rotation=45, fontsize=7.5)
    ax1.set_ylabel('年化夏普比率')
    ax1.set_title('逐调仓期：红色高塔（样本内）vs 蓝色塌方（样本外）')
    ax1.legend(fontsize=8.5)
    bars = ax2.bar(['平均\n样本内最优', '滚动最大夏普\n样本外实际', '等权\n样本外实际'],
                   [is_mean, oos_ms, oos_ew],
                   color=['red', 'blue', 'gray'], width=0.55)
    for b, v in zip(bars, [is_mean, oos_ms, oos_ew]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02 * np.sign(v), f'{v:.2f}',
                 ha='center', va='bottom' if v >= 0 else 'top', fontsize=11, fontweight='bold')
    ax2.axhline(0, c='k', lw=0.6)
    ax2.set_ylabel('年化夏普比率')
    ax2.set_title(f'夏普落差 = {is_mean:.2f} - {oos_ms:.2f} = {is_mean - oos_ms:.2f}\n'
                  f'（估计误差吃掉的正是这部分）')
    fig.suptitle('图04  优化的"样本内幻觉"：52周估计→优化最大夏普→持有13周→滚动推进', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_样本内外夏普落差.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------- 5. 绩效表：等权 / GMV / 最大夏普（固定 vs 滚动） ----------
    # 口径统一：全部在"样本外区间"（第52周之后）上计算，才公平可比。
    R_oos = R.iloc[EST_WEEKS:]
    rows = [
        perf_stats(R_oos.mean(axis=1), '等权组合'),
        perf_stats(pd.Series(R_oos.values @ w_gmv, index=R_oos.index), 'GMV（样本内固定权重）'),
        perf_stats(pd.Series(R_oos.values @ w_ms, index=R_oos.index), '最大夏普（样本内固定权重）'),
        perf_stats(gmv_oos, 'GMV（滚动再优化）'),
        perf_stats(ms_oos, '最大夏普（滚动再优化）'),
    ]
    port = pd.DataFrame(rows)
    port.to_csv(os.path.join(OUT_DIR, 'portfolios.csv'), index=False, encoding='utf-8-sig')
    log('【绩效表】\n' + port.to_string(index=False,
        formatters={'年化收益': '{:.2%}'.format, '年化波动': '{:.2%}'.format,
                    '夏普比率': '{:.2f}'.format, '最大回撤': '{:.2%}'.format}))

    # ---------- 6. 权重堆叠图：优化是否在"押注少数资产" ----------
    # roll_ws 是第4步滚动循环里顺带收集的每期最大夏普权重
    w_roll_mean = np.mean(roll_ws, axis=0)

    combos = [('GMV（全样本）', w_gmv), ('最大夏普（全样本）', w_ms),
              ('最大夏普（滚动·平均权重）', w_roll_mean)]
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    bottom = np.zeros(len(combos))
    for i, nm in enumerate(names):
        vals = np.array([c[1][i] for c in combos])
        ax.bar([c[0] for c in combos], vals, bottom=bottom, label=nm, color=colors[i])
        for k in range(len(combos)):
            if vals[k] > 0.045:      # 只标注占比够大的块，避免图太挤
                ax.text(k, bottom[k] + vals[k] / 2, f'{vals[k]:.0%}', ha='center',
                        va='center', fontsize=8, color='white', fontweight='bold')
        bottom += vals
    ax.set_ylabel('权重')
    ax.set_ylim(0, 1)
    ax.set_title('图05  优化组合的权重堆叠：均值-方差优化天然倾向"押注少数资产"\n'
                 '（最大夏普依赖均值估计，滚动平均权重也随时间大幅摆动）')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '05_组合权重堆叠.png'), dpi=150)
    plt.close(fig)

    # ---------- 汇总输出（README 引用的所有数字都以这里的打印为准） ----------
    log('================ 关键数字汇总（供 README 引用） ================')
    log(f'数据区间: {R.index[0].date()} ~ {R.index[-1].date()}，{N} 周，{n} 个行业')
    log(f'平均两两相关: {Corr[np.triu_indices(n, 1)].mean():.2f}')
    iu = np.triu_indices(n, 1)
    pairs = sorted(zip(Corr[iu], [names[i] for i in iu[0]], [names[j] for j in iu[1]]))
    log('相关性最低的3对行业: ' + '; '.join(f'{a}-{b} ({c:.2f})' for c, a, b in pairs[:3]))
    log(f'全样本GMV: 收益={mu @ w_gmv:.2%} 波动={np.sqrt(w_gmv @ Sigma @ w_gmv):.2%} '
        f'夏普={sharpe_of(mu, Sigma, w_gmv):.2f}')
    log(f'全样本最大夏普: 收益={mu @ w_ms:.2%} 波动={ms_vol:.2%} '
        f'夏普={sharpe_of(mu, Sigma, w_ms):.2f}')
    top = sorted(zip(names, w_ms), key=lambda x: -x[1])[:3]
    log('最大夏普前三大权重: ' + ', '.join(f'{k}={v:.0%}' for k, v in top) +
        '（前三名合计 {:.0%}，押注集中度一目了然）'.format(sum(v for _, v in top)))
    top_g = sorted(zip(names, w_gmv), key=lambda x: -x[1])[:3]
    log('GMV前三大权重: ' + ', '.join(f'{k}={v:.0%}' for k, v in top_g))
    log(f'滚动最大夏普平均权重前三大: ' +
        ', '.join(f'{k}={v:.0%}' for k, v in sorted(zip(names, w_roll_mean), key=lambda x: -x[1])[:3]))
    log(f'前沿(不允许卖空)最小波动={np.nanmin(vol_long):.2%} @ 目标收益={tgts[np.nanargmin(vol_long)]:.2%}')
    log(f'滚动实验: 调仓{len(rebal_points)}次, 最大夏普不可行回退GMV {n_fallback} 次')
    log(f'平均样本内最优夏普 = {is_mean:.2f}')
    log(f'滚动最大夏普样本外实际夏普 = {oos_ms:.2f}，夏普落差 = {is_mean - oos_ms:.2f}')
    log(f'等权样本外实际夏普 = {oos_ew:.2f}')
    log(f'绩效表（样本外口径）:\n{port.to_string(index=False)}')
    log('================ 全部输出已写入 outputs/ ================')


if __name__ == '__main__':
    main()
