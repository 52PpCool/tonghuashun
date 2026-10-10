# -*- coding: utf-8 -*-
"""
18_再平衡红利与波动拖累 —— 再平衡到底赚的是什么钱？（教学版）
=================================================================
这节课回答三个纠缠在一起的问题：
  1) 为什么"涨50%跌33%回到原点"？波动的数学成本 σ²/2 从哪来？
  2) Shannon's demon：一个"典型路径永不赚钱"的资产，为什么配上现金
     定期再平衡就能稳定赚钱？再平衡收割的是什么？
  3) 真实 A 股行业数据上，永不/月度/季度/年度再平衡，哪个好？
     再平衡的逐年贡献能否被分解出来？相关性在其中扮演什么角色？

单文件自包含、随机种子 42、总运行 < 2 分钟。
数据：../data/cache/board_*.csv（10个真实行业）。若缺失则用 rng(42) 造
合成数据兜底并全程标注"合成数据"，保证脚本永不失败。
清洗：行业板块存在"基期跳变"毛刺（单日 ±35% 以上假跳动），统一规则：
     单日 |收益| > 30% 视为跳变，该日收益置 0（A股有涨跌停，现实中不存在）。

产出（outputs/）：
  01_波动拖累曲线.png  02_Shannon恶魔实验.png  03_四种再平衡频率净值.png
  04_再平衡逐年贡献.png  05_相关性对再平衡红利的影响.png  rebalance_metrics.csv
"""
import os
import sys
import glob
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示器环境出图
import matplotlib.pyplot as plt

# ---------------- 全局规范 ----------------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
rng = np.random.default_rng(SEED)          # 全部随机数统一走这一个生成器

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'cache')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

JUMP_TH = 0.30        # 基期跳变判定：单日|收益|>30% 置 0
ANN = 252             # 年化系数（交易日/年）
RF_ANNUAL = 0.02      # 无风险利率（年化），本系列课程夏普统一减 rf=2%
COST_RATE = 0.0003    # 交易成本：双边合计万3（买入+卖出一个来回共0.03%），
                      # 作用于"单边换手金额"（0.5*Σ|目标权重-漂移权重|）
SYNTHETIC = False     # 真实数据缺失时置 True，图表标题会加"合成数据"标注


# =====================================================================
# 0. 数据加载与清洗（真实行业 → 共同重叠期面板）
# =====================================================================
def load_close(path):
    """读单个CSV → 按日期索引的收盘价 Series（剔除 close 缺失/重复日期）。"""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'close']).drop_duplicates('date').sort_values('date')
    return df.set_index('date')['close'].astype(float)


def make_synthetic_boards():
    """兜底：真实数据缺失时用 rng(42) 造 10 个合成行业（仅演示流程）。"""
    global SYNTHETIC
    SYNTHETIC = True
    print('[警告] 未找到 data/cache/board_*.csv，改用 rng(42) 合成数据（结果仅演示）')
    idx = pd.bdate_range('2021-12-29', periods=1119)
    cols = {}
    for i in range(10):
        mu = 0.0002 + 0.0001 * (i % 3)
        sig = 0.012 + 0.004 * (i % 4)
        cols[f'合成行业{i + 1}'] = pd.Series(
            100 * np.cumprod(1 + rng.normal(mu, sig, len(idx))), index=idx)
    return pd.DataFrame(cols)


def build_common_panel():
    """
    读取 10 个行业 → 取共同重叠期（10个行业都有收盘价的交易日）→ 清洗跳变。
    返回 (ret 清洗后日收益面板, panel_days 共同交易日数, n_jumps 清洗跳变数)。
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'board_*.csv')))
    if len(files) < 10:
        close = make_synthetic_boards()
    else:
        cols = {}
        for f in files:
            name = os.path.basename(f)[len('board_'):-4]   # board_银行.csv -> 银行
            cols[name] = load_close(f)
        close = pd.DataFrame(cols).dropna(how='any')       # 共同重叠期矩形面板
    ret = close.pct_change()
    jump = ret.abs() > JUMP_TH                             # 基期跳变：单日|收益|>30%
    n_jumps = int(jump.sum().sum())
    print(f'[数据] 行业数={close.shape[1]}  价格共同期={close.index[0].date()} ~ '
          f'{close.index[-1].date()}  交易日={len(close)}  清洗跳变={n_jumps}处')
    ret = ret.mask(jump, 0.0).iloc[1:].fillna(0.0)         # 跳变置0；首日无收益按0
    return ret, n_jumps


# =====================================================================
# 1. 数学打底：波动拖累  几何收益 ≈ 算术收益 − σ²/2
# =====================================================================
def demo_updown_50_33():
    """"涨50%跌33%回到原点"的精确账本：算术均值 8.3%，几何收益 0。"""
    ups, downs = 0.50, -1.0 / 3.0                    # +50% 与 −33.33%（1.5 × 2/3 = 1）
    arith = (ups + downs) / 2                        # 算术平均收益
    geo = np.sqrt((1 + ups) * (1 + downs)) - 1       # 几何收益（每期复利）
    var = (ups ** 2 + downs ** 2) / 2 - arith ** 2   # 简单收益方差
    print('\n' + '=' * 70)
    print('【1a】涨50%跌33%：算术 vs 几何')
    print(f'  先+50%再-33.33%: 1.5 × {2/3:.4f} = {1.5 * 2/3:.4f}  → 分文未赚')
    print(f'  算术平均收益  = {arith:+.4%}   （每一期"平均"都在赚？！）')
    print(f'  几何收益      = {geo:+.4%}   （长期复利的真实速度）')
    print(f'  收益方差 σ²   = {var:.4%}，  σ²/2 = {var / 2:.4%}')
    print(f'  验证: 算术 {arith:.4%} − σ²/2 {var/2:.4%} = {arith - var/2:+.4%} ≈ 几何 {geo:+.4%}')
    return arith, geo, var


def vol_drag_synthetic():
    """
    合成两资产：完全相同的算术收益（日均值 0.0003 ≈ 年化7.6%），
    但波动不同（日σ=1.0% vs 2.5%）。长期复利后差距巨大。
    """
    T_show = 2520                                   # 展示路径长度：10年
    mu = 0.0003
    sigmas = {'低波动(日σ=1.0%)': 0.010, '高波动(日σ=2.5%)': 0.025}
    paths, stats = {}, {}
    for name, sig in sigmas.items():
        r_show = rng.normal(mu, sig, T_show)        # 画图用的单条路径
        paths[name] = np.cumprod(1 + r_show)
        r_many = rng.normal(mu, sig, (5000, 1260))  # 大样本估计（消除路径运气）
        stats[name] = dict(
            arith_ann=r_many.mean() * ANN,
            geo_ann=np.log1p(r_many).mean() * ANN,  # 年化几何=日对数收益均值×252
            sig_ann=sig * np.sqrt(ANN))
        st = stats[name]
        print(f'  {name}: 实现算术年化={st["arith_ann"]:+.2%}  '
              f'实现几何年化={st["geo_ann"]:+.2%}  '
              f'公式预测 μ−σ²/2={st["arith_ann"] - st["sig_ann"]**2/2:+.2%}')
    return paths, stats, mu


def plot_vol_drag(paths, stats, mu):
    """图01：左=两资产净值路径；右=几何收益随波动率衰减（真实公式+模拟验证）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    colors = {'低波动(日σ=1.0%)': 'tab:blue', '高波动(日σ=2.5%)': 'tab:red'}
    for name, nav in paths.items():
        ax.plot(np.arange(len(nav)) / ANN, nav, color=colors[name], lw=1.4, label=name)
    ax.axhline(1, color='gray', lw=0.8, ls='--')
    ax.set_yscale('log'); ax.set_xlabel('年'); ax.set_ylabel('净值（对数轴）')
    st_lo, st_hi = stats['低波动(日σ=1.0%)'], stats['高波动(日σ=2.5%)']
    ax.set_title(f'相同算术收益（≈{mu*ANN:.1%}/年），波动不同，命运不同\n'
                 f'低波动几何年化 {st_lo["geo_ann"]:+.2%} vs 高波动 {st_hi["geo_ann"]:+.2%}，'
                 f'差 = σ²/2 拖累', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    mu_a = 0.10                                    # 固定算术年化 10%
    sig_grid = np.linspace(0, 0.60, 121)
    g_exact = (1 + mu_a) * np.exp(-sig_grid ** 2 / 2) - 1   # 对数正态精确式
    g_approx = mu_a - sig_grid ** 2 / 2                     # 近似式 μ−σ²/2
    ax.plot(sig_grid * 100, g_exact * 100, 'b-', lw=2,
            label='精确公式 (1+μ)·e^(−σ²/2)−1')
    ax.plot(sig_grid * 100, g_approx * 100, 'r--', lw=2,
            label='近似式 μ − σ²/2')
    # 模拟验证点：每个σ用5000条×1260天长样本估计实现几何年化
    sig_pts = np.arange(0.05, 0.61, 0.05)
    sim = []
    for s in sig_pts:
        r = rng.normal(mu_a / ANN, s / np.sqrt(ANN), (5000, 1260))
        sim.append(np.log1p(r).mean() * ANN * 100)
    ax.plot(sig_pts * 100, sim, 'ko', ms=5, label='模拟（5000路径×5年）')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('年化波动率 σ (%)'); ax.set_ylabel('几何年化收益 (%)')
    ax.set_title(f'波动拖累：算术年化固定 {mu_a:.0%}，σ 越大复利越慢\n'
                 f'σ≈{np.sqrt(2*mu_a):.0%} 时几何收益归零', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    tag = '（合成数据）' if SYNTHETIC else ''
    fig.suptitle(f'图01 波动拖累：几何收益 ≈ 算术收益 − σ²/2{tag}', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_波动拖累曲线.png'), dpi=150)
    plt.close(fig)


# =====================================================================
# 2. Shannon's demon：波动本身可以被"收割"
# =====================================================================
def shannon_demon():
    """
    资产每期等概率 ×2 或 ×0.5：
      - 每期对数收益期望 = (ln2+ln0.5)/2 = 0 → 几何增长 0，典型路径原地踏步；
      - 但算术期望因子 = 1.25（"期望收益"为正，被极少数幸运路径撑大）。
    与 50%现金 + 50%该资产、每期再平衡对比：
      组合每期因子 = 1 + 0.5×(f−1) ∈ {1.5, 0.75}，
      几何增长 = √(1.5×0.75) − 1 = +6.07%/期 → 波动被再平衡"收割"了。
    """
    T, n_paths = 120, 20000
    f = np.where(rng.random((n_paths, T)) < 0.5, 2.0, 0.5)   # 资产每期乘子
    nav_asset = np.cumprod(f, axis=1)
    nav_mix = np.cumprod(1 + 0.5 * (f - 1), axis=1)          # 50/50再平衡（现金收益0）
    fin_a, fin_m = nav_asset[:, -1], nav_mix[:, -1]
    print('\n' + '=' * 70)
    print(f"【2】Shannon's demon：×2/×0.5 等概率，{T}期，{n_paths}条路径")
    print(f'  资产单独持有 : 终值中位数={np.median(fin_a):.3f}  终值均值={fin_a.mean():.3e}'
          f'  （均值被极少数路径撑爆，中位数才是"你"）')
    print(f'  50/50再平衡  : 终值中位数={np.median(fin_m):.1f}  终值均值={fin_m.mean():.1f}')
    print(f'  理论: 资产几何=√(2×0.5)−1={np.sqrt(2*0.5)-1:+.2%}/期 | '
          f'组合几何=√(1.5×0.75)−1={np.sqrt(1.5*0.75)-1:+.2%}/期')
    print(f'  实证: 资产中位每期={np.median(fin_a)**(1/T)-1:+.2%}  '
          f'组合中位每期={np.median(fin_m)**(1/T)-1:+.2%}')
    return nav_asset, nav_mix, fin_a, fin_m, T


def plot_shannon(nav_asset, nav_mix, fin_a, fin_m, T):
    """图02：左=样本路径对比（对数轴）；右=终值分布（对数轴）+中位数。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for i in range(3):
        ax.plot(nav_asset[i], color='tab:red', lw=1, alpha=0.7,
                label='资产单独持有' if i == 0 else None)
    ax.plot(nav_mix[0], color='tab:blue', lw=1.6, label='50%现金+50%资产 每期再平衡')
    ax.set_yscale('log'); ax.set_xlabel('期'); ax.set_ylabel('净值（对数轴）')
    ax.set_title('同样掷硬币，"半仓再平衡"稳定增长，满仓原地踏步', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    bins = np.linspace(-4, 8, 80)
    ax.hist(np.log10(fin_a), bins=bins, color='tab:red', alpha=0.55,
            label=f'资产终值：中位={np.median(fin_a):.2f}，'
                  f'均值={fin_a.mean():.1e}（被极少数路径撑大）')
    ax.hist(np.log10(fin_m), bins=bins, color='tab:blue', alpha=0.55,
            label=f'再平衡终值：中位={np.median(fin_m):.0f}，均值={fin_m.mean():.1e}')
    ax.axvline(np.log10(np.median(fin_a)), color='tab:red', ls='--', lw=1.5)
    ax.axvline(np.log10(np.median(fin_m)), color='tab:blue', ls='--', lw=1.5)
    ax.set_xlabel('终值 log10'); ax.set_ylabel('路径数')
    ax.set_title(f'{T}期后终值分布（{fin_a.shape[0]}条路径）', fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    tag = '（合成数据）' if SYNTHETIC else ''
    fig.suptitle(f"图02 Shannon's demon：再平衡从波动中收割收益{tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_Shannon恶魔实验.png'), dpi=150)
    plt.close(fig)


# =====================================================================
# 3. 真实数据：10行业等权，四种再平衡频率回测
# =====================================================================
def period_end_dates(index, freq):
    """取每个自然月/季/年的最后一个交易日作为再平衡日（排除样本最后一天）。"""
    per = index.to_period({'M': 'M', 'Q': 'Q', 'Y': 'Y'}[freq])
    s = pd.Series(index, index=index)
    ends = s.groupby(per).last()
    return set(ends[ends < index[-1]])


def run_strategy(ret, freq, cost_rate=COST_RATE):
    """
    等权组合回测（权重逐日漂移）：
      日收益 r_p = w·r（用昨日收盘后的权重）→ 权重漂移 w_i←w_i(1+r_i)/(1+r_p)
      若当日为再平衡日：换手=0.5Σ|w−1/k|，成本=换手×双边万3，nav×=(1−成本)，
                        w 重置 1/k。
    freq: 'never' 买入持有漂移 / 'M' 月度 / 'Q' 季度 / 'Y' 年度
    返回 dict(nav, daily, n_rebal, turnover, cost)。
    """
    r = ret.values
    dates = ret.index
    n, k = r.shape
    rebal = set() if freq == 'never' else period_end_dates(dates, freq)
    w = np.full(k, 1.0 / k)
    nav = np.empty(n); nav[0] = 1.0
    daily = np.zeros(n)
    turnover = 0.0; cost_total = 0.0
    for t in range(1, n):
        rp = w @ r[t]                      # 当日组合收益（期初权重）
        daily[t] = rp
        nav[t] = nav[t - 1] * (1 + rp)
        w = w * (1 + r[t]) / (1 + rp)      # 权重随行情漂移
        if dates[t] in rebal:              # 收盘再平衡
            turn = 0.5 * np.abs(w - 1.0 / k).sum()   # 单边换手
            cost = turn * cost_rate                  # 双边合计万3
            nav[t] *= (1 - cost)
            daily[t] = nav[t] / nav[t - 1] - 1       # 把成本计入当日收益
            turnover += turn; cost_total += cost
            w = np.full(k, 1.0 / k)
    return dict(nav=pd.Series(nav, index=dates), daily=pd.Series(daily, index=dates),
                n_rebal=len(rebal), turnover=turnover, cost=cost_total)


def perf_stats(res):
    """年化收益/波动/夏普(rf=2%)/最大回撤。年化基数=日收益个数。"""
    nav, daily = res['nav'], res['daily'].iloc[1:]
    n_ret = len(daily)
    ann_ret = nav.iloc[-1] ** (ANN / n_ret) - 1
    ann_vol = daily.std(ddof=1) * np.sqrt(ANN)
    dd = (nav / nav.cummax() - 1).min()
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe, dd


def real_data_rebalance(ret):
    """四种频率对比 + 图03 + metrics CSV。"""
    print('\n' + '=' * 70)
    print('【3】真实数据：10行业等权，四种再平衡频率（成本=双边万3）')
    freqs = [('never', '永不(买入持有)'), ('M', '月度'), ('Q', '季度'), ('Y', '年度')]
    rows, navs, ress = [], {}, {}
    for fq, name in freqs:
        res = run_strategy(ret, fq)
        a, v, s, d = perf_stats(res)
        rows.append(dict(策略=name, 年化收益=f'{a:.2%}', 年化波动=f'{v:.2%}',
                         夏普=f'{s:.3f}', 最大回撤=f'{d:.2%}',
                         再平衡次数=res['n_rebal'],
                         累计单边换手=f"{res['turnover']:.2f}倍",
                         估算成本占期初=f"{res['cost']:.2%}",
                         期末净值=f"{res['nav'].iloc[-1]:.4f}"))
        navs[name], ress[fq] = res['nav'], res
        print(f'  {name:8s}: 年化={a:+.2%} 波动={v:.2%} 夏普={s:.3f} '
              f'最大回撤={d:.2%} 再平衡{res["n_rebal"]:3d}次 '
              f'累计成本={res["cost"]:.2%} 期末净值={res["nav"].iloc[-1]:.4f}')
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT_DIR, 'rebalance_metrics.csv'),
                   index=False, encoding='utf-8-sig')
    print('[输出] outputs/rebalance_metrics.csv')

    # ---- 图03：净值 + 回撤 ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax = axes[0]
    colors = {'never': 'tab:gray', 'M': 'tab:red', 'Q': 'tab:orange', 'Y': 'tab:green'}
    for fq, name in freqs:
        nav = navs[name]
        a = nav.iloc[-1] ** (ANN / (len(nav) - 1)) - 1
        ax.plot(nav.index, nav.values, color=colors[fq], lw=1.4,
                label=f'{name}（年化{a:+.1%}，期末{nav.iloc[-1]:.2f}）')
    ax.set_yscale('log'); ax.set_ylabel('净值（对数轴）')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title('10行业等权组合：四种再平衡频率（期初各1/10，双边万3成本）'
                 + ('（合成数据）' if SYNTHETIC else ''), fontsize=12)
    ax = axes[1]
    for fq, name in freqs:
        nav = navs[name]
        ax.plot(nav.index, (nav / nav.cummax() - 1) * 100, color=colors[fq], lw=1)
    ax.set_ylabel('回撤 (%)'); ax.grid(alpha=0.3)
    ax.set_xlabel('日期（A股2022~2026样本内结果只是历史，不代表未来）')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_四种再平衡频率净值.png'), dpi=150)
    plt.close(fig)
    return ress, metrics


# =====================================================================
# 4. 机制拆解：再平衡收益 = 买入持有收益 + 再平衡效应
# =====================================================================
def decompose_two_assets(ret):
    """
    双资产精确拆解（连续时间精确、日频高精度近似）：
      50/50逐日再平衡的对数增长 = 0.5·g1 + 0.5·g2 + D，
      再平衡溢价 D = Var(r1 − r2) / 8  ≥ 0。
    直觉：D 只依赖两资产收益"差"的方差——涨跌互见（低相关、高波动）时 D 大。
    实证恒等式（AM-GM，逐日非负）：
      D̂ = (1/T)Σ [ ln(1+0.5r1+0.5r2) − 0.5·ln(1+r1) − 0.5·ln(1+r2) ]
    """
    print('\n' + '=' * 70)
    print('【4a】双资产精确分解（示例：半导体 vs 银行）')
    names = list(ret.columns)
    if '半导体' in names and '银行' in names:      # 合成数据兜底时取前两列
        a, b = ret['半导体'], ret['银行']
    else:
        a, b = ret[names[0]], ret[names[1]]
        print('  （合成数据：改用前两个合成行业演示）')
    r1, r2 = a.values, b.values
    T = len(r1)
    g1, g2 = np.log1p(r1).mean() * ANN, np.log1p(r2).mean() * ANN
    mix_daily = 0.5 * (r1 + r2)                       # 逐日再平衡的组合日收益
    g_mix = np.log1p(mix_daily).mean() * ANN
    d_emp = (np.log1p(mix_daily) - 0.5 * np.log1p(r1)
             - 0.5 * np.log1p(r2)).mean() * ANN       # AM-GM 实证溢价
    d_theory = np.var(r1 - r2, ddof=1) / 8 * ANN      # 理论 Var(r1−r2)/8
    corr = np.corrcoef(r1, r2)[0, 1]
    # 实务口径：期初各半买入持有（永不平衡）组合的对数年化
    nav1, nav2 = np.cumprod(1 + r1), np.cumprod(1 + r2)
    g_bh = np.log(0.5 * nav1[-1] + 0.5 * nav2[-1]) / T * ANN
    print(f'  相关系数={corr:.2f}  σ1(年化)={r1.std()*np.sqrt(ANN):.1%}  '
          f'σ2(年化)={r2.std()*np.sqrt(ANN):.1%}')
    print(f'  逐日再平衡 50/50 对数年化 g_mix = {g_mix:+.2%}')
    print(f'  两腿加权几何    0.5(g1+g2)  = {0.5*(g1+g2):+.2%}')
    print(f'  → 实证再平衡溢价 D̂ = {d_emp:+.2%}   （逐日非负，AM-GM 恒等式）')
    print(f'  理论公式 D = Var(r1−r2)/8 = {d_theory:+.2%}   （与实证仅差高阶小量）')
    print(f'  对照：期初各半买入持有 g_bh = {g_bh:+.2%}，g_mix − g_bh = {g_mix - g_bh:+.2%}')
    return d_emp, d_theory, corr


def yearly_contribution(ress):
    """
    实证近似拆解（10行业等权）：
      某年再平衡贡献 = 当年"月度再平衡"年收益 − 当年"永不再平衡"年收益
    （月度策略的日收益已含成本，故贡献是净口径；可能为正也可能为负。）
    """
    print('\n' + '=' * 70)
    print('【4b】逐年再平衡贡献（月度再平衡 − 买入持有，含成本）')
    rm, rn = ress['M']['daily'], ress['never']['daily']
    years = [y for y in sorted(set(rm.index.year))
             if (rm.index.year == y).sum() >= 10]   # 剔除不足10个交易日的残缺年
    contrib = {}
    for i, y in enumerate(years):
        cm = (1 + rm[rm.index.year == y]).prod() - 1
        cn = (1 + rn[rn.index.year == y]).prod() - 1
        contrib[y] = cm - cn
        tag_y = '*' if i == len(years) - 1 else ' '
        print(f'  {y}{tag_y}: 月度={cm:+.2%}  永不={cn:+.2%}  贡献={cm - cn:+.2%}')
    print(f'  全样本累计贡献 = {sum(contrib.values()):+.2%}（*为截至9月中）')
    return contrib


def plot_contribution(contrib):
    """图04：逐年再平衡贡献柱状图（正负着色）。"""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [f'{y}{"*" if i == len(contrib) - 1 else ""}'
              for i, y in enumerate(contrib)]
    vals = np.array(list(contrib.values())) * 100
    colors = ['tab:red' if v >= 0 else 'tab:blue' for v in vals]
    ax.bar(labels, vals, color=colors, width=0.55)
    for i, v in enumerate(vals):
        # 标签统一放在柱体上方（负值也放在零轴上方，避免与x轴刻度重叠）
        ax.text(i, max(v, 0) + 0.06, f'{v:+.2f}%', ha='center', va='bottom',
                fontsize=10, color='k')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('年收益差（百分点）')
    ax.set_title('逐年"再平衡贡献" = 月度再平衡年收益 − 买入持有年收益\n'
                 '有正有负：震荡/分化年收割，单边趋势年拖后腿（2026*截至9月中）',
                 fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    tag = '（合成数据）' if SYNTHETIC else ''
    fig.suptitle(f'图04 再平衡逐年贡献拆解{tag}', fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, '04_再平衡逐年贡献.png'), dpi=150)
    plt.close(fig)


# =====================================================================
# 5. 相关性的角色：再平衡红利主要来自"低相关 + 高波动"
# =====================================================================
def correlation_experiment():
    """
    合成两资产（日μ=0.03%、σ=1.5%，改变相关系数ρ）：
      理论再平衡溢价 D = Var(r1−r2)/8 = σ²(1−ρ)/4 → ρ 越低、σ 越大，D 越大。
      ρ=0.9 时 D 只有 ρ=0 的十分之一。
    模拟：每配置1000条路径×1260天，按 AM-GM 恒等式逐路径计算年化溢价。
    """
    print('\n' + '=' * 70)
    print('【5】相关性实验：再平衡溢价 D = σ²(1−ρ)/4（年化）')
    mu, sig, T, n_paths = 0.0003, 0.015, 1260, 1000

    def sim_premium(rho, sigma):
        z = rng.standard_normal((n_paths, T, 2))
        e2 = rho * z[..., 0] + np.sqrt(1 - rho ** 2) * z[..., 1]
        r1, r2 = mu + sigma * z[..., 0], mu + sigma * e2
        mix = np.log1p(0.5 * (r1 + r2)) \
            - 0.5 * np.log1p(r1) - 0.5 * np.log1p(r2)   # AM-GM 逐日溢价
        return mix.mean(axis=1) * ANN, sigma ** 2 * (1 - rho) / 4 * ANN

    rho_list = [0.9, 0.5, 0.0, -0.5]
    rho_res = [sim_premium(r, sig) for r in rho_list]
    for rho, (sim, th) in zip(rho_list, rho_res):
        print(f'  ρ={rho:+.1f}: 模拟年化溢价={sim.mean():+.2%}±{sim.std():.2%}  '
              f'理论={th:+.2%}')
    sig_list = [0.010, 0.015, 0.020, 0.030]
    sig_res = [sim_premium(0.0, s) for s in sig_list]
    for s, (sim, th) in zip(sig_list, sig_res):
        print(f'  ρ=0, σ={s:.1%}: 模拟={sim.mean():+.2%}±{sim.std():.2%}  理论={th:+.2%}')
    return rho_list, rho_res, sig_list, sig_res, sig, T


def plot_correlation(rho_list, rho_res, sig_list, sig_res, sig, T):
    """图05：左=溢价 vs ρ；右=溢价 vs σ（模拟均值±std + 理论曲线）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    grid = np.linspace(-0.6, 0.95, 100)
    ax.plot(grid, sig ** 2 * (1 - grid) / 4 * ANN * 100, 'b-', lw=2,
            label=r'理论 $\sigma^2(1-\rho)/4$')
    means = [r[0].mean() * 100 for r in rho_res]
    stds = [r[0].std() * 100 for r in rho_res]
    ax.errorbar(rho_list, means, yerr=stds, fmt='ro', ms=7, capsize=4,
                label='模拟（1000路径×5年）')
    for x, m in zip(rho_list, means):
        ax.annotate(f'{m:.2f}%', (x, m), textcoords='offset points',
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel('相关系数 ρ'); ax.set_ylabel('再平衡年化溢价 (%)')
    ax.set_title(f'σ={sig:.1%} 固定：ρ=0.9 的红利只有 ρ=0 的约 1/10', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    grid_s = np.linspace(0.005, 0.035, 100)
    ax.plot(grid_s * 100, grid_s ** 2 / 4 * ANN * 100, 'b-', lw=2,
            label=r'理论 $\sigma^2/4$（ρ=0）')
    means = [r[0].mean() * 100 for r in sig_res]
    stds = [r[0].std() * 100 for r in sig_res]
    ax.errorbar(np.array(sig_list) * 100, means, yerr=stds, fmt='ro', ms=7,
                capsize=4, label='模拟（ρ=0）')
    for x, m in zip(sig_list, means):
        ax.annotate(f'{m:.2f}%', (x * 100, m), textcoords='offset points',
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel('单资产日波动 σ (%)'); ax.set_ylabel('再平衡年化溢价 (%)')
    ax.set_title('ρ=0 固定：溢价随 σ² 增长——高波动才有的红利', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle('图05 再平衡红利的来源：低相关 × 高波动（合成数据）', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '05_相关性对再平衡红利的影响.png'), dpi=150)
    plt.close(fig)


# =====================================================================
# main
# =====================================================================
def main():
    print('=' * 70)
    print('18_再平衡红利与波动拖累 （随机种子 42）')
    print('=' * 70)

    # 1) 数学打底：波动拖累
    demo_updown_50_33()
    paths, stats, mu = vol_drag_synthetic()
    plot_vol_drag(paths, stats, mu)
    print('[输出] outputs/01_波动拖累曲线.png')

    # 2) Shannon's demon
    nav_a, nav_m, fin_a, fin_m, T = shannon_demon()
    plot_shannon(nav_a, nav_m, fin_a, fin_m, T)
    print('[输出] outputs/02_Shannon恶魔实验.png')

    # 3) 真实数据：四种再平衡频率
    ret, _ = build_common_panel()
    ress, metrics = real_data_rebalance(ret)

    # 4) 机制拆解
    decompose_two_assets(ret)
    contrib = yearly_contribution(ress)
    plot_contribution(contrib)
    print('[输出] outputs/04_再平衡逐年贡献.png')

    # 5) 相关性角色
    rho_list, rho_res, sig_list, sig_res, sig, T = correlation_experiment()
    plot_correlation(rho_list, rho_res, sig_list, sig_res, sig, T)
    print('[输出] outputs/05_相关性对再平衡红利的影响.png')

    print('\n全部完成，图表与 CSV 见 outputs/')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # 课程硬性要求：任何异常都不以非0退出（打印完整堆栈便于排查）
        traceback.print_exc()
        print('[异常] 运行中断，但按课程规范以 0 退出（请检查上方堆栈修复）')
        sys.exit(0)
