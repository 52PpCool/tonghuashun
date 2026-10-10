# -*- coding: utf-8 -*-
"""
23_贝叶斯估计与统计功效
========================
回答量化研究里最深的两问：
  Q1: "年化夏普1.5"这个数字到底有多可信？           → 贝叶斯后验区间
  Q2: 一个真实有效的因子/策略，要多少年数据才能"证明自己"？ → 统计功效模拟

全部用 numpy/scipy 的共轭分布 + 蒙特卡洛模拟实现（不依赖 MCMC 库），
随机种子统一 42，可复现。运行时间 < 2 分钟。

五个实验：
  1. 正态-正态共轭：先验 → 后验随样本量演化（贝叶斯 = 证据 × 先验）
  2. 参数不确定性 → 上证指数过去120个月夏普的后验分布与90%可信区间
  3. 统计功效模拟：IC=0.03的因子、夏普1.0的策略各需要几年回测？
  4. 多重检验：同时检验20个噪声因子时的假阳性率（呼应课16）
  5. 汇总表 outputs/power_results.csv：各场景达到50%/80%功效所需样本量
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------
# 先验设定（教学重点：先验从哪来、为什么敢设）
#   μ ~ N(0.5%/月, 2%/月)：均值0.5%/月 ≈ 年化6%，是"股市长期给投资者的
#   风险补偿"这一常识量级；标准差2%/月故意设得很宽——相当于说
#   "我大概知道量级，但对具体取值非常不置可否"（弱先验）。
#   弱先验的好处：数据量稍大时后验几乎完全由数据决定，先验只防止极端。
# ------------------------------------------------------------------
PRIOR_MU = 0.005   # 先验均值：0.5% / 月
PRIOR_SD = 0.02    # 先验标准差：2% / 月（很宽 → 弱先验）


# ==================================================================
# 第0部分 数据加载：上证指数日线 → 月度收益（真实数据，缺失则合成兜底）
# ==================================================================
def load_monthly_returns():
    """返回 (月度收益序列, 数据来源说明)"""
    path = os.path.join(BASE, '..', 'data', 'cache', 'index_000001_上证指数.csv')
    try:
        df = pd.read_csv(path, encoding='utf-8-sig',
                         parse_dates=['date']).set_index('date')
        close = df['close'].resample('ME').last().dropna()   # 取每月收盘
        ret = close.pct_change().dropna()                    # 月度简单收益
        if len(ret) < 130:
            raise ValueError(f'月度样本只有{len(ret)}个，不足130')
        return ret, '真实数据：上证指数日线(2010-01~至今)聚合为月度收益'
    except Exception as e:
        print(f'[警告] 真实数据读取失败({e})，改用合成数据(种子42)')
        rng = np.random.default_rng(42)
        idx = pd.date_range('2016-09-30', periods=200, freq='ME')
        ret = pd.Series(rng.normal(0.004, 0.055, 200), index=idx, name='ret')
        return ret, '合成数据（真实数据缺失时的兜底，已标注）'


# ==================================================================
# 第1部分 贝叶斯均值估计：正态-正态共轭更新
# ==================================================================
def posterior_normal_normal(x, prior_mu=PRIOR_MU, prior_sd=PRIOR_SD, sigma2=None):
    """正态-正态共轭更新（σ视为已知，默认用全样本方差近似）。

    先验： μ ~ N(μ0, τ0²)                 ← τ0 = prior_sd
    似然： x_i | μ ~ N(μ, σ²)，i=1..n     ← σ 视为已知（σ²由长期数据估计）
    后验： μ | x ~ N(μn, τn²)，其中
        精度相加：  1/τn² = 1/τ0² + n/σ²          ← "信息 = 证据 + 先验"
        精度加权：  μn = (μ0/τ0² + n·x̄/σ²) / (1/τ0² + n/σ²)
    n 越大，n/σ² 项越主导 → 后验均值趋近样本均值、后验越窄，先验影响递减。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if sigma2 is None:
        sigma2 = x.var(ddof=1)                # σ²已知近似（n=1时需外部传入）
    prec0 = 1.0 / prior_sd ** 2               # 先验精度 1/τ0²
    prec_data = n / sigma2                    # 数据精度 n/σ²
    prec_post = prec0 + prec_data             # 后验精度 = 先验精度 + 数据精度
    mu_post = (prior_mu * prec0 + n * x.mean() / sigma2) / prec_post
    sd_post = 1.0 / np.sqrt(prec_post)
    return mu_post, sd_post


def part1_bayes_mean(ret, source):
    """先验→后验随样本量(1/6/12/36/120个月)的演化图"""
    print('=' * 70)
    print('第1部分 贝叶斯均值估计（正态-正态共轭） |', source)
    print('=' * 70)
    ret120 = ret.iloc[-120:]                  # 最近120个月
    sigma2_full = ret120.var(ddof=1)          # σ视为已知：用120个月估计的方差
    print(f'先验：μ ~ N({PRIOR_MU*100:.1f}%/月, {PRIOR_SD*100:.1f}%/月)  [弱先验]')
    print(f'{"n(月)":>6} | {"样本均值":>8} | {"后验均值":>8} | {"后验SD":>7} | 数据精度占比')
    rows = []
    grid = np.linspace(-0.03, 0.04, 600)      # μ 轴（%）
    fig, ax = plt.subplots(figsize=(10, 6))
    # 先验密度（虚线黑）：还没有任何数据时的信念
    prior_pdf = stats.norm.pdf(grid, PRIOR_MU, PRIOR_SD)
    ax.plot(grid * 100, prior_pdf, 'k--', lw=2, label='先验 N(0.5%, 2%²)')

    for i, n in enumerate([1, 6, 12, 36, 120]):
        x = ret120.iloc[:n]                   # 按时间顺序"喂"给模型前n个月
        mu_n, sd_n = posterior_normal_normal(x, sigma2=sigma2_full)
        share = (n / sigma2_full) / (1 / PRIOR_SD ** 2 + n / sigma2_full)
        rows.append([n, x.mean(), mu_n, sd_n, share])
        print(f'{n:>6} | {x.mean()*100:>7.3f}% | {mu_n*100:>7.3f}% | '
              f'{sd_n*100:>6.3f}% | {share*100:>5.1f}%')
        pdf = stats.norm.pdf(grid, mu_n, sd_n)
        ax.plot(grid * 100, pdf, lw=2, color=plt.cm.viridis(i / 4),
                label=f'n={n}个月 后验 N({mu_n*100:.2f}%, {sd_n*100:.2f}%²)')
        ax.fill_between(grid * 100, pdf, alpha=0.08, color=plt.cm.viridis(i / 4))

    ax.axvline(ret120.mean() * 100, color='crimson', ls=':', lw=1.5,
               label=f'120个月样本均值 {ret120.mean()*100:.2f}%')
    ax.set_xlabel('月度均值收益 μ (%)')
    ax.set_ylabel('概率密度')
    ax.set_title('先验 → 后验：数据越多，后验越窄，先验影响越小（上证指数月收益）')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '01_先验到后验演化.png'), dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows, columns=['n月', '样本均值', '后验均值', '后验SD', '数据精度占比'])


# ==================================================================
# 第2部分 参数不确定性 → 夏普的后验分布（90%可信区间）
# ==================================================================
def part2_sharpe_posterior(ret, n_sim=10000):
    """后验模拟 μ（正态）与 σ（卡方近似），每次算夏普 → 夏普后验直方图。

    μ | 数据 ~ N(μn, τn²)                    ← 第1部分的共轭后验，直接抽样
    σ² | 数据 ~ (n-1)s² / χ²(n-1)            ← 方差的卡方后验（缩放逆卡方）
    每次抽样一对 (μ, σ) 算年化夏普 SR = μ/σ·√12，
    一万次抽样得到夏普的"后验分布"——点估计只是这个分布的中心。
    """
    print('=' * 70)
    print('第2部分 参数不确定性 → 夏普后验分布（过去120个月，模拟%d次）' % n_sim)
    print('=' * 70)
    rng = np.random.default_rng(42)           # 独立种子，保证可复现
    x = ret.iloc[-120:]
    n = len(x)
    mu_n, sd_mu = posterior_normal_normal(x)          # μ 的后验（σ²由本窗口估计）
    s2 = x.var(ddof=1)
    mu_sim = rng.normal(mu_n, sd_mu, n_sim)           # ← μ 后验正态抽样
    chi2 = rng.chisquare(n - 1, n_sim)                # ← σ 后验：卡方近似抽样
    sigma_sim = np.sqrt((n - 1) * s2 / chi2)
    sr_sim = mu_sim / sigma_sim * np.sqrt(12)         # 年化夏普的后验样本

    sr_point = x.mean() / x.std(ddof=1) * np.sqrt(12) # 传统的夏普点估计
    lo, med, hi = np.percentile(sr_sim, [5, 50, 95])
    p_neg = (sr_sim < 0).mean()                       # P(真实夏普<0)
    p_gt1 = (sr_sim > 1.0).mean()                     # P(真实夏普>1.0)

    print(f'样本区间：{x.index[0].date()} ~ {x.index[-1].date()}（n={n}个月）')
    print(f'夏普点估计（教科书做法）：{sr_point:.3f}')
    print(f'夏普后验90%可信区间：[{lo:.3f}, {hi:.3f}]  宽度={hi-lo:.3f}')
    print(f'后验中位数={med:.3f} | P(夏普<0)={p_neg*100:.1f}% | P(夏普>1.0)={p_gt1*100:.1f}%')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sr_sim, bins=80, density=True, color='steelblue', alpha=0.75,
            edgecolor='white', lw=0.3)
    ax.axvline(sr_point, color='crimson', lw=2, label=f'点估计 {sr_point:.2f}')
    ax.axvline(lo, color='k', ls='--', lw=1.5, label=f'90%CI下限 {lo:.2f}')
    ax.axvline(hi, color='k', ls='--', lw=1.5, label=f'90%CI上限 {hi:.2f}')
    ax.axvspan(lo, hi, color='gold', alpha=0.15)
    ax.set_xlabel('年化夏普比率')
    ax.set_ylabel('后验密度')
    ax.set_title(f'上证指数近10年夏普的后验分布：90%可信区间宽达 {hi-lo:.2f}\n'
                 f'（点估计只是分布的中心，P(夏普<0)={p_neg*100:.1f}%）')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '02_夏普后验分布.png'), dpi=150)
    plt.close(fig)

    ci = pd.DataFrame([{
        '样本区间': f'{x.index[0].date()}~{x.index[-1].date()}',
        '月数': n,
        '夏普点估计(年化)': round(sr_point, 3),
        '后验中位数': round(med, 3),
        '90%CI下限': round(lo, 3),
        '90%CI上限': round(hi, 3),
        'CI宽度': round(hi - lo, 3),
        'P(夏普<0)': f'{p_neg*100:.1f}%',
        'P(夏普>1.0)': f'{p_gt1*100:.1f}%',
        '月均收益90%CI(%)': f'[{(mu_n-1.645*sd_mu)*100:.2f}, {(mu_n+1.645*sd_mu)*100:.2f}]',
        '月波动90%CI(%)': f'[{np.percentile(sigma_sim,5)*100:.2f}, {np.percentile(sigma_sim,95)*100:.2f}]',
    }])
    ci.to_csv(os.path.join(OUT, 'sharpe_ci.csv'), index=False, encoding='utf-8-sig')
    return sr_sim, (sr_point, lo, med, hi, p_neg, p_gt1)


# ==================================================================
# 第3部分 本课灵魂：统计功效模拟
#   "一个真实有效的策略/因子，需要多少年数据才能被显著检出？"
# ==================================================================
def _spearman_ic_matrix(rho, n_sim_chunk, T, n_stocks, rng):
    """模拟 n_sim_chunk×T 个月的月度RankIC序列。
    每月：n_stocks 只股票的因子值 z 与收益 y=ρ·z+√(1-ρ²)·ε，
    ρ 由目标"真实RankIC"换算（二元正态下 Spearman≈(6/π)arcsin(ρ/2)）。
    向量化计算每月截面的秩相关 = RankIC。返回形状 (n_sim_chunk, T)。
    """
    rho_p = 2 * np.sin(np.pi * rho / 6)       # 目标Spearman → Pearson ρ
    z = rng.standard_normal((n_sim_chunk, T, n_stocks))
    e = rng.standard_normal((n_sim_chunk, T, n_stocks))
    y = rho_p * z + np.sqrt(1 - rho_p ** 2) * e
    # 秩变换：argsort 两次得到 1..n 的名次（Spearman=名次的Pearson相关）
    rz = np.argsort(np.argsort(z, axis=-1), axis=-1).astype(np.float64)
    ry = np.argsort(np.argsort(y, axis=-1), axis=-1).astype(np.float64)
    rz -= rz.mean(-1, keepdims=True)
    ry -= ry.mean(-1, keepdims=True)
    ic = (rz * ry).sum(-1) / np.sqrt((rz ** 2).sum(-1) * (ry ** 2).sum(-1))
    return ic


def simulate_ic_power(rho_target, n_stocks, months_list, n_sim=1000, chunk=200):
    """功效模拟：真实RankIC=rho_target，每月n_stocks只股票，
    回看 T 个月的月度IC做 t 检验(双侧5%)，统计1000次模拟中显著的比例。
    分块模拟控制内存。返回 (months_list, 功效数组, 平均月度IC标准差)。
    """
    rng = np.random.default_rng(42)
    power = np.zeros(len(months_list))
    ic_std_all = []
    for j, T in enumerate(months_list):
        sig = []
        # 自适应分块：单块元素量控制在约2e6，防止大截面吃满内存
        chunk_eff = max(10, min(chunk, int(2e6 // (T * n_stocks))))
        for s in range(0, n_sim, chunk_eff):
            k = min(chunk_eff, n_sim - s)
            ic = _spearman_ic_matrix(rho_target, k, T, n_stocks, rng)  # (k,T)
            ic_mean = ic.mean(1)
            ic_std = ic.std(1, ddof=1)
            t = ic_mean / (ic_std / np.sqrt(T))          # ← 功效模拟的 t 统计量
            t_crit = stats.t.ppf(0.975, T - 1)           # 双侧5%临界值
            sig.append(np.abs(t) > t_crit)
        power[j] = np.concatenate(sig).mean()
        ic_std_all.append(ic.std())
    return months_list, power, float(np.mean(ic_std_all))


def simulate_sharpe_power(sr_true=1.0, months_list=(12, 36, 60, 96, 120),
                          n_sim=1000, days_per_month=21):
    """日频策略年化夏普=sr_true，回测 T 个月后做日收益 t 检验(双侧5%)，
    1000次模拟中显著的比例 = 该回测时长的检出概率（功效）。"""
    rng = np.random.default_rng(42)
    daily_mu = sr_true / np.sqrt(252)         # 日均值（日σ=1 → 年化夏普=sr_true）
    power = np.zeros(len(months_list))
    for j, T in enumerate(months_list):
        n_days = T * days_per_month
        r = rng.standard_normal((n_sim, n_days)) + daily_mu   # 模拟日收益
        t = r.mean(1) / (r.std(1, ddof=1) / np.sqrt(n_days))
        t_crit = stats.t.ppf(0.975, n_days - 1)
        power[j] = (np.abs(t) > t_crit).mean()
    return list(months_list), power


def years_for_power(years, power, target):
    """在模拟功效曲线上插值：达到 target 功效需要几年；达不到返回 nan"""
    for i, p in enumerate(power):
        if p >= target:
            if i == 0:
                return float(years[0])
            y0, y1, p0, p1 = years[i - 1], years[i], power[i - 1], power[i]
            return float(y0 + (target - p0) / (p1 - p0) * (y1 - y0))
    return np.nan


def part3_power():
    print('=' * 70)
    print('第3部分 统计功效模拟：真实有效的策略要几年才能"证明自己"？')
    print('=' * 70)
    results = []   # 汇总到 power_results.csv

    # ---- 3a IC=0.03 的因子，月度截面 N=30（主讲场景，T最长到240个月） ----
    months_ic = [12, 24, 36, 60, 120, 180, 240]
    scenarios = [
        ('因子: RankIC=0.03, 截面N=30', 0.03, 30),
        ('因子: RankIC=0.05, 截面N=30', 0.05, 30),   # 对照：IC更强
        ('因子: RankIC=0.03, 截面N=300', 0.03, 300), # 对照：截面更宽
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['crimson', 'seagreen', 'steelblue']
    for (name, ic, n_stk), c in zip(scenarios, colors):
        Ts, pw, ic_std = simulate_ic_power(ic, n_stk, months_ic, n_sim=1000)
        yrs = np.array(Ts) / 12
        y50 = years_for_power(yrs, pw, 0.5)
        y80 = years_for_power(yrs, pw, 0.8)
        f50 = f'{y50:.1f}' if np.isfinite(y50) else '>20(未达到)'
        f80 = f'{y80:.1f}' if np.isfinite(y80) else '>20(未达到)'
        print(f'[{name}] 月度IC标准差≈{ic_std:.3f} | 功效: ' +
              ' '.join(f'{T}月:{p*100:.0f}%' for T, p in zip(Ts, pw)))
        print(f'    → 50%功效需 {f50} 年, 80%功效需 {f80} 年')
        results.append({'场景': name, '检出标准': '月度IC序列 t检验 双侧5%',
                        '50%功效所需(年)': f50, '80%功效所需(年)': f80,
                        '备注': f'月度IC标准差≈{ic_std:.3f}'})
        ax.plot(yrs, pw * 100, 'o-', color=c, lw=2, label=name)

    for lv, c in [(50, 'gray'), (80, 'gray')]:
        ax.axhline(lv, color=c, ls=':', lw=1)
        ax.text(20.3, lv, f'{lv}%功效', va='center', fontsize=9, color=c)
    ax.set_xlabel('回测样本长度（年）')
    ax.set_ylabel('t检验显著(5%)的比例 = 统计功效 (%)')
    ax.set_title('IC=0.03、截面30只的因子：20年数据都未必"显著"\n'
                 '（月度RankIC t检验，1000次模拟，种子42）')
    ax.set_xlim(0, 21); ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '03_IC功效曲线.png'), dpi=150)
    plt.close(fig)

    # ---- 3b 年化夏普1.0的日频策略，需要几年回测（附夏普0.5对照） ----
    months_sr, pw_sr = simulate_sharpe_power(1.0, (12, 36, 60, 96, 120), 1000)
    months_sr2, pw_sr2 = simulate_sharpe_power(0.5, (12, 36, 60, 96, 120), 1000)
    yrs_sr = np.array(months_sr) / 12
    y50 = years_for_power(yrs_sr, pw_sr, 0.5)
    y80 = years_for_power(yrs_sr, pw_sr, 0.8)
    y50_2 = years_for_power(yrs_sr, pw_sr2, 0.5)
    y80_2 = years_for_power(yrs_sr, pw_sr2, 0.8)
    fmt = lambda v: f'{v:.1f}' if np.isfinite(v) else '>10(未达到)'
    print(f'[策略: 日频年化夏普=1.0] 功效: ' +
          ' '.join(f'{m}月:{p*100:.0f}%' for m, p in zip(months_sr, pw_sr)))
    print(f'    → 50%功效需 {fmt(y50)} 年, 80%功效需 {fmt(y80)} 年回测')
    print(f'[策略: 日频年化夏普=0.5] 功效: ' +
          ' '.join(f'{m}月:{p*100:.0f}%' for m, p in zip(months_sr2, pw_sr2)))
    print(f'    → 50%功效需 {fmt(y50_2)} 年, 80%功效需 {fmt(y80_2)} 年回测')
    results.append({'场景': '策略: 日频年化夏普=1.0', '检出标准': '日收益 t检验 双侧5%',
                    '50%功效所需(年)': fmt(y50), '80%功效所需(年)': fmt(y80),
                    '备注': '12/36/60/96/120个月回测窗，1000次模拟'})
    results.append({'场景': '策略: 日频年化夏普=0.5(对照)', '检出标准': '日收益 t检验 双侧5%',
                    '50%功效所需(年)': fmt(y50_2), '80%功效所需(年)': fmt(y80_2),
                    '备注': '弱策略在10年窗口内几乎无法被证明显著'})

    # ---- 3b 图：夏普1.0检出功效曲线（附夏普0.5对照） ----
    fig, ax = plt.subplots(figsize=(10, 6))
    yrs_grid = np.linspace(0.25, 10, 200)
    # 理论参考线：t ≈ 夏普×√年数（双侧5%）
    theory = stats.norm.cdf(1.0 * np.sqrt(yrs_grid) - 1.96) * 100
    ax.plot(yrs_grid, theory, 'k--', lw=1.2, label='理论近似 t≈SR×√年数')
    ax.plot(yrs_sr, pw_sr * 100, 'o-', color='crimson', lw=2.5,
            label='夏普1.0 策略（模拟，1000次）')
    ax.plot(yrs_sr, pw_sr2 * 100, 'o-', color='steelblue', lw=2,
            label='夏普0.5 策略（模拟，对照）')
    for yv, lab in [(y50, f'50%功效 ≈ {y50:.1f}年'), (y80, f'80%功效 ≈ {y80:.1f}年')]:
        ax.axvline(yv, color='gray', ls=':', lw=1.2)
        ax.annotate(lab, (yv, 8), textcoords='offset points', xytext=(6, 0),
                    fontsize=10, rotation=90, color='dimgray')
    ax.set_xlabel('回测样本长度（年）')
    ax.set_ylabel('t检验显著(5%)的比例 = 检出概率 (%)')
    ax.set_title('年化夏普1.0的真实策略：也要约8年回测才有80%把握"显著"\n'
                 '（日频模拟，种子42）')
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '04_夏普检出功效曲线.png'), dpi=150)
    plt.close(fig)
    return results


# ==================================================================
# 第4部分 多重检验：20个噪声因子里"发现"一个显著的概率
# ==================================================================
def part4_multiple_testing(n_factors=20, months=60, n_sim=1000):
    """20个真实无关(纯噪声)的因子同时检验：每个因子用60个月的月度IC做t检验。
    统计：(a)至少1个|t|>1.96的比例（家族错误率,呼应Bonferroni）
          (b)至少1个|t|>2.0的比例（"t=2.0的发现"有多假）
          (c)Bonferroni校正后(α/20)的假阳性率是否回到5%。
    """
    print('=' * 70)
    print(f'第4部分 多重检验：同时检验{n_factors}个纯噪声因子（{months}个月，{n_sim}次模拟）')
    print('=' * 70)
    rng = np.random.default_rng(42)
    t_all = np.zeros((n_sim, n_factors))
    for f in range(n_factors):                # 20个互不相关的噪声因子
        ic = _spearman_ic_matrix(0.0, n_sim, months, 30, rng)   # 真实IC=0
        t_all[:, f] = ic.mean(1) / (ic.std(1, ddof=1) / np.sqrt(months))
    max_t = np.abs(t_all).max(1)              # 每次模拟里"最强因子"的|t|
    t_bonf = stats.t.ppf(1 - 0.05 / n_factors / 2, months - 1)  # Bonferroni临界t

    p_196 = (max_t > 1.96).mean()
    p_20 = (max_t > 2.0).mean()
    p_bonf = (max_t > t_bonf).mean()
    exp_sig = (np.abs(t_all) > 1.96).sum(1).mean()
    print(f'单因子检验5%显著 → 至少1个因子"显著"的比例: {p_196*100:.1f}%'
          f'（理论 1-(0.95)^{n_factors}={ (1-0.95**n_factors)*100:.1f}%）')
    print(f'"最好因子" t≥2.0 的比例: {p_20*100:.1f}% | 平均每次"显著"因子数: {exp_sig:.2f}个')
    print(f'Bonferroni校正(α/{n_factors})后 t临界={t_bonf:.2f} → 假阳性率: {p_bonf*100:.1f}%')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(max_t, bins=60, density=True, color='mediumpurple', alpha=0.8,
            edgecolor='white', lw=0.3)
    for xv, lab, c in [(1.96, '单次检验5%临界 t=1.96', 'crimson'),
                       (2.0, '"可发表"的 t=2.0', 'darkorange'),
                       (t_bonf, f'Bonferroni校正临界 t={t_bonf:.2f}', 'seagreen')]:
        ax.axvline(xv, color=c, ls='--', lw=2, label=lab)
    ax.set_xlabel('20个噪声因子中"最强因子"的 |t|')
    ax.set_ylabel('密度')
    ax.set_title(f'同时检验{n_factors}个无关因子：{p_196*100:.0f}%的概率"发现"一个假因子\n'
                 f'（t≥2.0的假"发现"概率 {p_20*100:.0f}%；Bonferroni校正后假阳性 {p_bonf*100:.1f}%，呼应课16）')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '05_多重检验假阳性.png'), dpi=150)
    plt.close(fig)
    return p_196, p_20, p_bonf, exp_sig


# ==================================================================
# 第5部分 汇总输出
# ==================================================================
def main():
    ret, source = load_monthly_returns()
    print(f'数据来源：{source} | 月度样本 {ret.index[0].date()} ~ {ret.index[-1].date()} 共{len(ret)}个月\n')

    df_ev = part1_bayes_mean(ret, source)
    df_ev.to_csv(os.path.join(OUT, '后验演化明细.csv'), index=False, encoding='utf-8-sig')

    sr_sim, (sr_point, lo, med, hi, p_neg, p_gt1) = part2_sharpe_posterior(ret)

    results = part3_power()

    p_196, p_20, p_bonf, exp_sig = part4_multiple_testing()
    results.append({'场景': f'多重检验: 20个噪声因子', '检出标准': '至少1个|t|>1.96',
                    '50%功效所需(年)': '-', '80%功效所需(年)': '-',
                    '备注': f'假阳性率{p_196*100:.0f}%（Bonferroni校正后{p_bonf*100:.1f}%）'})

    pd.DataFrame(results).to_csv(os.path.join(OUT, 'power_results.csv'),
                                 index=False, encoding='utf-8-sig')

    print('=' * 70)
    print('诚实结论（详见 outputs/power_results.csv）')
    print('=' * 70)
    print(pd.DataFrame(results).to_string(index=False))
    print('\n全部完成，图表与CSV已保存到 outputs/ ← 退出码0')


if __name__ == '__main__':
    main()
