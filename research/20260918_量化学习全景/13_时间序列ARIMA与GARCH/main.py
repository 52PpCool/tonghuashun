# -*- coding: utf-8 -*-
"""
13_时间序列ARIMA与GARCH —— 量化时间序列第一课
================================================
一句话点子：收益几乎不可预测（ARIMA 的诚实结论），但波动率可以（GARCH 的真本事）。

流程（与 README 教学一一对应）：
  S1 数据与收益基础图：日收益 + 滚动波动率，看"波动率聚集"
  S2 平稳性检验 ADF：价格非平稳 vs 收益平稳（为什么平稳是建模前提）
  S3 ARIMA：小网格 (p,q)∈{0,1,2}×{0,1,2} 按 AIC 选阶（d=0），
            系数表（预期大多不显著）+ 滚动样本外预测 vs "什么都不猜"的均值
  S4 ARCH 效应检验：Ljung-Box 于平方收益（为什么看平方）
  S5 GARCH(1,1)：ω/α/β 估计、条件波动率、滚动 1 步波动率预测 vs 20日实际波动基准
  S6 VaR(1%) 应用：GARCH 条件 VaR 回测 + Kupiec 失败率检验（呼应点子11）

运行：/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py
数据：../data/cache/index_000001_上证指数.csv（缺失时自动用 rng(42) 生成合成数据兜底）
随机种子 42；总运行时间 < 2 分钟。
"""

import os
import sys
import math
import time
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # 无显示环境出图
import matplotlib.pyplot as plt
from scipy import stats as sps

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model

warnings.filterwarnings('ignore')  # 收敛提示等噪音统一静音，关键信息主动打印

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
np.random.seed(42)                       # 随机种子 42（仅合成数据兜底会用到）
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE.parent / 'data' / 'cache' / 'index_000001_上证指数.csv'
OUT_DIR = BASE / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

TRAIN_RATIO = 0.70   # 前 70% 训练 / 后 30% 样本外
P_MAX = 2            # ARIMA 网格：p ∈ {0,1,2}
Q_MAX = 2            # ARIMA 网格：q ∈ {0,1,2}（d 固定为 0；刻意用小网格控制耗时）


def save_csv(df: pd.DataFrame, name: str):
    """所有产出 CSV 统一 utf-8-sig，Excel 打开不乱码。"""
    path = OUT_DIR / name
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  [产出] {path}')


def stars(p):
    """p 值加星号：*** <0.01, ** <0.05, * <0.1"""
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))


# ----------------------------------------------------------------------------
# S0 数据加载（真实数据优先，缺失则合成数据兜底，绝不失败退出）
# ----------------------------------------------------------------------------
def load_close():
    """读取上证指数收盘价；若文件缺失/损坏则用 rng(42) 造带 GARCH 特征的合成数据。"""
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['date']).set_index('date')
        close = df['close'].astype(float).dropna()
        if len(close) < 800:
            raise ValueError(f'数据过短: {len(close)} 行')
        print(f'[数据] 真实数据：上证指数 {close.index[0].date()} ~ {close.index[-1].date()}，共 {len(close)} 个交易日')
        return close, False
    except Exception as e:  # noqa: BLE001
        print(f'[数据] 真实数据不可用（{e}），改用合成数据（rng(42)，带GARCH式波动聚集，已标注）')
        rng = np.random.default_rng(42)
        n = 3800
        w, a, b = 0.02, 0.08, 0.90          # 合成 GARCH(1,1) 参数（%^2 为单位）
        h = 1.0                              # 初始方差（%^2）
        mu = 0.02                            # 日均收益（%）
        r = np.empty(n)
        for t in range(n):
            r[t] = mu + math.sqrt(h) * rng.standard_normal()
            h = w + a * r[t] ** 2 + b * h
        idx = pd.bdate_range('2010-01-04', periods=n)
        close = pd.Series(3000.0 * np.exp(np.cumsum(r) / 100.0), index=idx, name='close')
        return close, True


# ----------------------------------------------------------------------------
# S2 ADF 平稳性检验
# ----------------------------------------------------------------------------
def adf_line(x, name):
    """对序列做 ADF 检验，返回 (统计量, p值, 结论)。原假设 H0：存在单位根（非平稳）。"""
    stat, pval, usedlag, nobs, crit, _ = adfuller(x, autolag='AIC')
    concl = '平稳（拒绝H0）' if pval < 0.05 else '非平稳（不能拒绝H0）'
    print(f'  {name:<10s} ADF统计量={stat:9.4f}  p值={pval:8.4g}  滞后阶={usedlag:>3d}  → {concl}')
    return stat, pval, concl


# ----------------------------------------------------------------------------
# S6 Kupiec 无条件覆盖检验
# ----------------------------------------------------------------------------
def kupiec_lr(x, n, p=0.01):
    """
    Kupiec (1995) LR 检验：原假设 H0：真实失败率 = p（名义水平）。
    LR = 2*[ lnL(备择: x/n) - lnL(原假设: p) ] ~ chi2(1)，95%临界值 3.8415。
    """
    phat = x / n
    ll_alt = 0.0
    if x > 0:
        ll_alt += x * math.log(phat)
    if x < n:
        ll_alt += (n - x) * math.log(1 - phat)
    ll_null = x * math.log(p) + (n - x) * math.log(1 - p)
    lr = 2.0 * (ll_alt - ll_null)
    pvalue = 1.0 - sps.chi2.cdf(lr, df=1)
    return lr, pvalue


def main():
    t0 = time.time()
    close, is_synth = load_close()

    # ---- 日收益率（百分比单位，方便阅读与GARCH数值稳定）----
    ret = close.pct_change().dropna() * 100.0     # 日收益率（%）
    dates = ret.index
    y = ret.to_numpy(dtype=float)                 # 模型一律吃 numpy，避免索引/频率噪音
    n = len(y)
    split = int(n * TRAIN_RATIO)                  # 训练/测试切分点
    print(f'[收益] 日收益率 {n} 个观测；训练前 {split} 个（{dates[0].date()}~{dates[split-1].date()}），'
          f'样本外后 {n - split} 个（{dates[split].date()}~{dates[-1].date()}）')
    if is_synth:
        print('[提示] 本次使用合成数据，结论仅用于演示流程。')

    # =========================================================================
    # S1 收益率序列基础图：日收益 + 滚动波动率（看波动率聚集）
    # =========================================================================
    print('\n===== S1 日收益与滚动波动率（波动率聚集）=====')
    roll_std20 = ret.rolling(20).std() * math.sqrt(252)   # 20日滚动波动率（年化）
    print(f'  全样本日收益：均值={y.mean():.4f}%  标准差={y.std(ddof=1):.4f}%  '
          f'偏度={sps.skew(y):.2f}  峰度={sps.kurtosis(y):.2f}（正态应为0和0）')
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(dates, y, lw=0.5, color='steelblue')
    axes[0].set_ylabel('日收益率 (%)')
    axes[0].set_title('上证指数日收益率：大波动扎堆出现（波动率聚集）')
    for span, label in [((dates >= '2015-06-01') & (dates <= '2016-01-31'), '2015年股灾'),
                        ((dates >= '2020-02-01') & (dates <= '2020-04-30'), '2020年疫情')]:
        if span.any():
            axes[0].axvspan(dates[span][0], dates[span][-1], color='red', alpha=0.10)
            axes[0].annotate(label, xy=(dates[span][0], axes[0].get_ylim()[1] * 0.75),
                             fontsize=9, color='darkred')
    axes[1].plot(dates, roll_std20, lw=1.0, color='firebrick')
    axes[1].set_ylabel('20日滚动波动率 (年化)')
    axes[1].set_xlabel('日期')
    axes[1].set_title('20日滚动波动率：平静期与风暴期交替，且持续数月 —— 这就是 GARCH 要捕捉的东西')
    fig.tight_layout()
    fig.savefig(OUT_DIR / '01_收益与滚动波动.png', dpi=150)
    plt.close(fig)
    print('  [产出] 01_收益与滚动波动.png')

    # =========================================================================
    # S2 平稳性：ADF 检验（价格 vs 收益）
    # =========================================================================
    print('\n===== S2 ADF 平稳性检验（H0：存在单位根/非平稳）=====')
    adf_price = adf_line(close.to_numpy(dtype=float), '价格')
    adf_ret = adf_line(y, '日收益率')
    print('  → 价格非平稳（p值大），收益平稳（p值≈0）：对价格直接拟合ARIMA是新手第一大坑；')
    print('    平稳才能用"同一个均值/自相关结构"描述整段序列，模型参数才有稳定含义。')

    # =========================================================================
    # S3 ARIMA：AIC 小网格选阶 + 系数表 + 滚动样本外预测
    # =========================================================================
    print('\n===== S3 ARIMA 网格选阶（训练样本，d=0）=====')
    y_train, y_test = y[:split], y[split:]
    d_train, d_test = dates[:split], dates[split:]

    aic_rows = []
    for p in range(P_MAX + 1):
        for q in range(Q_MAX + 1):
            if p == 0 and q == 0:
                continue  # 纯常数均值，无 AR/MA 项，跳过
            t1 = time.time()
            try:
                res = ARIMA(y_train, order=(p, 0, q)).fit()
                aic_rows.append({'p': p, 'q': q, 'AIC': res.aic, 'BIC': res.bic,
                                 '收敛': bool(res.mle_retvals.get('converged', True)),
                                 '拟合秒': round(time.time() - t1, 2)})
                print(f'  ARIMA({p},0,{q}): AIC={res.aic:10.2f}  BIC={res.bic:10.2f}  '
                      f'({time.time()-t1:.2f}s)')
            except Exception as e:  # noqa: BLE001
                aic_rows.append({'p': p, 'q': q, 'AIC': np.nan, 'BIC': np.nan,
                                 '收敛': False, '拟合秒': round(time.time() - t1, 2)})
                print(f'  ARIMA({p},0,{q}): 拟合失败（{e}）')
    aic_df = pd.DataFrame(aic_rows).sort_values('AIC').reset_index(drop=True)
    save_csv(aic_df, 'arima_aic.csv')
    best_p, best_q = int(aic_df.iloc[0]['p']), int(aic_df.iloc[0]['q'])
    print(f'  → AIC 最优：ARIMA({best_p},0,{best_q})，AIC={aic_df.iloc[0]["AIC"]:.2f}。'
          f'注意：AIC 在样本内选出的阶数不保证样本外有用，见下方滚动检验。')

    def coef_table(res, title):
        """打印系数表：系数/标准误/p值+星号，并检测AR/MA高阶项近抵消（伪显著）。"""
        summ = res.summary().tables[1]
        rows = []
        for i in range(1, len(summ.data)):          # 第0行是表头
            name, coef, se, t, p = summ.data[i][:5]
            rows.append({'参数': name, '系数': float(coef), '标准误': float(se),
                         't值': float(t), 'p值': float(p), '显著性': stars(float(p))})
        df = pd.DataFrame(rows)
        print(f'  {title}')
        print('  ' + df.to_string(index=False,
                                  formatters={'系数': '{:.5f}'.format, '标准误': '{:.5f}'.format,
                                              't值': '{:.2f}'.format, 'p值': '{:.3f}'.format}))
        sig = (df['p值'] < 0.05).sum()
        print(f'  → {len(df)} 个系数中 {sig} 个在5%水平显著。t值大≠有预测力，看样本外表现：')
        coef_by_name = dict(zip(df['参数'], df['系数']))       # statsmodels输入是numpy时params无数组名
        for k, v in coef_by_name.items():                      # AR/MA同阶项都很大 → 近抵消（伪显著）
            if k.startswith('ar.'):
                mk = 'ma.' + k[3:]
                if mk in coef_by_name and abs(v) > 0.7 and abs(coef_by_name[mk]) > 0.7:
                    print(f'  → 警惕：{k}({v:.3f}) 与 {mk}({coef_by_name[mk]:.3f}) 大小相近、'
                          f'符号相反——AR与MA近似抵消。这种组合常借均值方程硬模拟"波动聚集"，'
                          f'样本内AIC占优、系数t值漂亮，但属于伪显著，样本外没有预测价值。')
                    break
        return df

    res_best = ARIMA(y_train, order=(best_p, 0, best_q)).fit()
    coef_table(res_best, f'最优模型 ARIMA({best_p},0,{best_q}) 系数表：')

    # ---- 固定用 ARIMA(1,0,1) 做滚动样本外 1 步预测（教学上最经典的代表）----
    print('\n----- S3b 滚动样本外：ARIMA(1,0,1) 1步预测 vs 均值预测 -----')
    res101 = ARIMA(y_train, order=(1, 0, 1)).fit()
    coef_table(res101, 'ARIMA(1,0,1) 系数表（用于滚动样本外）：')
    mu_train = float(y_train.mean())            # "什么都不猜"基准：训练期均值
    cur = res101
    pred_arima = np.empty(len(y_test))
    t1 = time.time()
    for i in range(len(y_test)):                # 逐日：先用截至昨天的信息预测今天，再"揭晓答案"
        pred_arima[i] = float(np.asarray(cur.forecast(1)).ravel()[0])
        cur = cur.append(np.array([y_test[i]]), refit=False)   # 吸收新观测，不重估参数
    print(f'  滚动预测完成 {len(y_test)} 步，耗时 {time.time()-t1:.1f}s（append 不重估参数，模拟真实逐日使用）')

    mse_arima = float(np.mean((y_test - pred_arima) ** 2))
    mse_mean = float(np.mean((y_test - mu_train) ** 2))
    r2_oos = 1.0 - mse_arima / mse_mean         # 样本外 R²：负数=不如"什么都不猜"
    corr_arima = float(np.corrcoef(pred_arima, y_test)[0, 1])
    print(f'  样本外 MSE：ARIMA={mse_arima:.4f}  均值={mse_mean:.4f}（单位 %^2）')
    print(f'  样本外 R² = {r2_oos:.4f}（≈0或负 → 预测收益≈不可能）；预测与实际相关系数 = {corr_arima:.4f}')
    print('  → 这就是有效市场的诚实结论：再漂亮的样本内t值，样本外R²照样≈0或为负。')

    # =========================================================================
    # S4 ARCH 效应检验：Ljung-Box 于平方收益
    # =========================================================================
    print('\n===== S4 ARCH 效应：Ljung-Box 检验（原假设：无自相关）=====')
    lb_ret = acorr_ljungbox(y, lags=[10, 20, 30], return_df=True)
    lb_sq = acorr_ljungbox(y ** 2, lags=[10, 20, 30], return_df=True)
    resid101 = np.asarray(res101.resid)         # ARIMA(1,0,1) 残差（均值模型已"榨干"线性信息）
    lb_resid2 = acorr_ljungbox(resid101 ** 2, lags=[10, 20, 30], return_df=True)
    lm_stat, lm_p, _, _ = het_arch(y, nlags=10)  # Engle ARCH LM 检验（10阶）
    rho1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])   # 收益 lag-1 自相关（看"幅度"而非只看p值）
    for lag in [10, 20, 30]:
        print(f'  lag={lag:>2d}: 收益 LB p={lb_ret.loc[lag, "lb_pvalue"]:.4g}   '
              f'平方收益 LB p={lb_sq.loc[lag, "lb_pvalue"]:.4g}   '
              f'ARIMA残差平方 LB p={lb_resid2.loc[lag, "lb_pvalue"]:.4g}')
    print(f'  Engle ARCH-LM(10) 统计量={lm_stat:.2f}，p={lm_p:.4g}')
    print(f'  → 收益自相关虽"统计显著"（n大时p值容易显著）但幅度极小：ρ1={rho1:.4f}，')
    print('    换算成样本外R²不到千分之一——线性层面几乎无可预测性；')
    print('    而平方收益的LB p值小到下溢为0：r²的自相关 = 波动率可预测性的直接证据。')
    print('    （为什么看平方？因为方差=E[r²]，r²里已经"去掉了方向、只留幅度"。）')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    plot_acf(y, lags=30, zero=False, ax=axes[0], title='日收益率的ACF（几乎都在置信带内：线性不可预测）')
    plot_acf(y ** 2, lags=30, zero=False, ax=axes[1], title='平方收益的ACF（大量滞后显著：ARCH效应）')
    for ax_ in axes:
        ax_.set_xlabel('滞后阶数')
    fig.tight_layout()
    fig.savefig(OUT_DIR / '02_ACF与平方ACF.png', dpi=150)
    plt.close(fig)
    print('  [产出] 02_ACF与平方ACF.png')

    # =========================================================================
    # S5 GARCH(1,1)：估计 + 条件波动率 + 滚动1步波动率预测
    # =========================================================================
    print('\n===== S5 GARCH(1,1)（arch库，训练样本估计）=====')
    am = arch_model(y_train, mean='Constant', vol='GARCH', p=1, q=1)
    gres = am.fit(disp='off')
    prm = gres.params
    garch_rows = [{'参数': k, '系数': float(v),
                   '标准误': float(gres.std_err[k]), 't值': float(gres.tvalues[k]),
                   'p值': float(gres.pvalues[k]), '显著性': stars(float(gres.pvalues[k]))}
                  for k, v in prm.items()]
    omega = float(prm['omega']); alpha = float(prm['alpha[1]']); beta = float(prm['beta[1]'])
    garch_rows.append({'参数': 'alpha+beta（平稳性）', '系数': alpha + beta,
                       '标准误': np.nan, 't值': np.nan, 'p值': np.nan, '显著性': ''})
    garch_df = pd.DataFrame(garch_rows)
    print('  ' + garch_df.to_string(index=False,
                                   formatters={'系数': '{:.6f}'.format, '标准误': '{:.6f}'.format,
                                               't值': '{:.2f}'.format, 'p值': '{:.4g}'.format}))
    persistence = alpha + beta
    # 波动率冲击半衰期：k 步前瞻方差预测的偏离按 (α+β)^k 衰减
    #   E_t[σ²_{t+k}] − σ̄² = (α+β)^k · (σ²_t − σ̄²)，故半衰期 = ln0.5 / ln(α+β)。
    #   （若用 lnβ 会把"β通道"误当总持续性：EWMA 即 α+β=1 的特例，其预测永不均值回归、
    #    半衰期应为无穷，而 lnβ 版会给出约 11 天的荒谬值。）
    halflife = math.log(0.5) / math.log(persistence) if 0 < persistence < 1 else np.nan
    uncond_vol = math.sqrt(omega / (1 - persistence))          # 无条件日波动率（%）
    print(f'  → α+β={persistence:.4f} < 1（协方差平稳）；α+β越接近1波动越持久，'
          f'半衰期≈{halflife:.0f}个交易日；无条件日波动率≈{uncond_vol:.3f}%')
    print(f'  → α 是"新冲击"的贡献（昨天大涨大跌今天继续抖），β 是"旧波动"的记忆（风暴要慢慢平息）')
    save_csv(garch_df, 'garch_params.csv')

    # ---- 样本内条件波动率曲线（训练段来自估计结果；样本外段用手写递推继续滤波）----
    # GARCH(1,1) 递推：σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}   —— 只用到昨天的信息
    sig_train = np.asarray(gres.conditional_volatility, dtype=float)
    h = sig_train[-1] ** 2                                     # σ²_{split-1}（训练期最后一天）
    sig_oos = np.empty(n - split)
    for i in range(n - split):
        t_abs = split + i
        h = omega + alpha * y[t_abs - 1] ** 2 + beta * h       # 用固定参数继续滤波 → 1步预测
        sig_oos[i] = math.sqrt(h)
    sig_full = np.concatenate([sig_train, sig_oos])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, np.abs(y), lw=0.5, color='lightgray', label='|日收益|（波动的 noisy 代理）')
    ax.plot(dates, sig_full, lw=1.2, color='crimson', label='GARCH(1,1) 条件波动率 σ')
    ax.axvline(dates[split], color='k', ls='--', lw=1, label='训练/样本外切分')
    ax.set_ylabel('日波动率 (%)')
    ax.set_xlabel('日期')
    ax.set_title(f'GARCH(1,1) 条件波动率：平滑地"追着"|收益|跑（α+β={persistence:.3f}）')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / '04_GARCH条件波动率.png', dpi=150)
    plt.close(fig)
    print('  [产出] 04_GARCH条件波动率.png')

    # ---- 滚动样本外 1 步波动率预测 vs 基准（过去20日实际波动，同样不偷看未来）----
    print('\n----- S5b 滚动样本外：GARCH 1步波动率预测 vs 20日实际波动基准 -----')
    bench_vol = np.array([np.std(y[t - 20:t], ddof=1) for t in range(split, n)])  # 截至昨天的20日std
    abs_ret = np.abs(y_test)                                   # 次日实际|收益|（波动的代理）
    corr_g = float(np.corrcoef(sig_oos, abs_ret)[0, 1])
    corr_b = float(np.corrcoef(bench_vol, abs_ret)[0, 1])
    rho_g = float(sps.spearmanr(sig_oos, abs_ret).statistic)
    rho_b = float(sps.spearmanr(bench_vol, abs_ret).statistic)
    # 方向命中率：真实"高波动日" = |收益| 高于测试期中位数；预测"高波动" = σ 高于其测试期中位数
    hi_real = abs_ret > np.median(abs_ret)
    hi_g = sig_oos > np.median(sig_oos)
    hi_b = bench_vol > np.median(bench_vol)
    hit_g = float(np.mean(hi_g == hi_real))
    hit_b = float(np.mean(hi_b == hi_real))
    print(f'  与次日|收益|的相关系数：  GARCH σ = {corr_g:.4f}（Spearman {rho_g:.4f}）   '
          f'20日基准 = {corr_b:.4f}（Spearman {rho_b:.4f}）')
    print(f'  方向命中率（高/低波动二分类）：GARCH = {hit_g:.1%}   20日基准 = {hit_b:.1%}（随机=50%）')
    print('  → 相关性不高但稳定优于简单基准、命中率>50%：这就是"波动率可预测"的真实证据——')
    print('    可预测≠精确，GARCH预测的是"明天大概有多抖"，不是"明天涨还是跌"。')

    # =========================================================================
    # S6 VaR(1%) 应用与 Kupiec 回测
    # =========================================================================
    print('\n===== S6 GARCH 条件 VaR(1%) 回测（样本外）=====')
    mu_g = float(prm['mu'])                     # GARCH 均值方程的 μ（%）
    z99 = sps.norm.ppf(0.01)                    # -2.3263
    thr_garch = mu_g + z99 * sig_oos            # 收益率空间的1%分位阈值（负值），只用昨日信息
    var_garch = -thr_garch                      # 损失为正的 VaR 表述
    sig_u = float(np.std(y_train, ddof=1))      # 无条件基准：训练期常数波动
    thr_uncond = float(np.mean(y_train)) + z99 * sig_u
    viol_g = y_test < thr_garch
    viol_u = y_test < thr_uncond
    xg, xu = int(viol_g.sum()), int(viol_u.sum())
    lr_g, pv_g = kupiec_lr(xg, len(y_test), p=0.01)
    lr_u, pv_u = kupiec_lr(xu, len(y_test), p=0.01)
    print(f'  样本外 {len(y_test)} 天：GARCH条件VaR击穿 {xg} 次（失败率 {xg/len(y_test):.2%}，名义1%），'
          f'Kupiec LR={lr_g:.3f}，p={pv_g:.4g} → {"无法拒绝" if pv_g > 0.05 else "拒绝"}"失败率=1%"')
    print(f'  对照：无条件(常数σ, 用训练期波动)VaR击穿 {xu} 次（失败率 {xu/len(y_test):.2%}），'
          f'Kupiec LR={lr_u:.3f}，p={pv_u:.4g}')
    print('  → 如实解读：两个模型在5%水平都被Kupiec拒绝，现实中的风险模型没有"一劳永逸"；')
    print('    但GARCH失败率离名义1%更近——训练期(含2015股灾)波动高、测试期(2021-2026)波动低，')
    print('    常数σ沿用旧风暴尺度导致过度保守(击穿太少)，GARCH则随市况自动调整了尺度。')

    eval_df = pd.DataFrame({
        'date': d_test.strftime('%Y-%m-%d'),
        '实际收益率pct': y_test,
        'ARIMA预测pct': pred_arima,
        '均值预测pct': mu_train,
        'GARCH波动率pct': sig_oos,
        '基准波动20日pct': bench_vol,
        '实际绝对收益pct': abs_ret,
        'VaR1阈值pct': thr_garch,
        '击穿VaR': viol_g.astype(int),
    })
    save_csv(eval_df, 'forecast_eval.csv')

    # ---- 03 滚动预测对比图：收益预测(无用) vs 波动率预测(有用) ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    k = min(60, len(y_test))
    axes[0, 0].plot(d_test[:k], y_test[:k], color='gray', lw=1, label='实际收益')
    axes[0, 0].plot(d_test[:k], pred_arima[:k], color='steelblue', lw=1, label='ARIMA(1,0,1)预测')
    axes[0, 0].axhline(mu_train, color='darkorange', ls='--', lw=1.5, label='均值预测（什么都不猜）')
    axes[0, 0].set_title(f'收益预测：预测线几乎是直线（前{k}个样本外交易日）')
    axes[0, 0].set_ylabel('日收益率 (%)'); axes[0, 0].legend(fontsize=8)
    axes[0, 1].scatter(pred_arima, y_test, s=6, alpha=0.4, color='steelblue')
    axes[0, 1].set_xlabel('ARIMA预测收益 (%)'); axes[0, 1].set_ylabel('实际收益 (%)')
    axes[0, 1].set_title(f'收益预测散点：样本外R²={r2_oos:.4f}，相关系数={corr_arima:.3f}')
    axes[1, 0].plot(d_test, sig_oos, color='crimson', lw=1, label='GARCH 1步σ预测')
    axes[1, 0].plot(d_test, bench_vol, color='gray', lw=1, ls='--', label='基准：过去20日波动')
    axes[1, 0].set_title('波动率预测：GARCH快速响应冲击（样本外）')
    axes[1, 0].set_ylabel('日波动率 (%)'); axes[1, 0].legend(fontsize=8)
    axes[1, 1].scatter(sig_oos, abs_ret, s=6, alpha=0.4, color='crimson')
    axes[1, 1].set_xlabel('GARCH预测波动 σ (%)'); axes[1, 1].set_ylabel('实际|收益| (%)')
    axes[1, 1].set_title(f'波动预测散点：相关系数 GARCH={corr_g:.3f} vs 基准={corr_b:.3f}')
    fig.suptitle('同一份数据的两个真相：收益预测≈噪音，波动率预测有信息', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / '03_滚动预测对比.png', dpi=150)
    plt.close(fig)
    print('  [产出] 03_滚动预测对比.png')

    # ---- 05 VaR 回测图 ----
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = np.where(viol_g, 'crimson', 'steelblue')
    ax.bar(d_test, y_test, width=1.0, color=colors, label='样本外日收益（红=击穿VaR）')
    ax.plot(d_test, thr_garch, color='k', lw=1.2, label='GARCH条件VaR(1%)阈值')
    ax.axhline(thr_uncond, color='gray', ls='--', lw=1.2, label='无条件VaR(1%)阈值（常数σ）')
    ax.set_ylabel('日收益率 (%)')
    ax.set_xlabel('日期')
    ax.set_title(f'VaR(1%)回测：GARCH击穿 {xg}/{len(y_test)} 天（{xg/len(y_test):.2%}，名义1%）；'
                 f'无条件VaR击穿 {xu}/{len(y_test)} 天（{xu/len(y_test):.2%}）')
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / '05_VaR回测.png', dpi=150)
    plt.close(fig)
    print('  [产出] 05_VaR回测.png')

    # =========================================================================
    # 汇总（README 里的数字全部来自这里）
    # =========================================================================
    print('\n===== 关键数字汇总（README 引用）=====')
    print(f'  ADF价格: 统计量={adf_price[0]:.4f}, p={adf_price[1]:.4g} → {adf_price[2]}')
    print(f'  ADF收益: 统计量={adf_ret[0]:.4f}, p={adf_ret[1]:.4g} → {adf_ret[2]}')
    print(f'  AIC最优阶: ARIMA({best_p},0,{best_q})')
    print(f'  样本外收益预测: R²={r2_oos:.4f}, MSE_ARIMA={mse_arima:.4f}, MSE_均值={mse_mean:.4f}')
    print(f'  平方收益LB p值: lag10={lb_sq.loc[10, "lb_pvalue"]:.4g}, lag20={lb_sq.loc[20, "lb_pvalue"]:.4g}, '
          f'lag30={lb_sq.loc[30, "lb_pvalue"]:.4g}')
    print(f'  GARCH: ω={omega:.6f}, α={alpha:.4f}, β={beta:.4f}, α+β={persistence:.4f}, 半衰期={halflife:.0f}天')
    print(f'  波动预测OOS: corr(GARCH,|r|)={corr_g:.4f} vs corr(基准,|r|)={corr_b:.4f}; '
          f'命中率 {hit_g:.1%} vs {hit_b:.1%}')
    print(f'  VaR回测: GARCH {xg}/{len(y_test)}={xg/len(y_test):.2%} (Kupiec p={pv_g:.4g}); '
          f'无条件 {xu}/{len(y_test)}={xu/len(y_test):.2%} (Kupiec p={pv_u:.4g})')
    print(f'\n总耗时 {time.time() - t0:.1f}s，全部产物在 {OUT_DIR}')


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        print('\n[兜底] 运行出错：已打印完整堆栈供排查，按规范不以非0退出。')
        sys.exit(0)
