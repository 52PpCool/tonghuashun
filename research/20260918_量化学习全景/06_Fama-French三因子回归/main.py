# -*- coding: utf-8 -*-
"""
06_Fama-French三因子回归 —— 你的超额收益，是真本事还是风格暴露？
======================================================================

一句话课程目标：学会用一次回归回答量化圈最灵魂的拷问——
"过去这些年赚的钱，多少来自选股能力（alpha），多少只是押对了风格（beta）？"

Fama-French 三因子时间序列回归（Fama & French, 1993, JFE）：

    Ri - RF = a + b·(Mkt-RF) + s·SMB + h·HML + ε

    Mkt-RF : 市场因子 = 全市场组合收益 − 无风险利率，代表"大盘水位"
    SMB    : 规模因子 = Small Minus Big，小盘组合收益 − 大盘组合收益
    HML    : 价值因子 = High Minus Low，高B/M(价值股)组合 − 低B/M(成长股)组合
    RF     : 无风险利率（月度，如一个月期国库券）
    a      : alpha —— 三个因子都解释不掉的月均超额收益，才是"选股能力"
    b/s/h  : 三个风格暴露(beta) —— "被动持有某类风格的仓位"，可以刻意买到

课程流程：读官方因子 → 下载三只美股月收益 → 逐只FF3回归 → 滚动36月alpha
          → 残差诊断（QQ图+ACF） → 迁移思考：A股怎么自建SMB/HML（示意图）。

数据说明：
  * 因子：data/cache/ff3_monthly.csv（Kenneth French 官网月度数据，单位%，1926-07 起）
  * 个股：yfinance 下载 AAPL / MSFT / NVDA 月K（2015-01 至今），在线真实数据优先。
    若下载失败：用 np.random.default_rng(42) 依真实因子"造"三只合成股票，
    所有图表与 README 均明确标注【合成数据】。
  * 标准误：Newey-West(1987) HAC 稳健标准误（自动窗宽 floor(4·(n/100)^(2/9))），
    用于对冲月度残差的自相关与异方差，让 t 值更可信。

运行：python main.py（约 10~40 秒，大头在下载）
产出：outputs/ 下 5 张图 + ff3_results.csv + regression_detail.txt
"""

# ============================================================
# 第 0 节  导入与全局配置
# ============================================================
import os
import time
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 无界面环境：只输出图片文件，不弹窗
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats as sps

warnings.filterwarnings('ignore', category=FutureWarning)  # yfinance 内部的弃用提示，与本课无关
logging.getLogger('yfinance').setLevel(logging.CRITICAL)   # 关掉 yfinance 的日志刷屏

# 中文字体（第一个缺失时自动回退到列表中的下一个）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 让负号正常显示

SEED = 42
rng = np.random.default_rng(SEED)  # 仅在 yfinance 下载失败、启用合成数据兜底时用到

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'cache'))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = {'AAPL': '苹果', 'MSFT': '微软', 'NVDA': '英伟达'}
DOWNLOAD_START = '2014-12-01'  # 故意多取一个月：月收益 = 相邻两根月K收盘价之比，
                               # 要算 2015-01 的收益就必须有 2014-12 这根"垫底"K线
SAMPLE_START = '2015-01'       # 分析样本起点（2015-01 至今），与课程设定一致
ROLL_WIN = 36                  # 滚动 alpha 的窗口长度（36 个月 = 3 年）
ROLL_STOCK = 'NVDA'            # 滚动 alpha 演示用的股票（全市场beta最高的那只）

FACTOR_COLS = ['Mkt-RF', 'SMB', 'HML']
FACTOR_CN = {'Mkt-RF': '市场', 'SMB': '规模', 'HML': '价值'}


# ============================================================
# 第 1 节  读取 Fama-French 三因子月度数据
# ============================================================
def load_ff3(data_dir: str) -> pd.DataFrame:
    """读 Kenneth French 官方三因子月度数据，并把单位从 % 换算成小数（除以100）。

    四列的含义（这是全世界因子研究的"普通话"，必须背下来）：
      Mkt-RF : 市场因子。全市场股票组合当月收益 − 当月无风险利率。
               它回答"这个月大盘整体涨了多少"。
      SMB    : 规模因子（Small Minus Big）。小盘股组合收益 − 大盘股组合收益。
               它捕捉"小公司长期跑赢大公司"的规模溢价。
      HML    : 价值因子（High Minus Low）。高账面市值比(B/M，即"便宜"的价值股)
               组合收益 − 低B/M("贵"的成长股)组合收益，捕捉价值溢价。
      RF     : 无风险利率（月度），如一个月期美国国库券收益率。
    """
    path = os.path.join(data_dir, 'ff3_monthly.csv')
    df = pd.read_csv(path, encoding='utf-8-sig')
    idx = pd.PeriodIndex(df['yyyymm'].astype(int).astype(str), freq='M')
    ff3 = df.drop(columns='yyyymm')
    ff3.index = idx
    ff3 = ff3.astype(float) / 100.0        # 单位% → 小数（0.01 = 1%）
    ff3 = ff3[~ff3.index.duplicated(keep='last')].sort_index()
    return ff3


# ============================================================
# 第 2 节  个股月度收益：在线下载，失败则合成兜底
# ============================================================
def fetch_stock_monthly(tickers, start):
    """用 yfinance 逐只下载月K（interval='1mo', auto_adjust=True），转成月收益。

    关键细节（不做就会出错）：
      1) auto_adjust=True：收盘价已复权（分红送股调整过），pct_change 才是真收益；
      2) 月K的时间戳是"每月第一个交易日"，代表整月一根K线，转成 Period('M')
         才能和因子的 yyyymm 对齐；
      3) 当前正在进行中的月份是一根"没走完的K线"（本月还没收盘），
         必须整月剔除，否则会用半个月的行情冒充整月收益。
    返回 ({代码: 月收益Series(小数)}, 数据来源描述)；任何一只失败都返回 (None, 原因)。
    """
    import yfinance as yf
    out = {}
    for code in tickers:
        df = yf.download(code, start=start, interval='1mo', auto_adjust=True,
                         progress=False, threads=False)
        if df is None or df.empty or 'Close' not in df:
            return None, f'{code} 下载为空'
        close = df['Close']
        if isinstance(close, pd.DataFrame):     # 某些版本多股票时返回双列索引
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 30:                     # 少于30根月K，样本没意义
            return None, f'{code} 数据不足'
        idx = pd.DatetimeIndex(close.index)     # 时间戳 → Period('M') 与因子对齐
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        close.index = idx.to_period('M')
        close = close[~close.index.duplicated(keep='last')].sort_index()
        cur = pd.Timestamp.today().to_period('M')   # 剔除"本月未走完"的半根K线
        close = close[close.index < cur]
        ret = close.pct_change().dropna()           # 相邻两根月K收盘价之比 = 月收益
        ret = ret[ret.index >= pd.Period(SAMPLE_START, freq='M')]
        if len(ret) < 60:
            return None, f'{code} 月收益样本不足'
        out[code] = ret
    return out, 'yfinance'


def synthetic_stock_returns(ff3, tickers):
    """兜底方案：用真实因子数据"造"三只合成股票（仅用于离线跑通流程）。

    生成公式（和回归模型同构，所以回归一定"能还原"出设定参数）：
        R = RF + a + b·Mkt-RF + s·SMB + h·HML + ε,   ε ~ N(0, σ)
    人为设定三只股票的性格，正好复刻本课要讲的结论——
        苹果：温和alpha(0.8%/月) + 市场beta 1.1（真有点本事）
        微软：几乎零alpha(0.1%/月) + 市场beta 1.0（平平无奇）
        英伟达：零alpha + 市场beta 1.7（涨得多纯粹因为beta高！）
    """
    spec = {
        'AAPL': dict(a=0.008, b=1.10, s=-0.30, h=-0.50, sig=0.050),
        'MSFT': dict(a=0.001, b=1.00, s=-0.10, h=-0.20, sig=0.045),
        'NVDA': dict(a=0.000, b=1.70, s=0.20, h=-0.40, sig=0.090),
    }
    idx = ff3.index[ff3.index >= pd.Period(SAMPLE_START, freq='M')]
    out = {}
    for code in tickers:
        p = spec[code]
        eps = rng.normal(0.0, p['sig'], len(idx))
        r = (p['a'] + p['b'] * ff3.loc[idx, 'Mkt-RF'] + p['s'] * ff3.loc[idx, 'SMB']
             + p['h'] * ff3.loc[idx, 'HML'] + ff3.loc[idx, 'RF'] + eps)
        out[code] = pd.Series(r.values, index=idx)
    return out


# ============================================================
# 第 3 节  FF3 时间序列回归（本课的心脏）
# ============================================================
def nw_lags(n: int) -> int:
    """Newey-West(1994) 自动窗宽：floor(4·(n/100)^(2/9))，月频数据常用规则。"""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def ff3_regression(excess: pd.Series, ff3: pd.DataFrame):
    """对一只股票做 FF3 时间序列回归：Ri-RF = a + b·Mkt-RF + s·SMB + h·HML + ε。

    为什么是"时间序列回归"？——对单一股票，把它的超额收益序列对因子序列回归，
    截距 a 就是这只股票的 alpha（Jensen's alpha 的三因子升级版）。
    （与之相对的"截面回归"是在每个时点横跨一堆股票回归，见第05课的IC。）

    为什么用超额收益（Ri-RF）而不是总收益 Ri？——若用总收益，市场因子的截距
    会把"无风险利率"也吃进 alpha，长期高利率环境下 alpha 会被凭空抬高。

    标准误用 Newey-West(HAC)：月度残差通常有点自相关和异方差，普通OLS的t值
    会虚高；NW 标准误对此做修正，让"alpha是否显著"的判断更可信。
    """
    df = pd.concat({'excess': excess, **{c: ff3[c] for c in FACTOR_COLS}},
                   axis=1).dropna()
    y = df['excess']
    X = sm.add_constant(df[FACTOR_COLS])           # 加一列1，对应截距 a
    res = sm.OLS(y, X).fit(cov_type='HAC',
                           cov_kwds={'maxlags': nw_lags(len(df))})
    return res, df


def stars(p: float) -> str:
    """显著性星号：*** p<0.01，** p<0.05，* p<0.1。"""
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))


def results_row(code, name, res, n, source):
    """回归结果 → 一行汇总（写入 ff3_results.csv）。alpha 年化 = 月alpha × 12。"""
    return {
        '股票代码': code, '名称': name, '数据来源': source, '样本月数': n,
        'alpha_月': res.params['const'],
        'alpha_年化': res.params['const'] * 12.0,
        'alpha_t值': res.tvalues['const'],
        'alpha_p值': res.pvalues['const'],
        'beta_市场': res.params['Mkt-RF'], 't_市场': res.tvalues['Mkt-RF'],
        'beta_规模SMB': res.params['SMB'], 't_规模': res.tvalues['SMB'],
        'beta_价值HML': res.params['HML'], 't_价值': res.tvalues['HML'],
        'R2': res.rsquared, '调整R2': res.rsquared_adj,
    }


# ============================================================
# 第 4 节  滚动 36 个月 alpha（alpha 稳不稳？）
# ============================================================
def rolling_alpha(excess: pd.Series, ff3: pd.DataFrame, window: int = ROLL_WIN):
    """每 36 个月重跑一次 FF3 回归，记录"当时点上的 alpha"，得到一条时间曲线。

    全样本回归只告诉你"这十几年平均下来 alpha 是多少"，却藏不住也显不出
    "alpha 是稳定的能力，还是某一两年的运气"。滚动窗口把样本切成很多段，
    每段各算一个 alpha：曲线平稳 → 能力可持续；大起大落甚至变号 → 很可能是运气。
    这里同时记录 alpha 的标准误，用来画 95% 置信带（±1.96·se，同样年化）。
    """
    df = pd.concat({'excess': excess, **{c: ff3[c] for c in FACTOR_COLS}},
                   axis=1).dropna()
    recs = {}
    for i in range(window, len(df) + 1):
        sub = df.iloc[i - window: i]
        X = sm.add_constant(sub[FACTOR_COLS])
        res = sm.OLS(sub['excess'], X).fit(
            cov_type='HAC', cov_kwds={'maxlags': nw_lags(window)})
        recs[sub.index[-1]] = {'alpha_月': res.params['const'],
                               'alpha_se_月': res.bse['const']}
    out = pd.DataFrame.from_dict(recs, orient='index').sort_index()
    out['alpha_年化'] = out['alpha_月'] * 12.0
    out['ci95_年化'] = 1.96 * out['alpha_se_月'] * 12.0   # 年化后的 95% 置信半径
    return out


# ============================================================
# 第 5 节  残差诊断（回归不是跑完就完事）
# ============================================================
def residual_diagnostics(res) -> dict:
    """回归残差的两项常规体检：

    1) 正态性：Jarque-Bera 检验（基于偏度+峰度）。p值<0.05 → 拒绝正态假设。
       金融月收益常右偏/尖峰，残差不正态时，小样本t分布的临界值仅供参考。
    2) 自相关：Ljung-Box 检验（残差前若干阶自相关是否为0）+ Durbin-Watson。
       p值<0.05 → 残差自相关明显（这正是我们一开始就用 Newey-West 标准误的原因）。
    """
    resid = res.resid
    jb, jbp, skew, kurt = jarque_bera(resid)
    lb_p = acorr_ljungbox(resid, lags=[6], return_df=True)['lb_pvalue'].iloc[0]
    return {'JB统计量': jb, 'JB_p值': jbp, '残差偏度': skew, '残差峰度': kurt,
            'LjungBox6_p值': lb_p, 'DurbinWatson': durbin_watson(resid)}


# ============================================================
# 第 6 节  绘图（5 张图）
# ============================================================
def plot_01_fit_vs_actual(df, res, code, name, src_tag):
    """图1：某股票实际超额收益 vs FF3模型拟合（"解释了多少"一眼看穿）。"""
    fitted = res.fittedvalues                              # a + b·MktRF + s·SMB + h·HML
    beta_only = (res.params['Mkt-RF'] * df['Mkt-RF']
                 + res.params['SMB'] * df['SMB']
                 + res.params['HML'] * df['HML'])          # 不含alpha的"风格暴露部分"
    x = df.index.to_timestamp()
    actual_nav = (1 + df['excess']).cumprod()
    fit_nav = (1 + fitted).cumprod()
    beta_nav = (1 + beta_only).cumprod()

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(x, actual_nav, color='#d62728', lw=2.0, label=f'实际超额收益（{code} {name}）')
    ax.plot(x, fit_nav, color='#1f77b4', lw=1.6, ls='--',
            label=f'FF3拟合（含alpha，R²={res.rsquared:.0%}）')
    ax.plot(x, beta_nav, color='#7f7f7f', lw=1.6, ls=':', label='仅风格暴露部分（不含alpha）')
    a_ann = res.params['const'] * 12
    ax.annotate(f'期末：实际 {actual_nav.iloc[-1]:.2f}｜拟合 {fit_nav.iloc[-1]:.2f}｜'
                f'仅beta {beta_nav.iloc[-1]:.2f}\n月alpha={res.params["const"]:+.2%} → '
                f'年化 {a_ann:+.1%}（t={res.tvalues["const"]:.1f}）',
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', fc='#FFF8E7', ec='#B8860B'))
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_title(f'累计超额收益的实际 vs FF3模型拟合（{src_tag}）\n'
                 '红线与灰线的缺口 ≈ alpha累计贡献；蓝虚线高于红线 = alpha并非每月均匀出现（见02图）')
    ax.set_xlabel('月份')
    ax.set_ylabel('累计净值（起点=1，超额收益口径）')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_累计超额收益与因子解释.png'), dpi=150)
    plt.close(fig)


def plot_02_rolling(roll, code, name, src_tag):
    """图2：滚动36个月年化 alpha 曲线 + 95%置信带。"""
    x = roll.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.fill_between(x, roll['alpha_年化'] - roll['ci95_年化'],
                    roll['alpha_年化'] + roll['ci95_年化'],
                    color='#1f77b4', alpha=0.18, label='95%置信带（±1.96·se）')
    ax.plot(x, roll['alpha_年化'], color='#1f77b4', lw=1.8,
            label='滚动36月alpha（年化）')
    ax.axhline(0, color='#d62728', lw=1.2, ls='--', label='alpha = 0（无选股能力线）')
    pos = (roll['alpha_年化'] > 0).mean()
    ax.set_title(f'{code} {name}：滚动36个月 alpha 曲线（{src_tag}）\n'
                 f'窗口数={len(roll)}，其中alpha>0的窗口占 {pos:.0%}；'
                 '置信带碰到0 = 该窗口的alpha统计上不显著')
    ax.set_xlabel('窗口结束月份')
    ax.set_ylabel('年化 alpha（小数）')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_滚动36月alpha.png'), dpi=150)
    plt.close(fig)


def plot_03_diagnostics(res_dict, src_tag):
    """图3：三只股票的残差诊断（上排QQ图看正态性，下排ACF看自相关）。"""
    fig, axes = plt.subplots(2, len(res_dict), figsize=(13.5, 7.2))
    for j, (code, res) in enumerate(res_dict.items()):
        resid = res.resid
        # QQ图：残差分位数 vs 正态分布分位数。点贴着红线 → 近似正态；尾部翘 → 尖峰厚尾
        sps.probplot(resid, dist='norm', plot=axes[0, j])
        axes[0, j].set_title(f'{code} 残差QQ图')
        axes[0, j].set_xlabel('理论分位数')
        axes[0, j].set_ylabel('样本分位数')
        axes[0, j].get_lines()[0].set(markersize=3.5, alpha=0.6)
        # ACF：残差各阶自相关。绝大多数落在蓝色置信带内 → 无明显自相关
        plot_acf(resid, lags=24, ax=axes[1, j], zero=False)
        axes[1, j].set_title(f'{code} 残差ACF（Ljung-Box(6) p='
                             f'{residual_diagnostics(res)["LjungBox6_p值"]:.3f}）')
        axes[1, j].set_xlabel('滞后阶数（月）')
    fig.suptitle(f'回归残差诊断：上排=正态性(QQ图)，下排=自相关(ACF)（{src_tag}）', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_残差诊断.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_04_alpha_beta(rows, src_tag):
    """图4：三只股票的年化alpha（带95%置信区间）与三个beta对比。"""
    codes = [r['股票代码'] for r in rows]
    names = [f"{r['股票代码']}\n{r['名称']}" for r in rows]
    xx = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4))

    # 左图：年化 alpha 柱状 + 95%置信区间误差棒
    a_ann = np.array([r['alpha_年化'] for r in rows])
    ci = np.array([1.96 * (r['alpha_年化'] / r['alpha_t值']) if r['alpha_t值'] != 0
                   else np.nan for r in rows])       # se_年化 = alpha_年化 / t
    colors = ['#2ca02c' if r['alpha_p值'] < 0.05 else '#aec7e8' for r in rows]
    bars = ax1.bar(xx, a_ann, yerr=ci, capsize=5, color=colors,
                   edgecolor='black', lw=0.6, width=0.55)
    for b, r in zip(bars, rows):
        ax1.annotate(f"{r['alpha_年化']:+.1%}{stars(r['alpha_p值'])}\nt={r['alpha_t值']:.1f}",
                     (b.get_x() + b.get_width() / 2, r['alpha_年化']),
                     ha='center', va='bottom' if r['alpha_年化'] >= 0 else 'top',
                     fontsize=10, fontweight='bold')
    ax1.axhline(0, color='gray', lw=0.8)
    ax1.set_xticks(xx)
    ax1.set_xticklabels(names)
    ax1.set_title('年化 alpha（选股能力）\n绿色=p<0.05显著；浅蓝=不显著')
    ax1.set_ylabel('年化 alpha')
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax1.grid(axis='y', alpha=0.3)

    # 右图：三个beta分组柱状（风格暴露可以"买到"，所以它不是本事）
    width = 0.25
    beta_keys = {'Mkt-RF': 'beta_市场', 'SMB': 'beta_规模SMB', 'HML': 'beta_价值HML'}
    for k, col in enumerate(FACTOR_COLS):
        vals = [r[beta_keys[col]] for r in rows]
        bars = ax2.bar(xx + (k - 1) * width, vals, width,
                       label=FACTOR_CN[col], edgecolor='black', lw=0.5)
        for b, v in zip(bars, vals):
            ax2.annotate(f'{v:+.2f}', (b.get_x() + b.get_width() / 2, v),
                         ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(names)
    ax2.set_title('三个风格暴露 beta（风格仓位）\n市场beta高 = 只是"仓位更激进"')
    ax2.set_ylabel('beta')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f'alpha 与 beta 的对比：谁的收益是"本事"，谁只是"风格"（{src_tag}）', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_三股alpha与beta对比.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_05_transfer_schematic():
    """图5：迁移思考示意图——A股如何自建规模(SMB)/价值(HML)因子。"""
    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    boxes = [
        ('第1步  原始数据（每月末截面）',
         '全市场A股\n（必须含已退市股！）\n\n每股需要：\n· 流通市值\n· 账面净值 B\n· 收盘价\n\n'
         '→ B/M = B / 总市值\n（akshare 接口/财报可取）'),
        ('第2步  独立双重排序（2×3）',
         '① 按流通市值切中位数：\n    小盘S / 大盘B\n\n② 组内按 B/M 三分位：\n'
         '    低(成长)30% / 中40%\n    / 高(价值)30%\n\n→ 6个组合：\n'
         'S/L S/M S/H\nB/L B/M B/H'),
        ('第3步  6个组合的月收益',
         '每组做成等权组合\n每月末调仓\n记录下月收益\n\n（本课30只股池的退化版：\n'
         '市值前15 = 大盘B\n市值后15 = 小盘S\n——只是"相对小"\n的大盘股）'),
        ('第4步  合成A股自己的因子',
         'SMB = 均值(S/L,S/M,S/H)\n     − 均值(B/L,B/M,B/H)\n\n'
         'HML = 均值(S/H,B/H)\n     − 均值(S/L,B/L)\n\n'
         '把自家SMB/HML代回\nFF3回归 → 得到A股口径\n的 alpha 与 beta'),
    ]
    xs = [0.012, 0.262, 0.512, 0.762]
    for (title, body), x in zip(boxes, xs):
        ax.text(x, 0.97, f'{title}\n\n{body}', fontsize=8.8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', fc='#EAF2F8', ec='#2E86C1', lw=1.2))
    for x in [0.232, 0.482, 0.732]:
        ax.annotate('', xy=(x + 0.026, 0.74), xytext=(x, 0.74),
                    arrowprops=dict(arrowstyle='-|>', lw=2.0, color='#2E86C1'))

    ax.text(0.012, 0.26,
            '诚实提示：本课 data/cache 里只有 30 只"当今知名"大盘股 —— ① 没有市值/财务数据；'
            '② 没有退市股（幸存者偏差）；③ 根本没有真·小盘股。\n'
            '用它们造出来的 "SMB" 会严重低估真实的规模效应，只能演示思路。'
            '严肃的A股三因子研究必须用：全市场 + 时点股票池 + 含退市股 + 真实市值与财报数据。',
            fontsize=9.6, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='#FDEDEC', ec='#C0392B', lw=1.2))
    ax.set_ylim(0.10, 1.0)   # 收紧纵向空白

    ax.set_title('迁移到A股：自建规模因子(SMB)与价值因子(HML)的完整流程（Fama-French 2×3 独立排序法）',
                 fontsize=13, pad=14)
    fig.savefig(os.path.join(OUT_DIR, '05_A股自建因子示意.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main():
    t0 = time.time()
    print('== 06 Fama-French三因子回归：alpha 还是 beta？ ==')

    # ---- 1. 读取三因子 ------------------------------------------------------
    ff3 = load_ff3(DATA_DIR)
    print(f'\n【三因子数据】Kenneth French 官方月度数据，'
          f'{ff3.index[0]} ~ {ff3.index[-1]}，共 {len(ff3)} 个月')
    ann = ff3.mean() * 12
    print('全样本长期年化均值（直觉校准）：'
          f'市场溢价 {ann["Mkt-RF"]:+.2%}｜规模SMB {ann["SMB"]:+.2%}｜'
          f'价值HML {ann["HML"]:+.2%}｜无风险RF {ann["RF"]:.2%}')

    # ---- 2. 三只美股的月度收益（在线优先，失败合成兜底） ----------------------
    try:
        rets, source = fetch_stock_monthly(list(TICKERS), DOWNLOAD_START)
    except Exception as e:                       # 网络断/接口变更等任何异常 → 走合成
        rets, source = None, f'下载异常: {type(e).__name__}: {e}'
    if rets is None:
        print(f'\n【数据兜底】yfinance 下载失败（{source}）。'
              f'改用 np.random.default_rng({SEED}) 依真实因子生成三只【合成数据】股票，'
              '以下所有结果仅演示方法。')
        rets = synthetic_stock_returns(ff3, list(TICKERS))
        src_tag = '合成数据！'
        source = f'合成数据(rng={SEED})'
    else:
        dl_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        print(f'\n【个股数据】yfinance 真实数据，下载日期 {dl_date}，'
              f'月K(auto_adjust) 2015-01 至今')
        src_tag = 'yfinance真实数据'
        source = f'yfinance({dl_date})'
    for code, r in rets.items():
        print(f'  {code}: {r.index[0]} ~ {r.index[-1]}，{len(r)} 个月收益')

    # ---- 3. 逐只 FF3 回归 ----------------------------------------------------
    rows, res_dict, diag_rows = [], {}, {}
    detail = ['=' * 78,
              '06_Fama-French三因子回归 —— 完整回归摘要（regression_detail.txt）',
              f'生成时间：{pd.Timestamp.today():%Y-%m-%d %H:%M}',
              f'个股数据来源：{source}（2015-01 至今，月频）',
              f'因子数据来源：Kenneth French 官网（data/cache/ff3_monthly.csv，单位已换算为小数）',
              f'模型：Ri - RF = a + b·Mkt-RF + s·SMB + h·HML + ε（OLS，'
              f'Newey-West HAC 标准误，maxlags=floor(4·(n/100)^(2/9))）',
              '=' * 78, '']
    print('\n【FF3 回归结果】Ri - RF = a + b·Mkt-RF + s·SMB + h·HML + ε')
    for code, name in TICKERS.items():
        excess = (rets[code] - ff3['RF']).dropna()   # 关键一步：总收益 − 无风险收益
        res, df = ff3_regression(excess, ff3)
        rows.append(results_row(code, name, res, len(df), source))
        res_dict[code] = res
        diag_rows[code] = residual_diagnostics(res)

        r = rows[-1]
        # OLS 恒等式：超额收益均值 = alpha + beta贡献。年化后拆账最直观——
        # "年化超额 X% = 风格贡献 Y% + alpha Z%"，Z 才是选股能力的部分。
        beta_contrib = float(excess.mean() - res.params['const'])
        print(f'\n  ■ {code} {name}（样本 {df.index[0]} ~ {df.index[-1]}，n={len(df)}）')
        print(f'    alpha = {r["alpha_月"]:+.3%}/月 → 年化 {r["alpha_年化"]:+.1%}    '
              f't = {r["alpha_t值"]:+.2f}   p = {r["alpha_p值"]:.4f} {stars(r["alpha_p值"])}')
        print(f'    beta  市场 {r["beta_市场"]:+.2f} (t={r["t_市场"]:+.1f})  |  '
              f'规模SMB {r["beta_规模SMB"]:+.2f} (t={r["t_规模"]:+.1f})  |  '
              f'价值HML {r["beta_价值HML"]:+.2f} (t={r["t_价值"]:+.1f})')
        print(f'    R² = {r["R2"]:.1%}   （三个因子解释了超额收益方差的多大比例）')
        print(f'    收益拆账（年化）：超额收益 {excess.mean() * 12:+.1%} '
              f'= 风格(beta)贡献 {beta_contrib * 12:+.1%} + alpha {r["alpha_年化"]:+.1%}')

        d = diag_rows[code]
        detail += [f'【{code} {name}】样本 {df.index[0]} ~ {df.index[-1]}，n={len(df)}，'
                   f'数据来源：{source}', '-' * 78, str(res.summary()).strip(), '',
                   '残差诊断：',
                   f'  正态性  Jarque-Bera = {d["JB统计量"]:.2f}，p = {d["JB_p值"]:.4f}'
                   f'（p<0.05 → 拒绝正态，金融月收益常见）',
                   f'  偏度 = {d["残差偏度"]:+.2f}，峰度 = {d["残差峰度"]:.2f}',
                   f'  自相关  Ljung-Box(6) p = {d["LjungBox6_p值"]:.4f}，'
                   f'Durbin-Watson = {d["DurbinWatson"]:.2f}',
                   f'  （这正是使用 Newey-West 标准误的原因）', '=' * 78, '']

    results = pd.DataFrame(rows)
    results.round(6).to_csv(os.path.join(OUT_DIR, 'ff3_results.csv'),
                            index=False, encoding='utf-8-sig')
    with open(os.path.join(OUT_DIR, 'regression_detail.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(detail))

    # ---- 4. 滚动 36 月 alpha（选 NVDA：beta 最高的那只最典型） ----------------
    roll = rolling_alpha(rets[ROLL_STOCK] - ff3['RF'], ff3)
    roll.to_csv(os.path.join(OUT_DIR, 'rolling_alpha.csv'),
                encoding='utf-8-sig')              # 顺手存档，README 可引用
    pos_pct = (roll['alpha_年化'] > 0).mean()
    print(f'\n【滚动 {ROLL_WIN} 月 alpha —— {ROLL_STOCK}】'
          f'{len(roll)} 个窗口，alpha>0 的窗口占 {pos_pct:.0%}；'
          f'最新窗口（{roll.index[-1]}）年化 alpha = {roll["alpha_年化"].iloc[-1]:+.1%}')
    print(f'  曲线是否稳定、是否多次穿过 0 → 见 outputs/02_滚动36月alpha.png（结论如实写在 README）')

    # ---- 5. 五张图 -----------------------------------------------------------
    nvda_res, nvda_df = ff3_regression(rets[ROLL_STOCK] - ff3['RF'], ff3)
    plot_01_fit_vs_actual(nvda_df, nvda_res, ROLL_STOCK, TICKERS[ROLL_STOCK], src_tag)
    plot_02_rolling(roll, ROLL_STOCK, TICKERS[ROLL_STOCK], src_tag)
    plot_03_diagnostics(res_dict, src_tag)
    plot_04_alpha_beta(rows, src_tag)
    plot_05_transfer_schematic()
    print('\n【残差诊断】（详表见 regression_detail.txt）')
    diag_df = pd.DataFrame(diag_rows).T.round(4)
    print(diag_df.to_string())

    print(f'\n输出目录：{OUT_DIR}')
    print(f'运行耗时 {time.time() - t0:.1f} 秒')


if __name__ == '__main__':
    main()
