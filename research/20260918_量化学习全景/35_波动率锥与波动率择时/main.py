# -*- coding: utf-8 -*-
"""
35_波动率锥与波动率择时 —— 期权交易员桌上的那张图
=====================================================
一句话点子：用真实数据画出"波动率锥"（不同期限已实现波动率的历史分位带，
期权做市商判断隐含波动率贵不贵的桌面参照系，连接课12），并检验
"高波动之后波动率倾向于回落"（均值回归，呼应课13 GARCH 与课14 OU），
最后回测一个最朴素的波动率择时规则——如实呈现它的右尾代价（呼应课28 Kelly）。

流程（与 README 教学一一对应）：
  S0 数据加载：上证指数(2010起) / 贵州茅台(2015起)；缺失时 rng(42) 合成兜底（标注）
  S1 已实现波动率：20/60/120/250 日滚动窗口，年化口径统一 √244
  S2 波动率锥：10/25/50/75/90 分位带用 2010-2023（茅台 2015-2023）估计，
              2024 之后作为"样本外走位"叠画 → 图01/02 + outputs/vol_percentile_now.csv
  S3 均值回归检验：未来20日vol变化 ~ 当前偏离250日均值的幅度（HAC稳健t值）；
              日收益平方 AR(1) 与 20日vol AR(1) → 半衰期 → 图03 + outputs/mean_reversion_stats.csv
  S4 波动率择时回测：20日vol > 滚动历史80分位 → 减半仓；回落到50分位以下 → 恢复满仓
              （滞回状态机，万3换手成本），对比买入持有 → 图04
  S5 波动率期限结构与倒挂：v20/v60/v120/v250 叠画，v20>v250（倒挂）区间着色，
              标注几次大倒挂事件的真实日期与指数位置 → 图05
  S6 GARCH条件vol vs 已实现vol vs 锥分位 三线对比（呼应课13/23）→ 图06

运行：/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py
数据：../data/cache/index_000001_上证指数.csv、stock_600519_贵州茅台.csv（缺失时合成兜底）
随机种子 42；总运行时间 < 2 分钟。
"""

import math
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # 无显示环境出图
import matplotlib.pyplot as plt

from scipy import stats as sps
import statsmodels.api as sm
from arch import arch_model

warnings.filterwarnings('ignore')  # 收敛/弃用提示统一静音，关键数字主动打印

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
np.random.seed(42)                 # 随机种子 42（合成数据兜底用）
RNG = np.random.default_rng(42)

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs'
OUT.mkdir(exist_ok=True)
DATA = BASE.parent / 'data' / 'cache'

ANNUAL = 244                       # A股年化交易日口径：全程统一 √244（见 README "常见的坑"）
SQ = math.sqrt(ANNUAL)
RF_ANNUAL = 0.02                   # 无风险利率 2% 年化（本课夏普为纯多头口径，须减 rf——与课02/07/34 等系列口径一致）
RF_LOG = math.log1p(RF_ANNUAL)     # 对数收益口径下 rf 的对数收益 ln(1.02)：择时与买入持有的收益均为对数收益
WINDOWS = (20, 60, 120, 250)       # 四个期限（交易日）
BAND_Q = [10, 25, 50, 75, 90]      # 锥的分位带
EST_END = '2023-12-31'             # 分位带估计样本：~2023-12-31
OOS_START = '2024-01-01'           # 样本外走位：2024-01-01 起
COST = 0.0003                      # 万3 单边换手成本（按仓位变动幅度计）

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 分位带配色：p50 最深，p10/p90 最浅（对称）
BAND_COLOR = {10: '#9ecae1', 25: '#4292c6', 50: '#08519c', 75: '#4292c6', 90: '#9ecae1'}


def log_ret(close: pd.Series) -> pd.Series:
    """日对数收益。"""
    return np.log(close / close.shift(1))


def realized_vols(close: pd.Series) -> dict:
    """滚动已实现波动率（年化，百分数单位，如 18.5 表示 18.5%）。

    vol(w) = std(最近 w 个日对数收益) × √244 —— "常见的坑"之一：年化因子必须全程统一。
    """
    r = log_ret(close)
    return {w: r.rolling(w).std() * SQ * 100 for w in WINDOWS}


# ============================================================================
# S0 数据加载（真实缓存优先，缺失时 rng(42) 合成兜底）
# ============================================================================
def _synthetic(start: str, n: int, s0: float, mu_d: float, sig_d: float) -> pd.DataFrame:
    """合成几何随机游走（仅当缓存缺失时使用，图表/README 会标注"合成数据"）。"""
    idx = pd.bdate_range(start, periods=n)
    r = RNG.normal(mu_d, sig_d, n)
    close = pd.Series(s0 * np.exp(np.cumsum(r)), index=idx, name='close')
    return close.to_frame()


def load_asset(fname: str, synth_kwargs: dict, min_date: str | None = None):
    """读缓存 CSV（utf-8-sig，含BOM）；缺失则合成。返回 (close序列, 是否合成)。"""
    fp = DATA / fname
    if fp.exists():
        df = pd.read_csv(fp, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        close = df['close'].astype(float)
        synth = False
    else:
        close = _synthetic(**synth_kwargs)['close']
        synth = True
        print(f'  [警告] {fname} 缺失，使用 rng(42) 合成数据（结果标注"合成数据"）')
    if min_date is not None:
        close = close[close.index >= min_date]
    return close, synth


# ============================================================================
# S2 波动率锥
# ============================================================================
def cone_bands(vols: dict) -> dict:
    """对每个期限计算：样本内分位带(10/25/50/75/90)、样本外分位带、当前值。

    关键纪律：分位带只用 ~2023-12-31 的数据估计，2024 之后整段留作样本外——
    若用全样本估分位再看"当前分位"，就是用了未来信息（前视偏差）。
    """
    out = {}
    for w in WINDOWS:
        s = vols[w].dropna()
        est = s[s.index <= EST_END]
        oos = s[s.index >= OOS_START]
        p_est = np.percentile(est.values, BAND_Q)
        p_oos = np.percentile(oos.values, BAND_Q) if len(oos) >= 60 else np.full(5, np.nan)
        out[w] = dict(est=p_est, oos=p_oos, cur=float(s.iloc[-1]),
                      cur_date=s.index[-1], n_est=len(est), n_oos=len(oos))
    return out


def percentile_rank(sample: np.ndarray, x: float) -> float:
    """x 在样本中的分位（weak 口径：样本内 <= x 的观测占比，0~100）。"""
    return float(sps.percentileofscore(sample, x, kind='weak'))


def percentile_table(name: str, vols: dict, bands: dict) -> pd.DataFrame:
    """最新分位位置表 → outputs/vol_percentile_now.csv。"""
    rows = []
    for w in WINDOWS:
        s = vols[w].dropna()
        est = s[s.index <= EST_END].values
        cur = bands[w]['cur']
        rows.append({
            '资产': name,
            '期限(交易日)': w,
            '最新日期': str(bands[w]['cur_date'].date()),
            '当前年化vol(%)': round(cur, 2),
            '分位估计样本': f'~{EST_END[:4]}年',
            '样本内分位(%)': round(percentile_rank(est, cur), 1),
            '全样本分位(%)': round(percentile_rank(s.values, cur), 1),
            '样本内p10(%)': round(bands[w]['est'][0], 2),
            '样本内p25(%)': round(bands[w]['est'][1], 2),
            '样本内p50(%)': round(bands[w]['est'][2], 2),
            '样本内p75(%)': round(bands[w]['est'][3], 2),
            '样本内p90(%)': round(bands[w]['est'][4], 2),
            '样本内观测数': bands[w]['n_est'],
        })
    return pd.DataFrame(rows)


def plot_cone(num: int, name: str, close: pd.Series, vols: dict, bands: dict,
              synth: bool, fname: str):
    """图：上排=经典锥视图（分位 vs 期限），下排=四个期限的样本外走位时间序列。"""
    fig = plt.figure(figsize=(14.5, 11.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.0],
                          hspace=0.44, wspace=0.18)
    tag = '（合成数据）' if synth else ''
    sup = (f'图{num}  {name}波动率锥{tag} —— 分位带用 ~2023 年数据估计，'
           f'2024 年之后为样本外走位')
    fig.suptitle(sup, fontsize=13, fontweight='bold')

    # ---- 上排：期权交易员的锥视图（x=期限, y=年化vol） ----
    axc = fig.add_subplot(gs[0, :])
    xs = np.array(WINDOWS, dtype=float)
    est_mat = np.vstack([bands[w]['est'] for w in WINDOWS])   # (5期限档 x 5分位)
    oos_mat = np.vstack([bands[w]['oos'] for w in WINDOWS])
    cur_vec = np.array([bands[w]['cur'] for w in WINDOWS])
    for i, q in enumerate(BAND_Q):
        axc.plot(xs, est_mat[:, i], '-o', color=BAND_COLOR[q], lw=1.8, ms=5,
                 label='样本内分位带 10/25/50/75/90（~2023估计）' if q == 50 else None)
        if np.isfinite(oos_mat[:, i]).any():
            axc.plot(xs, oos_mat[:, i], '--s', color=BAND_COLOR[q], lw=1.2, ms=4,
                     alpha=0.9, label='样本外分位带（2024–）' if q == 50 else None)
    axc.plot(xs, cur_vec, '-*', color='black', lw=2.6, ms=13,
             label='当前 vol 期限结构')
    for x, y in zip(xs, cur_vec):
        axc.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=8.5, color='black')
    # 在右端标注各分位线身份
    for i, q in enumerate(BAND_Q):
        axc.annotate(f'p{q}', (xs[-1], est_mat[-1, i]), textcoords='offset points',
                     xytext=(14, -3), fontsize=8, color=BAND_COLOR[q])
    axc.set_xscale('log')
    axc.set_xticks(xs)
    axc.set_xticklabels([f'{w}日' for w in WINDOWS])
    axc.minorticks_off()
    axc.set_ylabel('年化已实现波动率（%）')
    axc.set_title('锥视图：当前期限结构挂在历史分布的什么位置（做市商对标隐含波动率的参照系）',
                  fontsize=10.5)
    axc.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
    axc.grid(alpha=0.25)

    # ---- 下排 2x2：每个期限一条时间序列，静态分位带 + 样本外走位 ----
    pos = {20: gs[1, 0], 60: gs[1, 1], 120: gs[2, 0], 250: gs[2, 1]}
    for k, w in enumerate(WINDOWS):
        ax = fig.add_subplot(pos[w])
        s = vols[w].dropna()
        est, oos = s[s.index <= EST_END], s[s.index >= OOS_START]
        pe = bands[w]['est']
        ax.axhspan(pe[1], pe[3], color='#deebf7', alpha=0.65, zorder=0)
        for i, q in enumerate(BAND_Q):
            ax.axhline(pe[i], color=BAND_COLOR[q], lw=1.0, zorder=1)
        ax.plot(est.index, est.values, color='0.55', lw=0.6, zorder=2)
        ax.plot(oos.index, oos.values, color='#d62728', lw=1.6, zorder=3)
        ax.axvline(pd.Timestamp(OOS_START), color='0.3', ls='--', lw=1.0)
        ax.plot([s.index[-1]], [s.iloc[-1]], 'o', color='#d62728', ms=5, zorder=4)
        ax.annotate(f'最新 {s.iloc[-1]:.1f}%\n(样本内分位 '
                    f'{percentile_rank(s[s.index <= EST_END].values, s.iloc[-1]):.0f}%)',
                    (s.index[-1], s.iloc[-1]), textcoords='offset points',
                    xytext=(-88, -36), fontsize=8, color='#7f1d1d', zorder=5,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='0.8'))
        ax.set_title(f'{w}日已实现波动率（年化%）', fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlim(s.index[0] - pd.Timedelta(days=120), s.index[-1] + pd.Timedelta(days=160))
        if k == 0:
            from matplotlib.lines import Line2D
            handles = [Line2D([], [], color='0.55', lw=1, label='样本内 vol（~2023）'),
                       Line2D([], [], color='#d62728', lw=1.6, label='样本外走位（2024–）'),
                       Line2D([], [], color='#4292c6', lw=1, label='p10/25/50/75/90 分位线')]
            ax.legend(handles=handles, loc='upper right', fontsize=7.5, framealpha=0.9)

    fig.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# S3 均值回归检验
# ============================================================================
def mean_reversion(close: pd.Series, v20: pd.Series) -> dict:
    """三个统计证据：
    (1) 未来20日vol变化 dv ~ 对数偏离 dev = ln(v20/MA250)，HAC(20) t值（重叠窗口→自相关）
    (2) 相对收敛回归 (dv/MA250) ~ dev，斜率≈ -f：每20日收敛偏离的比例 f
    (3) AR(1)：日收益平方 与 20日vol 各自的 φ 与半衰期 HL = -ln2/ln(φ)（呼应课14 OU）
    """
    r = log_ret(close)
    ma250 = v20.rolling(250).mean()
    dev = np.log(v20 / ma250)
    dv = v20.shift(-20) - v20                       # 未来20日 vol 变化（百分点）
    rel = (v20.shift(-20) - v20) / ma250            # 相对变化（对250日均值的比例）
    d = pd.DataFrame({'dv': dv, 'dev': dev, 'rel': rel}).dropna()

    # (1) 原始回归：dv ~ dev（任务口径）
    X = sm.add_constant(d['dev'])
    res1 = sm.OLS(d['dv'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 20})
    b, t1, p1 = res1.params.iloc[1], res1.tvalues.iloc[1], res1.pvalues.iloc[1]
    r2 = res1.rsquared

    # (2) 可解释回归：rel ~ dev → 斜率 ≈ -f（每20日收敛比例）
    res2 = sm.OLS(d['rel'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 20})
    f20 = -res2.params.iloc[1]
    f_t = -res2.tvalues.iloc[1]
    f_p = res2.pvalues.iloc[1]
    hl_f = 20 * math.log(0.5) / math.log(1 - f20) if 0 < f20 < 1 else float('nan')

    # (3a) AR(1) on 平方日收益：r_t^2 = a + φ r_{t-1}^2 + e
    x = (r ** 2).dropna()
    dx = pd.DataFrame({'x': x, 'x1': x.shift(1)}).dropna()
    res3 = sm.OLS(dx['x'], sm.add_constant(dx['x1'])).fit()
    phi_r2 = res3.params.iloc[1]
    hl_r2 = -math.log(2) / math.log(phi_r2) if 0 < phi_r2 < 1 else float('nan')

    # (3b) AR(1) on 20日vol 水平（平滑序列，φ 更高但受重叠平滑影响）
    dvv = pd.DataFrame({'v': v20, 'v1': v20.shift(1)}).dropna()
    res4 = sm.OLS(dvv['v'], sm.add_constant(dvv['v1'])).fit()
    phi_v = res4.params.iloc[1]
    hl_v = -math.log(2) / math.log(phi_v) if 0 < phi_v < 1 else float('nan')

    return dict(res1=res1, res2=res2, d=d, n=len(d),
                b=b, t1=t1, p1=p1, r2=r2, f20=f20, f_t=f_t, f_p=f_p, hl_f=hl_f,
                phi_r2=phi_r2, hl_r2=hl_r2, phi_v=phi_v, hl_v=hl_v)


def mean_reversion_table(name: str, mr: dict) -> dict:
    return {
        '资产': name,
        '样本数': mr['n'],
        '斜率b_未来20日vol变化(pp/对数偏离)': round(mr['b'], 3),
        'b_t值_HAC20': round(mr['t1'], 2),
        'b_p值': f"{mr['p1']:.2e}",
        'R2': round(mr['r2'], 3),
        '每20日收敛比例f': round(mr['f20'], 3),
        'f_t值': round(mr['f_t'], 2),
        '半衰期_偏离收敛(交易日)': round(mr['hl_f'], 1),
        'AR1phi_日收益平方': round(mr['phi_r2'], 3),
        '半衰期_日收益平方(日)': round(mr['hl_r2'], 2),
        'AR1phi_20日vol': round(mr['phi_v'], 3),
        '半衰期_20日vol(日)': round(mr['hl_v'], 1),
    }


def plot_mean_reversion(mrs: dict, fname: str):
    """图03：dev vs 未来20日vol变化 散点 + 拟合线（左右两资产）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, (name, mr) in zip(axes, mrs.items()):
        d = mr['d']
        ax.scatter(d['dev'] * 100, d['dv'], s=7, alpha=0.22, color='#4292c6',
                   edgecolors='none')
        xs = np.linspace(d['dev'].min(), d['dev'].max(), 50)
        ax.plot(xs * 100, mr['res1'].params.iloc[0] + mr['res1'].params.iloc[1] * xs,
                color='#d62728', lw=2.2, label='OLS拟合（HAC t值）')
        ax.axhline(0, color='0.4', lw=0.8, ls=':')
        txt = (f"斜率 b = {mr['b']:.2f} vol百分点 / 对数偏离\n"
               f"t = {mr['t1']:.1f}（HAC20）,  p = {mr['p1']:.1e},  R² = {mr['r2']:.3f}\n"
               f"每20日收敛比例 f = {mr['f20']:.0%}（t={mr['f_t']:.1f}）\n"
               f"半衰期 ≈ {mr['hl_f']:.0f} 个交易日")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='0.7'))
        ax.set_xlabel('当前20日vol对250日均值的对数偏离（%）')
        ax.set_ylabel('未来20日vol变化（百分点）')
        ax.set_title(f'{name}：偏离越大，未来vol越倾向回落', fontsize=11)
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle('图03  波动率均值回归检验（重叠窗口 → Newey-West HAC 稳健t值）',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# S4 波动率择时回测
# ============================================================================
def vol_timing(close: pd.Series) -> dict:
    """滞回状态机：v20 > 滚动历史80分位 → 目标仓位0.5；v20 < 50分位 → 回到1.0；
    50~80之间维持原仓位。分位只用"昨日及以前"的 expanding 分布（防前视）。
    仓位在次日生效；换手收万3成本。
    """
    r = log_ret(close)
    v20 = r.rolling(20).std() * SQ * 100
    hist = v20.shift(1)                              # 只用截至昨日的分布
    p80 = hist.expanding(min_periods=250).quantile(0.80)
    p50 = hist.expanding(min_periods=250).quantile(0.50)

    target = pd.Series(np.nan, index=close.index)
    cur = 1.0
    vals = []
    for t in close.index:
        v, a, b = v20.loc[t], p80.loc[t], p50.loc[t]
        if pd.notna(v) and pd.notna(a) and pd.notna(b):
            if v > a:
                cur = 0.5                            # 高波动 → 减半仓
            elif v < b:
                cur = 1.0                            # 回落到50分位以下 → 恢复满仓
        vals.append(cur)
    target[:] = vals

    eff = target.shift(1)                            # 信号次日生效
    cost = eff.diff().abs().fillna(0.0) * COST
    strat = eff * r - cost
    first = strat.first_valid_index()
    strat = strat.loc[first:]
    bh = r.loc[first:]

    nav_s = np.exp(strat.cumsum())
    nav_b = np.exp(bh.cumsum())
    dd_s = nav_s / nav_s.cummax() - 1
    dd_b = nav_b / nav_b.cummax() - 1

    n_days = len(strat)
    years = n_days / ANNUAL

    def metrics(nav, ret):
        cagr = nav.iloc[-1] ** (1 / years) - 1
        vol_a = ret.std() * SQ
        # 夏普 = (年化对数收益 − rf) / 年化波动；本课收益为对数口径，rf=2% 换算为 ln(1.02)。
        # 择时与买入持有均为纯多头（0.5/1.0 仓），统一减 rf=2%（复核修复：原版未减 rf）。
        sharpe = (ret.mean() * ANNUAL - RF_LOG) / vol_a if vol_a > 0 else np.nan
        return dict(nav=float(nav.iloc[-1]), cagr=float(cagr),
                    vol=float(vol_a), sharpe=float(sharpe),
                    maxdd=float(dd_s.min() if nav is nav_s else dd_b.min()))

    m_s = metrics(nav_s, strat)
    m_b = metrics(nav_b, bh)

    half = (eff == 0.5).reindex(bh.index).fillna(False)   # 与回测区间对齐
    full = (eff == 1.0).reindex(bh.index).fillna(False)
    r_half = bh[half.values]
    r_full = bh[full.values]
    # 右尾证据：买入持有最好的1%交易日，有多少落在减仓区间
    top = bh.nlargest(max(1, int(len(bh) * 0.01)))
    top_in_half = int(half.loc[top.index].sum())

    switches = int((eff.diff().abs() > 0).sum())
    cost_drag = float(cost.loc[first:].sum())        # 累计成本（对数收益口径）

    return dict(r=r, v20=v20, p80=p80, p50=p50, eff=eff, strat=strat, bh=bh,
                nav_s=nav_s, nav_b=nav_b, dd_s=dd_s, dd_b=dd_b,
                m_s=m_s, m_b=m_b, half=half, r_half=r_half, r_full=r_full,
                top_in_half=top_in_half, n_top=len(top), switches=switches,
                cost_drag=cost_drag, first=first, years=years,
                half_share=float(half.loc[first:].mean()))


def plot_timing(name: str, bt: dict, synth: bool, fname: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.12})
    tag = '（合成数据）' if synth else ''
    ax1.fill_between(bt['nav_s'].index, 0, 1, where=bt['half'].loc[bt['nav_s'].index],
                     transform=ax1.get_xaxis_transform(), color='#d62728', alpha=0.07,
                     label='减半仓区间（v20>历史80分位）')
    ax1.plot(bt['nav_b'], color='0.45', lw=1.3,
             label=f"买入持有：净值 {bt['m_b']['nav']:.2f}，年化 {bt['m_b']['cagr']:.1%}，"
                   f"波动 {bt['m_b']['vol']:.1%}，最大回撤 {bt['m_b']['maxdd']:.1%}")
    ax1.plot(bt['nav_s'], color='#d62728', lw=1.5,
             label=f"vol择时：净值 {bt['m_s']['nav']:.2f}，年化 {bt['m_s']['cagr']:.1%}，"
                   f"波动 {bt['m_s']['vol']:.1%}，最大回撤 {bt['m_s']['maxdd']:.1%}")
    ax1.set_yscale('log')
    ax1.set_ylabel('净值（对数轴）')
    ax1.set_title(f'图04  {name}波动率择时 vs 买入持有{tag}——'
                  f'80/50分位滞回，万3成本（红色底纹=减半仓期间）', fontsize=11.5)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.25, which='both')

    ax2.plot(bt['dd_b'] * 100, color='0.45', lw=1.0, label='买入持有回撤')
    ax2.plot(bt['dd_s'] * 100, color='#d62728', lw=1.1, label='vol择时回撤')
    ax2.set_ylabel('回撤（%）')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(alpha=0.25)
    fig.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# S5 期限结构与倒挂
# ============================================================================
def find_episodes(inv: pd.Series, spread: pd.Series, min_days: int = 5) -> list:
    """v20>v250 的连续区段（>=min_days），按峰值价差排序信息。"""
    vals = inv.values
    idx = inv.index
    sp = spread.values
    eps, i, n = [], 0, len(vals)
    while i < n:
        if vals[i]:
            j = i
            while j + 1 < n and vals[j + 1]:
                j += 1
            if j - i + 1 >= min_days:
                k = i + int(np.argmax(sp[i:j + 1]))
                eps.append(dict(start=idx[i], end=idx[j], days=j - i + 1,
                                peak_date=idx[k], peak_spread=float(sp[k])))
            i = j + 1
        else:
            i += 1
    return eps


def plot_term_structure(name: str, close: pd.Series, vols: dict, synth: bool, fname: str):
    v20, v250 = vols[20], vols[250]
    spread = v20 - v250
    inv = (v20 > v250) & v20.notna() & v250.notna()
    eps = find_episodes(inv, spread)
    top = sorted(eps, key=lambda e: -e['peak_spread'])[:5]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [1.1, 1], 'hspace': 0.10})
    tag = '（合成数据）' if synth else ''
    ax1.plot(close.index, close.values, color='black', lw=1.0, label=f'{name} 收盘价')
    ax1.fill_between(close.index, 0, 1, where=inv.reindex(close.index, fill_value=False),
                     transform=ax1.get_xaxis_transform(), color='#d62728', alpha=0.10,
                     label='期限倒挂区间（v20>v250）')
    ax1.set_yscale('log')
    ax1.set_ylabel('点位（对数轴）')
    ax1.set_title(f'图05  {name}波动率期限结构倒挂 vs 指数位置{tag}——'
                  f'短期vol>长期vol 常出现在恐慌底部附近（经验信号，非确定性）',
                  fontsize=11.5)
    # 按峰值价差排名错开标注位置，避免文字互相重叠
    offs = [(-150, 35), (-175, 55), (-175, -50), (-165, -35), (-40, 85)]
    for k, e in enumerate(top):
        px = float(close.loc[e['peak_date']])
        dx, dy = offs[k % len(offs)]
        ax1.annotate(f"{e['peak_date'].date()}\n{px:.0f}点  v20-v250={e['peak_spread']:.1f}pp",
                     (e['peak_date'], px), textcoords='offset points', xytext=(dx, dy),
                     fontsize=8.5, color='#7f1d1d', zorder=6,
                     bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='0.8'),
                     arrowprops=dict(arrowstyle='->', color='#7f1d1d', lw=1.0))
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.25, which='both')

    ax2.plot(vols[20].index, vols[20].values, color='#d62728', lw=1.1, label='20日vol')
    ax2.plot(vols[60].index, vols[60].values, color='#ff7f0e', lw=1.0, label='60日vol')
    ax2.plot(vols[120].index, vols[120].values, color='#2ca02c', lw=1.0, label='120日vol')
    ax2.plot(vols[250].index, vols[250].values, color='#1f77b4', lw=1.2, label='250日vol')
    ax2.fill_between(v20.index, 0, 1, where=inv, transform=ax2.get_xaxis_transform(),
                     color='#d62728', alpha=0.10)
    ax2.set_ylabel('年化vol（%）')
    ax2.legend(loc='upper right', fontsize=9, ncol=4)
    ax2.grid(alpha=0.25)
    fig.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return eps, top, float(inv.mean() * 100)


# ============================================================================
# S6 GARCH 条件vol vs 已实现vol vs 锥分位
# ============================================================================
def plot_garch(name: str, close: pd.Series, vols: dict, bands: dict,
               synth: bool, fname: str, win_start: str = '2022-01-01'):
    r = (log_ret(close).dropna() * 100)              # arch 用百分数收益
    am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    om = float(res.params['omega'])
    al = float(res.params['alpha[1]'])
    be = float(res.params['beta[1]'])
    pers = al + be
    hl_g = math.log(0.5) / math.log(pers) if 0 < pers < 1 else float('inf')
    cond = pd.Series(res.conditional_volatility, index=r.index) / 100 * SQ * 100  # 年化%

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    tag = '（合成数据）' if synth else ''
    w = cond[cond.index >= win_start]
    ax.plot(w.index, w.values, color='#1f77b4', lw=1.3, label='GARCH(1,1) 条件vol（课13）')
    v20w = vols[20][vols[20].index >= win_start]
    ax.plot(v20w.index, v20w.values, color='#d62728', lw=1.3, label='20日已实现vol（滚动）')
    for i, q in enumerate(BAND_Q):
        ax.axhline(bands[20]['est'][i], color=BAND_COLOR[q], lw=1.0, ls='--')
        ax.annotate(f'锥p{q}', (w.index[int(len(w) * 0.28)], bands[20]['est'][i]),
                    textcoords='offset points', xytext=(0, 4), fontsize=8, zorder=5,
                    color=BAND_COLOR[q],
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.75, ec='none'))
    ax.set_ylabel('年化波动率（%）')
    ax.set_title(f'图06  {name}：GARCH条件vol vs 已实现vol vs 波动率锥分位{tag}\n'
                 f'GARCH(1,1) 全样本拟合：ω={om:.4f}, α={al:.3f}, β={be:.3f}, '
                 f'α+β={pers:.3f}（条件vol半衰期≈{hl_g:.0f}日，呼应课13/14）',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return dict(omega=om, alpha=al, beta=be, pers=pers, hl=hl_g,
                llf=float(res.loglikelihood), n=int(res.nobs))


# ============================================================================
# 主流程
# ============================================================================
def main():
    t0 = time.time()
    print('=' * 72)
    print('35_波动率锥与波动率择时 —— 期权交易员桌上的那张图（随机种子42）')
    print('=' * 72)

    # ---------- S0 数据 ----------
    idx_close, synth_idx = load_asset('index_000001_上证指数.csv',
                                      dict(start='2010-01-04', n=4055, s0=3243.0,
                                           mu_d=0.0, sig_d=0.22 / SQ))
    mt_close, synth_mt = load_asset('stock_600519_贵州茅台.csv',
                                    dict(start='2015-01-05', n=2845, s0=143.0,
                                         mu_d=0.0004, sig_d=0.30 / SQ))
    print(f"[S0] 上证指数 {idx_close.index[0].date()} ~ {idx_close.index[-1].date()}，"
          f"{len(idx_close)}个交易日（合成={synth_idx}）")
    print(f"[S0] 贵州茅台 {mt_close.index[0].date()} ~ {mt_close.index[-1].date()}，"
          f"{len(mt_close)}个交易日（合成={synth_mt}）")

    # ---------- S1/S2 波动率锥 ----------
    print('\n' + '-' * 72)
    print('[S2] 波动率锥：分位带 10/25/50/75/90（样本 ~%s），样本外 %s 起' %
          (EST_END[:4], OOS_START[:4]))
    tables = []
    mrs = {}
    assets = [('上证指数', idx_close, synth_idx, '01_上证波动率锥.png'),
              ('贵州茅台', mt_close, synth_mt, '02_茅台波动率锥.png')]
    for name, close, synth, figname in assets:
        vols = realized_vols(close)
        bands = cone_bands(vols)
        tab = percentile_table(name, vols, bands)
        tables.append(tab)
        plot_cone(1 if name == '上证指数' else 2, name, close, vols, bands,
                  synth, figname)
        print(f'\n  【{name}】当前各期限vol与历史分位（分位带样本 ~{EST_END[:4]} 年）：')
        show = tab[['期限(交易日)', '当前年化vol(%)', '样本内分位(%)', '全样本分位(%)',
                    '样本内p10(%)', '样本内p50(%)', '样本内p90(%)']]
        print(show.to_string(index=False))

        # ---------- S3 均值回归 ----------
        v20 = vols[20]
        mr = mean_reversion(close, v20)
        mrs[name] = mr
        print(f'\n  【{name}】均值回归检验（样本 n={mr["n"]}）：')
        print(f"    dv~dev：斜率 b={mr['b']:.2f} pp，t={mr['t1']:.1f}(HAC20)，"
              f"p={mr['p1']:.1e}，R²={mr['r2']:.3f}")
        print(f"    相对收敛：每20日收敛比例 f={mr['f20']:.1%}（t={mr['f_t']:.1f}），"
              f"半衰期≈{mr['hl_f']:.0f}个交易日")
        print(f"    AR(1)日收益平方：φ={mr['phi_r2']:.3f} → 半衰期 {mr['hl_r2']:.2f} 日"
              f"（采样噪声大，见README）")
        print(f"    AR(1) 20日vol：φ={mr['phi_v']:.3f} → 半衰期 {mr['hl_v']:.0f} 日")
    pd.concat(tables, ignore_index=True).to_csv(
        OUT / 'vol_percentile_now.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([mean_reversion_table(n, m) for n, m in mrs.items()]).to_csv(
        OUT / 'mean_reversion_stats.csv', index=False, encoding='utf-8-sig')
    plot_mean_reversion(mrs, '03_均值回归回归散点.png')
    print(f'\n  已输出：outputs/vol_percentile_now.csv、outputs/mean_reversion_stats.csv')

    # ---------- S4 波动率择时回测 ----------
    print('\n' + '-' * 72)
    print('[S4] 波动率择时回测（上证指数）：v20>历史80分位→减半仓；<50分位→恢复满仓；万3成本')
    bt = vol_timing(idx_close)
    m_s, m_b = bt['m_s'], bt['m_b']
    print(f"    回测区间：{bt['first'].date()} ~ {bt['nav_s'].index[-1].date()}"
          f"（{bt['years']:.1f}年），减半仓时间占比 {bt['half_share']:.1%}，"
          f"调仓 {bt['switches']} 次，累计成本 {bt['cost_drag']:.2%}")
    print(f"    买入持有：净值 {m_b['nav']:.2f}，年化 {m_b['cagr']:.2%}，"
          f"波动 {m_b['vol']:.2%}，夏普 {m_b['sharpe']:.2f}，最大回撤 {m_b['maxdd']:.1%}")
    print(f"    vol择时：净值 {m_s['nav']:.2f}，年化 {m_s['cagr']:.2%}，"
          f"波动 {m_s['vol']:.2%}，夏普 {m_s['sharpe']:.2f}，最大回撤 {m_s['maxdd']:.1%}")
    print(f"    右尾代价：减半仓期间日均收益 {bt['r_half'].mean():.4%} vs "
          f"满仓期间 {bt['r_full'].mean():.4%}；"
          f"买入持有最好的 {bt['n_top']} 个交易日中 {bt['top_in_half']} 个落在减仓区间")
    plot_timing('上证指数', bt, synth_idx, '04_vol择时净值.png')

    # ---------- S5 期限结构倒挂 ----------
    print('\n' + '-' * 72)
    print('[S5] 波动率期限结构与倒挂（上证指数）：v20>v250 视为倒挂')
    idx_vols = realized_vols(idx_close)
    eps, top, inv_share = plot_term_structure('上证指数', idx_close, idx_vols,
                                              synth_idx, '05_期限结构倒挂标注.png')
    print(f"    倒挂时间占比 {inv_share:.1f}%，共 {len(eps)} 段（>=5日），峰值价差前5段：")
    for e in top:
        px = float(idx_close.loc[e['peak_date']])
        print(f"      {e['start'].date()} ~ {e['end'].date()}（{e['days']}日），"
              f"峰值 {e['peak_date'].date()}：v20-v250={e['peak_spread']:.1f}pp，"
              f"指数 {px:.0f} 点")

    # ---------- S6 GARCH 对照 ----------
    print('\n' + '-' * 72)
    print('[S6] GARCH(1,1) 条件vol vs 20日已实现vol vs 锥分位（2022年起展示）')
    gp = plot_garch('上证指数', idx_close, idx_vols,
                    cone_bands(idx_vols), synth_idx, '06_GARCH_vs_已实现vol.png')
    print(f"    ω={gp['omega']:.4f}，α={gp['alpha']:.3f}，β={gp['beta']:.3f}，"
          f"α+β={gp['pers']:.3f}，条件vol半衰期≈{gp['hl']:.0f}日（n={gp['n']}）")

    print('\n' + '=' * 72)
    print(f'全部完成，用时 {time.time() - t0:.1f} 秒。图表与CSV见 outputs/')
    print('=' * 72)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # 教学约定：任何异常都打印但不以非0退出（数据/环境问题兜底）
        traceback.print_exc()
