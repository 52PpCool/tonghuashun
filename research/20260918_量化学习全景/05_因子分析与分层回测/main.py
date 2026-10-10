# -*- coding: utf-8 -*-
"""
05_因子分析与分层回测 —— 量化选股的机构最小完整工作流
=========================================================

一节课跑通因子研究的四步闭环：

    构造因子 → 截面预处理（去极值 + 标准化） → IC检验（能不能预测） → 分层回测（能不能赚钱）

三个经典因子（每月末计算一次"截面"，方向统一为：值越大 → 预期收益越高）：
  1) 动量因子   ：过去12个月收益，剔除最近1个月（"12-1动量"）
  2) 反转因子   ：过去5个交易日收益的相反数（短期反转效应）
  3) 低波动因子 ：过去60个交易日收益波动率的相反数（低波动异象）

======================================================================
【严谨性声明 —— 请务必先读】
======================================================================
1. 幸存者偏差：本课股票池是 30 只"今天的知名公司"。它们今天还在名单上，
   本身就说明过去活得好。用这种"事后名单"回测，结果会系统性偏乐观；
   真实机构必须用"时点股票池"（每个月末只看当时已上市、后来未退市的
   全市场股票）。本课目的是掌握方法，不是复现收益。
2. T+1 对齐：因子用"截至 t 日（月末）收盘"的数据计算，t 日收盘那一刻
   你还没下单，最早 t+1 日才能成交。因此持有收益严格从 t+1 日收盘价
   起算：持有收益 = 下月末次日收盘 / 本月末次日收盘 - 1。
   若直接用"下月末收盘 / 本月末收盘"对齐因子，就偷用了 t 日当天的
   未来信息（前视偏差），回测收益会被虚高。
======================================================================
"""

# ============================================================
# 第 0 节  导入与全局配置
# ============================================================
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 无界面环境：只输出图片文件，不弹窗
import matplotlib.pyplot as plt

# 中文字体（第一个缺失时自动回退到列表中的下一个）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 让负号正常显示

SEED = 42
rng = np.random.default_rng(SEED)  # 本流程是确定性的（无随机步骤），仍按规范固定种子，
                                   # 保证将来加入随机环节（如抽样、扰动）时结果可复现

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'cache'))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS = 5            # 分层数：Q1(因子值最低) ~ Q5(因子值最高)
MIN_STOCKS = 10         # 某月截面上有效股票少于该数 → 丢弃该月
MOM_LOOKBACK = 12       # 动量回看的月数
MOM_SKIP = 1            # 动量剔除最近的月数（12-1 动量）
REV_LOOKBACK = 5        # 反转回看的交易日数
VOL_LOOKBACK = 60       # 低波动回看的交易日数
ANALYSIS_START = '2019-01'   # 统一分析起点：保证三个因子对全部个股都有完整历史
                             #（28/30 只股票的缓存从 2018 年开始，2018 年用作因子预热）
FACTOR_NAMES = ['动量', '反转', '低波动']


# ============================================================
# 第 1 节  数据加载与"面板"构造
# ============================================================
def load_close_panel(data_dir: str) -> pd.DataFrame:
    """把 data/cache/stock_*.csv 全部读入，拼成收盘价"面板"。

    面板（panel）就是一张大表：行=交易日，列=股票代码，值=前复权收盘价。
    有了面板，任何"截面"就是其中的一行，任何"时间序列"就是其中的一列。
    个股停牌/未上市造成的缺失保留为 NaN，后续每个截面各自 dropna 处理。
    """
    files = sorted(f for f in os.listdir(data_dir)
                   if f.startswith('stock_') and f.endswith('.csv'))
    if not files:
        raise FileNotFoundError(f'{data_dir} 下没有找到 stock_*.csv 缓存文件')
    cols = {}
    for fname in files:
        code = fname.split('_')[1]           # 文件名形如 stock_000001_平安银行.csv
        df = pd.read_csv(os.path.join(data_dir, fname), encoding='utf-8-sig',
                         usecols=['date', 'close'], parse_dates=['date'])
        s = (df.drop_duplicates(subset='date', keep='last')
               .set_index('date')['close'].sort_index())
        cols[code] = s
    close = pd.DataFrame(cols).sort_index()
    return close


def get_month_end_days(close: pd.DataFrame) -> pd.Series:
    """取每个月最后一个"真实交易日"（而不是自然月的最后一天）。

    返回 Series：index=月份(Period)，value=该月最后一个交易日的 Timestamp。
    A股节假日会导致自然月末不是交易日，必须用真实交易日做月末快照，
    才能和"次日才能成交"的 T+1 逻辑对上。
    若缓存截止日在月中（最后一个自然月不完整），整月剔除，
    否则"半个月的行情"会被当成整月收益，污染回测。
    """
    me = close.index.to_series().groupby(close.index.to_period('M')).max()
    last_day = close.index[-1]
    if last_day < last_day + pd.offsets.MonthEnd(0):     # 最后一个月不完整
        me = me[me.index < last_day.to_period('M')]
    return me


# ============================================================
# 第 2 节  因子构造（月末截面）
# ============================================================
def build_factors(close, me_all, me_an):
    """在每个月末截面上构造三个因子，方向统一：值越大 → 预期收益越高。

    参数 me_all：全部月份的月末（含预热期，供动量取 13 个月前的价格）；
         me_an ：分析期（ANALYSIS_START 起）的月末，因子只在这些时点输出。
    返回 dict[因子名] = DataFrame(index=月末交易日, columns=股票代码)。
    """
    close_m = close.loc[me_all.values]        # 月末收盘面板（一行=一个月）

    # ---- 1) 动量因子（12-1）：从 t-12 月末到 t-1 月末的收益 ---------------
    # 为什么剔除最近1个月？因为约1个月的短期内存在"反转效应"：刚暴涨的
    # 股票下个月往往回调。不剔除的话，动量信号会被最近一个月的反向运动
    # 污染，IC 明显衰减（Jegadeesh & Titman, 1993 的经典做法）。
    mom_raw = close_m.shift(MOM_SKIP) / close_m.shift(MOM_LOOKBACK) - 1.0
    mom = mom_raw.loc[me_an.values]

    # ---- 2) 反转因子：过去5个交易日收益取负号 ----------------------------
    # A股短线"涨多了要跌、跌多了要弹"的反转效应很强：过去5日收益对未来
    # 收益是负向预测。取负号后，"反转因子值越大 = 跌得越狠 = 越值得买"，
    # 与另外两个因子方向统一，方便比较 IC、画图与分层。
    ret5 = close / close.shift(REV_LOOKBACK) - 1.0
    rev = (-ret5).loc[me_an.values]

    # ---- 3) 低波动因子：过去60个交易日收益波动率取负号 -------------------
    # "低波动异象"：波动小的股票长期反而跑赢。原始信号是波动率（越大越差），
    # 取负号统一方向：因子值越大 = 波动越低 = 越值得买。
    vol60 = close.pct_change().rolling(VOL_LOOKBACK, min_periods=40).std()
    lowvol = (-vol60).loc[me_an.values]

    return {'动量': mom, '反转': rev, '低波动': lowvol}


# ============================================================
# 第 3 节  截面预处理：MAD 去极值 + z-score 标准化
# ============================================================
def cross_section_process(row: pd.Series) -> pd.Series:
    """对"一个月末截面"（因子面板的一行）做两步标准化处理。

    第1步 MAD 去极值：把超出 [中位数±3×1.4826×MAD] 的值"拉回"边界。
      为什么必须做？截面里常有涨跌停、复牌跳空制造的极端因子值。极端值会
      同时绑架"均值"和"标准差"，z-score 之后绝大多数股票被挤在极小区间、
      少数妖股独占 ±3，多因子合成时权重失真、分层名存实亡。
      中位数和 MAD 是稳健统计量（对极端值不敏感），用它们定边界最稳妥；
      1.4826 是让 MAD 在正态假设下约等于 1 个标准差的修正系数。
    第2步 z-score 标准化：(x - 均值) / 标准差。
      为什么必须做？不同因子量纲和尺度差异巨大，标准化后每个因子都变成
      "相对截面平均偏离了几个标准差"，因子之间才可比、可相关、可加权合成。
    注意：这两步都是"保序"变换——不改变截面排序，因此对 RankIC 和分层
    结果没有影响；它们影响的是依赖"数值大小"的场合（Pearson 相关、
    多因子等权合成、回归）。
    """
    x = row.dropna()
    if len(x) < MIN_STOCKS:
        return pd.Series(np.nan, index=row.index)
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:      # 截面几乎无差异（罕见），放弃该月
        return pd.Series(np.nan, index=row.index)
    upper, lower = med + 3 * 1.4826 * mad, med - 3 * 1.4826 * mad
    x = x.clip(lower=lower, upper=upper)
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=row.index)
    return ((x - x.mean()) / sd).reindex(row.index)


# ============================================================
# 第 4 节  T+1 对齐的下月持有收益 与 IC 检验
# ============================================================
def build_hold_returns(close, me_all, me_an):
    """构造"T+1 对齐"的下月持有收益矩阵（每只股票每期的持有收益率）。

    时间线：t 日（月末）收盘 → 因子算好 → t+1 日收盘买入（进场价 entry）
            → 下个月末 t' 的次日收盘卖出（出场价 = entry'）
    持有收益 = entry' / entry - 1。
    两个调仓日之间被完整覆盖、不重叠，且不含任何未来信息。
    """
    entry_m = close.shift(-1).loc[me_all.values]   # 每个月末"次日"的收盘价（进场价）
    hold = entry_m.shift(-1) / entry_m - 1.0       # 本月末次日 → 下月末次日
    return hold.loc[me_an.values]


def monthly_ic(f_row, r_row, method='spearman', min_n=MIN_STOCKS):
    """单月截面 IC：因子值与下月收益的相关系数。

    method='spearman' 即 RankIC（秩相关）：只看排序不看数值，
    对极端值稳健，是业界检验因子预测力的标准指标。
    """
    df = pd.concat({'f': f_row, 'r': r_row}, axis=1).dropna()
    if len(df) < min_n:
        return np.nan
    return df['f'].corr(df['r'], method=method)


def ic_stats_table(ic: pd.DataFrame) -> pd.DataFrame:
    """逐月 RankIC 序列 → 汇总统计：均值 / 标准差 / ICIR / IC>0占比 / t值。

    ICIR = IC均值 / IC标准差，衡量"预测能力的稳定性"，
    月频因子 |ICIR|>0.5 已算相当不错（类比：夏普比率之于收益率）。
    t值 = ICIR × sqrt(月数)，|t|>2 大致可认为 IC 显著不为 0。
    """
    rows = {}
    for name in ic.columns:
        s = ic[name].dropna()
        n = len(s)
        mean, sd = s.mean(), s.std(ddof=1)
        icir = mean / sd if sd > 0 else np.nan
        rows[name] = {'IC均值': mean, 'IC标准差': sd, 'ICIR': icir,
                      'IC大于0占比': (s > 0).mean(), 't值': icir * np.sqrt(n),
                      '有效月数': n}
    return pd.DataFrame(rows).T


# ============================================================
# 第 5 节  分层回测（5 分位组 + 多空对冲）
# ============================================================
def layer_backtest(fz: pd.DataFrame, hold: pd.DataFrame,
                   n_layers: int = N_LAYERS) -> pd.DataFrame:
    """每月末按因子值把股票分成 n_layers 组，等权持有到下月末。

    Q1 = 因子值最低的一组 …… Q5 = 因子值最高的一组（方向统一后 Q5 即
    "预期最好"组）。每组收益 = 组内个股当期收益的等权平均。
    多空对冲 = Q5 - Q1：做多 Q5、做空 Q1，剔除了市场涨跌的近似中性组合。
    返回 DataFrame：index=月末，columns=[Q1..Q5, 多空Q5减Q1]（月收益）。
    """
    recs = {}
    for t in fz.index:
        df = pd.concat({'f': fz.loc[t], 'r': hold.loc[t]}, axis=1).dropna()
        if len(df) < MIN_STOCKS:
            continue
        # 先 rank(method='first') 把并列值按固定顺序打散，保证分组数量均匀
        grp = pd.qcut(df['f'].rank(method='first'), n_layers, labels=False)
        mean_r = df.groupby(grp)['r'].mean()
        rec = {f'Q{i + 1}': mean_r.get(i, np.nan) for i in range(n_layers)}
        rec['多空Q5减Q1'] = mean_r.get(n_layers - 1, np.nan) - mean_r.get(0, np.nan)
        recs[t] = rec
    out = pd.DataFrame.from_dict(recs, orient='index').sort_index()
    return out[[f'Q{i + 1}' for i in range(n_layers)] + ['多空Q5减Q1']]


def perf_stats(monthly_ret: pd.Series, rf_annual: float = 0.0) -> dict:
    """月收益序列 → 月均/年化收益、年化波动、夏普、t值、期末净值。

    年化收益用几何法：期末净值 ** (12/月数) - 1；年化波动 = 月波动 × sqrt(12)；
    夏普 = (月均收益 − rf/12)/月波动 × sqrt(12)：纯多头层 rf=2% 年化（月扣 0.02/12），
    多空 Q5−Q1 自融资组合 rf=0（README 已声明）。不扣交易成本——读数时请打折。
    """
    r = monthly_ret.dropna()
    n = len(r)
    nav = (1.0 + r).cumprod()
    sd = r.std(ddof=1)
    return {
        '月均收益': r.mean() if n else np.nan,
        '年化收益': nav.iloc[-1] ** (12.0 / n) - 1.0 if n else np.nan,
        '年化波动': sd * np.sqrt(12) if n > 1 else np.nan,
        '夏普': (r.mean() - rf_annual / 12.0) / sd * np.sqrt(12) if (n > 1 and sd > 0) else np.nan,
        't值': r.mean() / (sd / np.sqrt(n)) if (n > 1 and sd > 0) else np.nan,
        '期末净值': nav.iloc[-1] if n else np.nan,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    t0 = time.time()
    print('== 05 因子分析与分层回测 ==')

    # ---- 1. 面板与月末对齐 -------------------------------------------------
    close = load_close_panel(DATA_DIR)
    me_all = get_month_end_days(close)                      # 全部月份（含预热）
    me_an = me_all[me_all.index >= pd.Period(ANALYSIS_START)]  # 分析期月末
    print(f'面板：{close.shape[1]} 只股票 × {close.shape[0]} 个交易日 '
          f'（{close.index[0].date()} ~ {close.index[-1].date()}），'
          f'分析期 {me_an.index[0]} ~ {me_an.index[-1]}（{len(me_an)} 个月末）')

    # ---- 2. 因子构造 + 截面预处理 -------------------------------------------
    factors = build_factors(close, me_all, me_an)           # 原始因子
    fz = {name: fac.apply(cross_section_process, axis=1)    # 处理后因子（逐月截面）
          for name, fac in factors.items()}

    # ---- 3. T+1 持有收益 + IC 检验 -------------------------------------------
    hold = build_hold_returns(close, me_all, me_an)
    ic = pd.DataFrame({name: pd.Series({t: monthly_ic(fz[name].loc[t], hold.loc[t])
                                        for t in fz[name].index}).dropna()
                       for name in FACTOR_NAMES})
    stats = ic_stats_table(ic)
    stats.index.name = '因子'
    stats.round(4).to_csv(os.path.join(OUT_DIR, 'ic_stats.csv'), encoding='utf-8-sig')
    print('\n【IC 统计】（逐月 Spearman RankIC）')
    print(stats.round(4).to_string())

    # 眼见为实：去极值前后 Pearson IC 对比（极端值如何"绑架"普通相关系数）
    print('\n【去极值演示】Pearson IC 均值：原始因子 vs 处理后因子')
    for name in FACTOR_NAMES:
        raw = pd.Series({t: monthly_ic(factors[name].loc[t], hold.loc[t], 'pearson')
                         for t in factors[name].index}).dropna()
        proc = pd.Series({t: monthly_ic(fz[name].loc[t], hold.loc[t], 'pearson')
                          for t in fz[name].index}).dropna()
        print(f'  {name}: 原始 {raw.mean():+.4f} → 处理后 {proc.mean():+.4f}')

    # ---- 4. 分层回测 ----------------------------------------------------------
    layers = {name: layer_backtest(fz[name], hold) for name in FACTOR_NAMES}
    layer_frames = []
    for name in FACTOR_NAMES:
        tmp = layers[name].copy()
        tmp.insert(0, '因子', name)
        tmp.insert(0, '月份', tmp.index.strftime('%Y-%m-%d'))
        layer_frames.append(tmp)
    layer_all = pd.concat(layer_frames, ignore_index=True)
    layer_all.round(6).to_csv(os.path.join(OUT_DIR, 'layer_returns.csv'),
                              index=False, encoding='utf-8-sig')

    print('\n【分层平均月收益（%）】Q1=因子值最低 … Q5=因子值最高')
    for name in FACTOR_NAMES:
        m = layers[name][[f'Q{i + 1}' for i in range(N_LAYERS)]].mean() * 100
        print(f'  {name}: ' + '  '.join(f'Q{i + 1}={m[f"Q{i + 1}"]:+.2f}'
                                        for i in range(N_LAYERS)))

    print('\n【分层明细】（月均收益 | 年化收益 | 期末净值）')
    for name in FACTOR_NAMES:
        parts = []
        for i in range(N_LAYERS):
            st = perf_stats(layers[name][f'Q{i + 1}'], rf_annual=0.02)  # 纯多头层：统一口径 rf=2% 年化（月扣 0.02/12）
            parts.append(f'Q{i + 1} {st["月均收益"]:+.2%}/{st["年化收益"]:+.2%}/{st["期末净值"]:.2f}')
        print(f'  {name}: ' + '  '.join(parts))

    print('\n【多空 Q5-Q1 表现】')
    for name in FACTOR_NAMES:
        st = perf_stats(layers[name]['多空Q5减Q1'])
        print(f'  {name}: 月均 {st["月均收益"]:+.2%} | 年化 {st["年化收益"]:+.2%} | '
              f'年化波动 {st["年化波动"]:.2%} | 夏普 {st["夏普"]:+.2f} | t值 {st["t值"]:+.2f}')

    # ---- 5. 因子相关性（多因子为什么要低相关） ---------------------------------
    pooled = pd.concat([fz[name].stack() for name in FACTOR_NAMES], axis=1)
    pooled.columns = FACTOR_NAMES
    corr = pooled.corr(method='spearman')
    print('\n【因子截面值 Spearman 相关】（全部分析月份合并）')
    print(corr.round(3).to_string())

    # ============================================================
    # 绘图
    # ============================================================
    # 图1：三因子累计 RankIC
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in FACTOR_NAMES:
        s = ic[name].dropna()
        sd = s.std(ddof=1)
        ax.plot(s.index, s.cumsum(),
                label=f'{name}：IC均值={s.mean():+.3f}，ICIR={s.mean() / sd:+.2f}')
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.set_title('三因子累计 RankIC（月末截面 vs 下月收益，Spearman 秩相关）')
    ax.set_xlabel('月末日期')
    ax.set_ylabel('累计 RankIC')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_三因子累计RankIC.png'), dpi=150)
    plt.close(fig)

    # 图2：动量因子分层净值 + 多空对冲
    lr = layers['动量']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ['#1f77b4', '#2ca02c', '#7f7f7f', '#ff7f0e', '#d62728']
    for i in range(N_LAYERS):
        nav = (1.0 + lr[f'Q{i + 1}']).cumprod()
        tag = '（因子值最低）' if i == 0 else ('（因子值最高）' if i == N_LAYERS - 1 else '')
        ax.plot(nav.index, nav, color=colors[i], lw=1.6,
                label=f'Q{i + 1}{tag} 期末净值 {nav.iloc[-1]:.2f}')
    ls_nav = (1.0 + lr['多空Q5减Q1']).cumprod()
    ax.plot(ls_nav.index, ls_nav, color='black', ls='--', lw=2.0,
            label=f'多空 Q5-Q1  期末净值 {ls_nav.iloc[-1]:.2f}')
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_title('动量因子分层回测：Q1~Q5 等权累计净值与多空对冲\n'
                 '（月末调仓，T+1 收盘进场，未扣交易成本）')
    ax.set_xlabel('月末日期')
    ax.set_ylabel('累计净值（起点=1）')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_动量因子分层净值.png'), dpi=150)
    plt.close(fig)

    # 图3：三个因子分层平均月收益柱状图（看单调性）
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax_, name in zip(axes, FACTOR_NAMES):
        mean_r = layers[name][[f'Q{i + 1}' for i in range(N_LAYERS)]].mean() * 100
        bars = ax_.bar(range(1, N_LAYERS + 1), mean_r,
                       color='#4C72B0', edgecolor='black', lw=0.6)
        for b, v in zip(bars, mean_r):
            ax_.annotate(f'{v:+.2f}%', (b.get_x() + b.get_width() / 2, v),
                         ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
        ax_.axhline(0, color='gray', lw=0.8)
        ax_.set_title(f'{name}因子')
        ax_.set_xticks(range(1, N_LAYERS + 1))
        ax_.set_xticklabels([f'Q{i}' for i in range(1, N_LAYERS + 1)])
        ax_.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('平均月收益（%）')
    fig.suptitle('分层平均月收益：检验 Q1→Q5 是否单调（Q1=因子值最低，Q5=最高）', y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_分层平均月收益.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # 图4：因子相关性热力图
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(FACTOR_NAMES)))
    ax.set_xticklabels(FACTOR_NAMES)
    ax.set_yticks(range(len(FACTOR_NAMES)))
    ax.set_yticklabels(FACTOR_NAMES)
    for i in range(len(FACTOR_NAMES)):
        for j in range(len(FACTOR_NAMES)):
            v = corr.values[i, j]
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                    color='white' if abs(v) > 0.6 else 'black', fontsize=12)
    fig.colorbar(im, ax=ax, label='Spearman 相关系数')
    ax.set_title('因子截面值相关性热力图\n（全部分析月份的截面值合并计算）')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_因子相关性热力图.png'), dpi=150)
    plt.close(fig)

    print(f'\n输出目录：{OUT_DIR}')
    print(f'运行耗时 {time.time() - t0:.1f} 秒')


if __name__ == '__main__':
    main()
