# -*- coding: utf-8 -*-
"""
15_行业动量轮动 —— 经典动量轮动框架（教学版）
====================================================
思路：每月末给 10 个行业打"过去 N 个交易日的动量分"，买入最强的 K=3 个行业，
等权持有，下一个月末再平衡；与两个基准对比：10行业等权组合、沪深300。

本脚本单文件自包含，运行 < 2 分钟。随机种子 42（本课流程是确定性的，
设种子是为了满足课程规范，并保证任何随机兜底逻辑可复现）。

流程：
  1) 读入 10 个真实行业CSV + 沪深300，做"基期跳变"清洗（单日|收益|>30% 视为毛刺置 0）
  2) 画清洗前后单日收益分布对比图
  3) 动量打分（N=20 / N=60 交易日），每月末排序
  4) 回测：持有最强 K=3 行业等权、月度再平衡（权重逐日漂移）
  5) 输出净值对比、持仓热力图、分年收益、动量窗口敏感性、绩效表 CSV
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

# ---------- 全局规范 ----------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'cache')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

JUMP_TH = 0.30        # 跳变判定阈值：单日收益绝对值 > 30%（A股指数单日不可能有这种波动）
K_HOLD = 3            # 每月持有行业数
MAIN_WINDOWS = [20, 60]      # 主对比的两种动量窗口（交易日）：约1个月 / 约3个月
SENS_WINDOWS = [10, 20, 40, 60, 120]  # 敏感性测试的窗口网格
ANN = 252             # 年化系数（国际惯例 252 个交易日）
SYNTHETIC = False     # 若真实数据缺失则转为合成数据并置 True


# =====================================================================
# 1. 数据加载与清洗
# =====================================================================
def load_close(path):
    """读取单个CSV → 以日期为索引的收盘价 Series（剔除收盘价缺失/重复日期）。"""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close']).drop_duplicates(subset='date').sort_values('date')
    return df.set_index('date')['close'].astype(float)


def make_synthetic_data():
    """数据缺失兜底：用 rng(42) 造 10 个行业 + 基准的合成收盘价（仅教学演示）。"""
    global SYNTHETIC
    SYNTHETIC = True
    print('[警告] 未找到真实行业数据，改用 rng(42) 合成数据（结果仅演示流程）')
    dates = pd.bdate_range('2021-01-04', periods=900)
    cols = {}
    for i in range(10):
        mu = 0.0002 + 0.0001 * (i % 3)
        sig = 0.012 + 0.004 * (i % 4)
        r = rng.normal(mu, sig, len(dates))
        cols[f'合成行业{i + 1}'] = pd.Series(100 * np.cumprod(1 + r), index=dates)
    cols['沪深300'] = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0002, 0.010, len(dates))), index=dates)
    return cols


def load_all_data():
    """加载 10 个行业 + 沪深300，拼成收盘价宽表（索引=各表交易日的并集）。"""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'board_*.csv')))
    if len(files) < 10 or not os.path.exists(os.path.join(DATA_DIR, 'index_000300_沪深300.csv')):
        cols = make_synthetic_data()
        close = pd.DataFrame(cols)
        ind_names = [c for c in close.columns if c != '沪深300']
        return close, close['沪深300'], ind_names
    cols = {}
    for f in files:
        name = os.path.basename(f)[len('board_'):-4]  # board_银行.csv -> 银行
        cols[name] = load_close(f)
    close = pd.DataFrame(cols)                       # 列=行业，索引=交易日并集
    idx300 = load_close(os.path.join(DATA_DIR, 'index_000300_沪深300.csv'))
    return close, idx300, list(close.columns)


def board_jump_stats(close_raw, ind_names):
    """
    在每个行业"自身日历"上检测基期跳变（用于统计与图01）：
      - 各行业起始日期不同、个别日期缺失，若在并集日历上算收益，
        缺失日次日会被算成 NaN 而漏检跳变；因此逐行业用自己的日期序列检测。
      - 规则：单日收益绝对值 > 30% 判为数据源基期切换毛刺，该日收益置 0。
        依据：A股个股有 10% 涨跌停限制，行业指数单日 ±30% 在现实中不存在。
    返回：每行业原始收益 dict、清洗后收益 dict、跳变数量 Series。
    """
    ret_raw_d, ret_clean_d, cnt = {}, {}, {}
    for n in ind_names:
        s = close_raw[n].dropna()
        r = s.pct_change()
        jump = r.abs() > JUMP_TH
        ret_raw_d[n] = r
        ret_clean_d[n] = r.mask(jump, 0.0)
        cnt[n] = int(jump.sum())
    return ret_raw_d, ret_clean_d, pd.Series(cnt)


def build_common_panel(close_raw, ind_names):
    """
    构建"共同日历"研究面板：只保留 10 个行业都有收盘价的交易日。
      - 行业指数发布时间不一（银行2018-01起、电力2021-12起），
        共同起点由最晚发布的行业决定（本数据为 2021-12-29）。
      - 在该矩形面板上计算日收益，单日|收益|>30% 置 0（同一跳变规则）；
        个别行业缺口造成的"跨缺口收益"若 >30% 同样被该规则兜住。
    返回：ret_panel 清洗后日收益面板（首行NaN按0）、jump_cnt 共同期跳变数。
    """
    common_index = close_raw[ind_names].dropna(how='any').index
    panel = close_raw.loc[common_index, ind_names]
    r = panel.pct_change()
    jump = r.abs() > JUMP_TH
    ret_panel = r.mask(jump, 0.0).fillna(0.0)   # 首行无收益，按 0
    return ret_panel, int(jump.sum().sum()), common_index


# =====================================================================
# 2. 绩效统计与回测引擎
# =====================================================================
def perf_stats(ser, turnover=None):
    """输入日收益 Series（可选单边换手 Series），输出绩效字典。夏普统一 rf=2% 年化：(日均收益 − 0.02/252)/日波动×√252（全部组合为纯多头）。"""
    nav = (1.0 + ser).cumprod()
    years = max((ser.index[-1] - ser.index[0]).days / 365.25, 1e-9)
    total = nav.iloc[-1] - 1.0
    ann = (1.0 + total) ** (1.0 / years) - 1.0
    vol = ser.std(ddof=1) * np.sqrt(ANN)
    sharpe = ((ser.mean() - 0.02 / ANN) / ser.std(ddof=1) * np.sqrt(ANN)) if ser.std(ddof=1) > 0 else np.nan
    max_dd = float((nav / nav.cummax() - 1.0).min())
    tu_ann = float(turnover.sum() / years) if turnover is not None and len(turnover) else 0.0
    return {'起始日': ser.index[0], '截止日': ser.index[-1],
            '累计收益': total, '年化收益': ann, '年化波动': vol,
            '夏普比率': sharpe, '最大回撤': max_dd, '年均单边换手': tu_ann}


def yearly_returns(ser):
    """按自然年复利，输出分年收益 Series。"""
    return (1.0 + ser).groupby(ser.index.year).prod() - 1.0


def backtest(weights_by_date, ret_panel, all_days):
    """
    月度再平衡回测引擎（权重逐日漂移，无交易成本）。
      weights_by_date: {调仓日Timestamp: 目标权重ndarray}，调仓在当日收盘设定、次日起生效
      ret_panel: 日收益 DataFrame（行业×交易日），缺失已按 0 处理
      all_days: 回测交易日索引（严格晚于首个调仓日）
    返回：日收益 Series、单边换手 Series（0.5*Σ|Δw|，即换掉组合的一半）。
    """
    w = np.asarray(weights_by_date[sorted(weights_by_date)[0]], dtype=float)  # 首个调仓日收盘建仓
    port_r, tu_dates, tu_vals = [], [], []
    for t in all_days:
        r = ret_panel.loc[t].to_numpy(dtype=float)
        pr = float(np.dot(w, r))                  # 当日组合收益（用昨日收盘权重）
        port_r.append(pr)
        w = w * (1.0 + r) / (1.0 + pr)            # 权重随行情漂移到当日收盘
        if t in weights_by_date:                  # 当日为调仓日：收盘换仓
            w_new = np.asarray(weights_by_date[t], dtype=float)
            tu_dates.append(t)
            tu_vals.append(0.5 * float(np.abs(w_new - w).sum()))
            w = w_new.copy()
    return pd.Series(port_r, index=all_days), pd.Series(tu_vals, index=tu_dates)


def month_end_dates(index):
    """取每个自然月最后一个交易日（DatetimeIndex → 按月取最后一天的 Series）。"""
    s = pd.Series(index, index=index)
    return s.groupby(index.to_period('M')).max()


def topk_weight_targets(mom, month_ends, k):
    """
    动量打分 → 每月末目标权重：
      对每个月末，取动量分最高的 k 个行业、各 1/k 等权。
      只要该月末 10 个行业动量分齐全才作为调仓日（保证"全部行业可打分"才轮动）。
    返回 {调仓日: 权重ndarray} 与 {调仓日: 持有行业列表}。
    """
    targets, holdings = {}, {}
    for d in month_ends:
        s = mom.loc[d].dropna()
        if len(s) < mom.shape[1]:        # 有行业动量分缺失（历史不足 N 日），该月不调仓
            continue
        top = s.sort_values(ascending=False, kind='mergesort').index[:k]  # mergesort保证并列时结果确定
        w = np.zeros(mom.shape[1])
        w[[mom.columns.get_loc(c) for c in top]] = 1.0 / k
        targets[d] = w
        holdings[d] = list(top)
    return targets, holdings


# =====================================================================
# 3. 绘图函数
# =====================================================================
def fig01_return_dist(ret_raw_d, ret_clean_d, jump_cnt, n_jump_total):
    """清洗前后单日收益分布对比：左图全量程对数y轴（突出毛刺尖峰），右图放大主体。"""
    raw_pool = pd.concat(ret_raw_d.values()).dropna()
    cln_pool = pd.concat(ret_clean_d.values()).dropna()
    jump_vals = raw_pool[raw_pool.abs() > JUMP_TH]          # 被清洗的跳变点
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    # 左图：全量程、计数、对数y轴 —— 让29个毛刺点可见（正常日上万天、毛刺只有29天）
    bins_all = np.linspace(-0.75, 2.1, 90)
    axes[0].hist(raw_pool.clip(-0.75, 2.1), bins=bins_all, color='crimson',
                 alpha=0.65, label='清洗前')
    axes[0].hist(cln_pool.clip(-0.75, 2.1), bins=bins_all, color='steelblue',
                 alpha=0.65, label='清洗后（跳变日收益置0）')
    axes[0].plot(jump_vals, np.full(len(jump_vals), 0.65), '|', color='darkred',
                 markersize=11, label=f'被清洗的跳变点（{n_jump_total} 个）')
    axes[0].set_yscale('log')
    axes[0].set_ylim(0.5, None)
    axes[0].annotate('基期跳变尖刺：\n±30%~±200%，现实中不可能出现', xy=(1.75, 2),
                     xytext=(0.75, 300), fontsize=9, color='darkred',
                     arrowprops=dict(arrowstyle='->', color='darkred'))
    axes[0].set_title('全量程（对数纵轴，可见尾部毛刺尖峰）', fontsize=11)
    axes[0].set_xlabel('单日收益')
    axes[0].set_ylabel('天数（对数刻度）')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25, which='both')
    # 右图：放大主体区间 —— 清洗前后主体完全重合，说明清洗只动尾部
    bins_zoom = np.linspace(-0.12, 0.12, 97)
    axes[1].hist(raw_pool, bins=bins_zoom, density=True, alpha=0.55,
                 color='crimson', label='清洗前')
    axes[1].hist(cln_pool, bins=bins_zoom, density=True, alpha=0.60,
                 color='steelblue', label='清洗后')
    axes[1].set_title('放大主体区间 [-12%, +12%]（两分布几乎重合）', fontsize=11)
    axes[1].set_xlabel('单日收益')
    axes[1].set_ylabel('概率密度')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    fig.suptitle('图01 数据清洗前后：10个行业单日收益分布对比（各行业自身日历全样本，毛刺=基期跳变）',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT_DIR, '01_清洗前后收益分布对比.png'), dpi=150)
    plt.close(fig)


def fig02_nav_curve(navs, stats_map):
    """净值对比曲线（图例含年化/回撤/夏普）。"""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = {'动量N=20_K3': '#d62728', '动量N=60_K3': '#1f77b4',
              '10行业等权基准': '#7f7f7f', '沪深300': '#e6a100'}
    for name, nav in navs.items():
        st = stats_map[name]
        ax.plot(nav.index, nav.values, lw=1.8, color=colors.get(name),
                label=f"{name}（年化 {st['年化收益']:+.1%}，回撤 {st['最大回撤']:.1%}，夏普 {st['夏普比率']:.2f}）")
    ax.axhline(1.0, color='k', lw=0.6, ls='--', alpha=0.5)
    ax.set_title('图02 行业动量轮动净值对比：持有过去N日最强3行业，月度再平衡（无交易成本）', fontsize=12)
    ax.set_xlabel('日期')
    ax.set_ylabel('净值（期初=1）')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_动量轮动净值对比.png'), dpi=150)
    plt.close(fig)


def fig03_holdings_heatmap(hold_map, ind_names):
    """持仓热力图：行=调仓月份，列=行业，色块=该月持有（两窗口上下对照）。"""
    fig, axes = plt.subplots(len(hold_map), 1,
                             figsize=(10, 2.6 + 3.6 * len(hold_map)), squeeze=False)
    for ax, (name, holdings) in zip(axes.ravel(), hold_map.items()):
        dates = sorted(holdings)
        mat = np.zeros((len(dates), len(ind_names)))
        for i, d in enumerate(dates):
            for c in holdings[d]:
                mat[i, ind_names.index(c)] = 1
        ax.imshow(mat, aspect='auto', cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(len(ind_names)))
        ax.set_xticklabels(ind_names, rotation=30, ha='right', fontsize=9)
        step = max(2, len(dates) // 28)
        ax.set_yticks(range(0, len(dates), step))
        ax.set_yticklabels([dates[i].strftime('%Y-%m') for i in range(0, len(dates), step)], fontsize=8)
        ax.set_title(f'{name}：每月持有的 {len(holdings[dates[0]])} 个行业（共 {len(dates)} 次调仓）',
                     fontsize=11)
        ax.set_xticks(np.arange(-0.5, len(ind_names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(dates), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.6)
        ax.tick_params(which='minor', length=0)
    fig.suptitle('图03 持仓热力图：轮动轨迹（时间×行业，色块=该月持有）', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, '03_持仓热力图.png'), dpi=150)
    plt.close(fig)


def fig04_yearly_bars(yearly_df):
    """分年收益分组柱状图：策略 vs 两个基准。"""
    fig, ax = plt.subplots(figsize=(11, 5))
    years = yearly_df.index.astype(str)
    x = np.arange(len(years))
    width = 0.2
    items = [('动量N=20_K3', '#d62728'), ('动量N=60_K3', '#1f77b4'),
             ('10行业等权基准', '#7f7f7f'), ('沪深300', '#e6a100')]
    for i, (name, color) in enumerate(items):
        vals = yearly_df[name].values * 100
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=color, alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (1.2 if v >= 0 else -3.6),
                    f'{v:.0f}', ha='center', fontsize=7.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel('年度收益 (%)')
    ax.set_title('图04 分年收益对照（2026年为不完整年度，截至9月中旬）', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_分年收益对比.png'), dpi=150)
    plt.close(fig)


def fig05_sensitivity(sens_df, ew_ann):
    """动量窗口敏感性：不同 N 的年化收益（柱）与夏普（线），统一在同一窗口评估。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(sens_df))
    colors = ['#d62728' if n in MAIN_WINDOWS else '#9ebcda' for n in sens_df['N']]
    bars = ax.bar(xs, sens_df['年化收益'] * 100, color=colors, alpha=0.9, width=0.55)
    vals = sens_df['年化收益'] * 100
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.35 if v >= 0 else -0.55),
                f'{v:+.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    ax.set_ylim(min(vals) - 1.6, max(vals) + 1.6)   # 留出标签空间
    ax.axhline(ew_ann * 100, color='#7f7f7f', ls='--', lw=1.2,
               label=f'10行业等权基准年化 {ew_ann:+.1%}')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'N={n}' for n in sens_df['N']])
    ax.set_ylabel('年化收益 (%)')
    ax.set_xlabel('动量回看窗口 N（交易日）')
    ax2 = ax.twinx()
    ax2.plot(xs, sens_df['夏普比率'], 'o-', color='#2ca02c', lw=1.6, label='夏普比率（右轴）')
    ax2.set_ylabel('夏普比率')
    ax.set_title('图05 动量窗口敏感性：各N统一从同一月末起算以保证可比（红柱=主策略窗口N=20/60）',
                 fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '05_动量窗口敏感性.png'), dpi=150)
    plt.close(fig)


# =====================================================================
# 4. 主流程
# =====================================================================
def main():
    # ---------- 4.1 数据加载与清洗 ----------
    close_raw, idx300, ind_names = load_all_data()
    ret_raw_d, ret_clean_d, jump_cnt = board_jump_stats(close_raw, ind_names)
    n_jump_total = int(jump_cnt.sum())

    print('=' * 72)
    print('步骤1 数据清洗：单日收益绝对值 > 30% 判为基期跳变，该日收益置 0')
    print('各行业跳变点数量（各行业自身日历、全样本）：')
    for n in ind_names:
        print(f'  {n}: {int(jump_cnt[n])} 个')
    print(f'  合计: {n_jump_total} 个（全为数据源基期切换毛刺，非真实行情）')

    # 共同日历研究面板：10个行业全有数据的交易日
    ret_panel, n_jump_common, common_index = build_common_panel(close_raw, ind_names)
    common_start = common_index[0]
    print(f'共同研究期（10个行业全有数据）: {common_start.date()} ~ {common_index[-1].date()}，'
          f'共 {len(common_index)} 个交易日')
    print(f'共同期内清洗的跳变点: {n_jump_common} 个 → 已在动量打分与回测中置 0')
    print('说明：行业指数发布时间不一（银行/证券等2018起、电力2021-12起），样本期偏短，'
          '轮动结论仅为教学演示。')

    # ---------- 4.2 图01：清洗前后收益分布 ----------
    fig01_return_dist(ret_raw_d, ret_clean_d, jump_cnt, n_jump_total)
    print('已输出: outputs/01_清洗前后收益分布对比.png')

    # ---------- 4.3 动量打分与调仓日 ----------
    month_ends = list(month_end_dates(common_index))
    logr = np.log1p(ret_panel)

    def momentum(n):
        """过去 n 个交易日累计收益（对数收益滚动求和再还原）；每月末给 10 个行业打分。"""
        return np.expm1(logr.rolling(n, min_periods=n).sum())

    mom = {n: momentum(n) for n in SENS_WINDOWS}
    # 回测起点：首个"10个行业 N=60 动量分齐全"的月末；两种窗口统一从该日起算，保证可比
    start_date = min(d for d in month_ends if mom[MAIN_WINDOWS[1]].loc[d].notna().all())
    bt_days = common_index[common_index > start_date]
    years_bt = (bt_days[-1] - bt_days[0]).days / 365.25
    print(f'回测起点（首个N=60可打分的月末）: {start_date.date()}，'
          f'区间 {bt_days[0].date()} ~ {bt_days[-1].date()}，共 {len(bt_days)} 个交易日 / {years_bt:.1f} 年')

    # ---------- 4.4 回测：两种动量窗口 + 两个基准 ----------
    stats_map, navs, yearly_map, hold_map, ser_map = {}, {}, {}, {}, {}
    for n in MAIN_WINDOWS:
        targets, holdings = topk_weight_targets(mom[n], month_ends, K_HOLD)
        targets = {d: w for d, w in targets.items() if d >= start_date}
        holdings = {d: h for d, h in holdings.items() if d >= start_date}
        ser, tu = backtest(targets, ret_panel, bt_days)
        name = f'动量N={n}_K{K_HOLD}'
        stats_map[name] = perf_stats(ser, tu)
        navs[name] = (1.0 + ser).cumprod()
        yearly_map[name] = yearly_returns(ser)
        hold_map[name] = holdings
        ser_map[name] = ser

    # 基准1：10行业等权、同样月度再平衡
    ew_targets = {d: np.full(len(ind_names), 1.0 / len(ind_names))
                  for d in month_ends if d >= start_date}
    ew_ser, ew_tu = backtest(ew_targets, ret_panel, bt_days)
    stats_map['10行业等权基准'] = perf_stats(ew_ser, ew_tu)
    navs['10行业等权基准'] = (1.0 + ew_ser).cumprod()
    yearly_map['10行业等权基准'] = yearly_returns(ew_ser)

    # 基准2：沪深300 买入持有（同一区间；按其自身日历算收益再对齐，缺失日按0）
    r300 = idx300.pct_change().reindex(bt_days).fillna(0.0)
    stats_map['沪深300'] = perf_stats(r300)
    navs['沪深300'] = (1.0 + r300).cumprod()
    yearly_map['沪深300'] = yearly_returns(r300)

    # ---------- 4.5 图02 净值对比 ----------
    fig02_nav_curve(navs, stats_map)
    print('已输出: outputs/02_动量轮动净值对比.png')

    # ---------- 4.6 图03 持仓热力图 ----------
    fig03_holdings_heatmap(hold_map, ind_names)
    print('已输出: outputs/03_持仓热力图.png')

    # ---------- 4.7 绩效表 outputs/portfolios.csv（含月胜率） ----------
    order = [f'动量N={n}_K{K_HOLD}' for n in MAIN_WINDOWS] + ['10行业等权基准', '沪深300']
    port_df = pd.DataFrame(stats_map).T.loc[order]
    # 月胜率：按月复利后，策略月收益 > 等权基准月收益 的月份占比
    ew_m = (1.0 + ew_ser).groupby(ew_ser.index.to_period('M')).prod() - 1.0
    win_rate = {}
    for name, ser in ser_map.items():
        m = (1.0 + ser).groupby(ser.index.to_period('M')).prod() - 1.0
        win_rate[name] = float((m > ew_m).mean())
    port_df['月胜率vs等权'] = pd.Series(win_rate)

    show_cols = ['起始日', '截止日', '累计收益', '年化收益', '年化波动',
                 '夏普比率', '最大回撤', '年均单边换手', '月胜率vs等权']
    out_port = port_df[show_cols].copy()
    out_port['起始日'] = pd.to_datetime(out_port['起始日']).dt.strftime('%Y-%m-%d')
    out_port['截止日'] = pd.to_datetime(out_port['截止日']).dt.strftime('%Y-%m-%d')
    fmt = lambda x: '' if pd.isna(x) else f'{x:.4f}'   # 不适用（如基准的胜率）留空
    for c in ['累计收益', '年化收益', '年化波动', '最大回撤', '年均单边换手', '月胜率vs等权']:
        out_port[c] = out_port[c].astype(float).map(fmt)
    out_port['夏普比率'] = out_port['夏普比率'].astype(float).map(fmt)
    out_port.index.name = '策略'
    out_port.to_csv(os.path.join(OUT_DIR, 'portfolios.csv'), encoding='utf-8-sig')
    print('\n===== 绩效表（无交易成本，年化系数252，换手为单边） =====')
    print(port_df[show_cols].to_string(float_format=lambda x: f'{x:.4f}'))
    print('已输出: outputs/portfolios.csv')

    # ---------- 4.8 分年收益 outputs/yearly_returns.csv ----------
    yearly_df = pd.DataFrame(yearly_map).sort_index()
    yearly_df['N20超额vs等权'] = yearly_df[f'动量N=20_K{K_HOLD}'] - yearly_df['10行业等权基准']
    yearly_df['N60超额vs等权'] = yearly_df[f'动量N=60_K{K_HOLD}'] - yearly_df['10行业等权基准']
    yearly_df.index.name = '年份'
    yearly_df.to_csv(os.path.join(OUT_DIR, 'yearly_returns.csv'), encoding='utf-8-sig')
    print('\n===== 分年收益对照（2026年为不完整年度，截至9月中旬） =====')
    print(yearly_df.to_string(float_format=lambda x: f'{x:+.2%}'))
    print('已输出: outputs/yearly_returns.csv')

    # ---------- 4.9 图04 分年收益柱状图 ----------
    fig04_yearly_bars(yearly_df[[f'动量N=20_K{K_HOLD}', f'动量N=60_K{K_HOLD}',
                                 '10行业等权基准', '沪深300']])
    print('已输出: outputs/04_分年收益对比.png')

    # ---------- 4.10 图05 动量窗口敏感性（统一从 N=120 可打分的月末起算，保证可比） ----------
    sens_start = min(d for d in month_ends if mom[max(SENS_WINDOWS)].loc[d].notna().all())
    sens_days = common_index[common_index > sens_start]
    rows = []
    for n in SENS_WINDOWS:
        targets, _ = topk_weight_targets(mom[n], month_ends, K_HOLD)
        targets = {d: w for d, w in targets.items() if d >= sens_start}
        ser, tu = backtest(targets, ret_panel, sens_days)
        st = perf_stats(ser, tu)
        rows.append({'N': n, '年化收益': st['年化收益'], '夏普比率': st['夏普比率'],
                     '最大回撤': st['最大回撤']})
    sens_df = pd.DataFrame(rows)
    # 参考基准：等权组合在"敏感性同一窗口"上的年化（保证与柱子可比）
    ew_targets_s = {d: np.full(len(ind_names), 1.0 / len(ind_names))
                    for d in month_ends if d >= sens_start}
    ew_ser_s, _ = backtest(ew_targets_s, ret_panel, sens_days)
    ew_ann_s = perf_stats(ew_ser_s)['年化收益']
    print(f'\n===== 动量窗口敏感性（统一自 {sens_start.date()} 起算，K={K_HOLD}；'
          f'等权基准同窗口年化 {ew_ann_s:+.2%}） =====')
    print(sens_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    fig05_sensitivity(sens_df, ew_ann_s)
    print('已输出: outputs/05_动量窗口敏感性.png')

    # ---------- 4.11 收尾摘要 ----------
    print('\n' + '=' * 72)
    print('完成。产出目录:', OUT_DIR)
    print('提示：无交易成本、月度再平衡；样本期偏短，结论仅供教学演示。')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # 课程硬性要求：任何异常都不以非0退出（打印完整堆栈便于排查）
        traceback.print_exc()
        print('[异常] 运行中断，但按课程规范以 0 退出（请检查上方堆栈修复）')
        sys.exit(0)
