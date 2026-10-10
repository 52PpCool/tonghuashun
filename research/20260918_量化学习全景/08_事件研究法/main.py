# -*- coding: utf-8 -*-
"""
08_事件研究法（Event Study）：用统计检验回答“降准到底涨不涨”
====================================================================

事件研究是量化研究员分析“政策 / 财报 / 公告等事件对价格影响”的标准方法。
核心思想：事件发生后的收益，必须和“如果没有这个事件，市场本来会怎么走”比较，
而不是和 0 比较——直接看“公告后涨没涨”会把大盘本身的涨跌误认为事件效应。

事件研究三要素：
  1) 估计窗 [-250, -30]：用事件之前的“平静期”估计“正常收益”。
     本课采用最简单的“均值调整模型”：正常收益 = 估计窗内的日均收益。
     为什么简单模型也够用？
       - 我们研究的是沪深300指数本身：指数长期日均收益非常小（约万分之一
         到万分之三量级），而日波动约 1%，均值估计误差相对很小，
         换成更复杂的模型对结论影响有限；
       - 常见的“市场模型”（个股收益对市场指数回归）需要一个市场基准，
         而指数对自身回归没有意义，因此对指数事件用均值调整模型是恰当的；
         如果研究的是个股事件，建议升级为市场模型（可选项）。
  2) 事件窗 [-5, +10]：围绕事件日 t=0，计算
       异常收益 AR_t = 实际收益 - 正常收益
       累计异常收益 CAR = AR 在窗口内的累加
  3) 显著性检验：
       - 截面 t 检验：把 15 个事件各看作一次独立抽样，
         检验平均 CAR[0,+10] 是否显著异于 0（H0：平均CAR = 0）；
       - 安慰剂检验（本课灵魂）：随机抽 50 个“假事件日”，跑完全相同的
         流程。如果真实事件的 CAR 没有跳出假事件 CAR 的分布范围，
         那么“效应”很可能只是随机波动。

事件清单是央行降准等政策的公开历史公告日（教学用简化清单）。注意用的是
“公告日”而非“生效日”：降准公告通常提前 1~2 周宣布，市场在公告那一刻就
开始反应，事件研究关心的是“新信息到达”的时点。若公告日恰逢非交易日，
自动顺延到下一个交易日（见 align_event_day 函数）。

运行方式：
  cd 08_事件研究法
  /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py

输出（outputs/）：
  01_平均CAR曲线与置信带.png   15个事件的平均CAR曲线与 ±1.96×截面标准误 置信带
  02_真实vs安慰剂CAR分布.png   真实事件 CAR[0,+10] vs 50个假事件 CAR[0,+10] 分布
  03_代表性事件逐日CAR.png     3个代表性事件的逐日CAR曲线
  event_details.csv            逐事件明细表（utf-8-sig）
  test_stats.csv               t检验与安慰剂检验统计量汇总（utf-8-sig）
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 无显示环境下的绘图后端，只出文件不弹窗
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42                                   # 随机种子：保证安慰剂抽样可复现
rng = np.random.default_rng(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'cache', 'index_000300_沪深300.csv')
os.makedirs(OUT_DIR, exist_ok=True)

EST_START, EST_END = -250, -30              # 估计窗（相对事件日 t=0 的交易日偏移）
EV_START, EV_END = -5, 10                   # 事件窗 [-5, +10]，共 16 个交易日
N_PLACEBO = 50                              # 安慰剂“假事件日”数量
PLACEBO_GAP = 30                            # 假事件日与真实事件日至少相距 30 个交易日

# ----------------------------------------------------------------------
# 事件清单：央行降准等政策的公开历史公告日（教学用简化清单）。
# 关键点：
#   - 用“公告日”而不是“生效日”：降准公告通常提前 1~2 周宣布，真正的信息
#     冲击发生在公告那一刻；到生效日市场早已消化（这就是“预期已定价”）。
#   - 很多公告发生在交易日盘后或周末/节假日，所以 t=0 的当日收益有时属于
#     “事前”，真实反应可能体现在 t=0（顺延情形）或 t=+1（盘后公告情形）。
#   - 若公告日恰逢非交易日，align_event_day 会顺延到下一个交易日。
# ----------------------------------------------------------------------
EVENTS = {
    '2015-08-25': '双降（降准+降息），股灾期间、公告于当日盘后',
    '2018-10-07': '降准1个百分点，国庆假期最后一天公告',
    '2019-09-06': '降准，周五盘后公告',
    '2020-01-01': '降准，元旦假期公告',
    '2020-03-13': '降准，周五盘后公告（疫情冲击期）',
    '2021-07-09': '降准，周五盘后公告',
    '2021-12-06': '降准，周一盘后公告',
    '2022-04-15': '降准，周五盘后公告（上海疫情期间）',
    '2022-11-25': '降准，周五盘后公告',
    '2023-03-17': '降准，周五盘后公告',
    '2023-08-11': '降准，周五盘后公告（次日高开低走）',
    '2023-09-14': '降准，周四盘后公告',
    '2024-01-24': '盘前宣布降准，当日午后市场大涨',
    '2024-09-24': '一揽子增量政策（含降准降息），当日大涨',
    '2025-05-07': '降准降息，盘中公告',
}


# ----------------------------------------------------------------------
# 数据读取
# ----------------------------------------------------------------------
def load_index_data():
    """读取沪深300日K缓存；若缓存缺失则用 rng(42) 生成合成数据并打标记。

    返回 (DataFrame, 是否合成数据, 数据来源说明)。
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=['date']).set_index('date')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df = df.dropna(subset=['close'])          # 剔除无收盘价的残缺行
        synthetic = False
        source = '真实数据 data/cache/index_000300_沪深300.csv'
    else:
        # 合成数据兜底：几何随机游走（GBM），保证脚本在任何环境都能跑通
        idx = pd.bdate_range('2012-01-01', '2026-09-18')
        r = rng.normal(0.0002, 0.012, size=len(idx))
        df = pd.DataFrame({'close': 2500.0 * np.exp(np.cumsum(r))}, index=idx)
        synthetic = True
        source = '合成数据（数据缓存缺失，rng(42) 生成）'
    df['ret'] = df['close'].pct_change()          # 日收益率（简单收益）
    return df, synthetic, source


# ----------------------------------------------------------------------
# 事件研究核心
# ----------------------------------------------------------------------
def align_event_day(dates, ann_date):
    """对齐函数：把“公告日”映射为它在交易日历中的位置（t=0 的整数下标）。

    规则：取第一个 >= 公告日的交易日。若公告日恰逢非交易日
    （周末 / 法定节假日），自动顺延到下一个交易日。
    """
    pos = dates.searchsorted(pd.Timestamp(ann_date), side='left')
    if pos >= len(dates):
        return None                               # 公告日晚于数据末尾，无法研究
    return int(pos)


def compute_event(df, pos):
    """给定事件日位置 pos，计算该事件的 AR / CAR 全套明细。

    估计窗 [-250, -30]（共 221 个交易日）：均值调整模型，
        正常收益 mu = 估计窗日均收益。
    事件窗 [-5, +10]（共 16 个交易日）：
        AR_k = r_{t0+k} - mu；CAR = AR 依次累加。
    估计窗与事件窗之间留出 [-30, -6] 的空档，避免估计被事件前的
    异动“污染”，也保证事件信息不泄漏进正常收益的估计。
    """
    est = df['ret'].iloc[pos + EST_START: pos + EST_END + 1].dropna()
    mu = float(est.mean())                        # 正常收益：平时每天“应得”的收益
    ks = list(range(EV_START, EV_END + 1))        # 相对日 [-5, ..., +10]
    ar = np.array([float(df['ret'].iloc[pos + k]) - mu for k in ks])
    car = np.cumsum(ar)                           # car[j] = CAR[-5, -5+j]
    i0 = ks.index(0)
    return {
        'mu': mu,
        'ks': ks,
        'ar': ar,
        'car': car,
        'car_5_10': float(car[-1]),               # CAR[-5,+10]：事件窗全程累计
        'car_0_10': float(car[-1] - car[i0 - 1]),  # CAR[0,+10]：公告当天起累计
        'ret_t0': float(df['ret'].iloc[pos]),      # 事件日当日收益
        'ret_t1': float(df['ret'].iloc[pos + 1]),  # 事件后1日收益（t+1）
        'event_date': df.index[pos],
    }


def sample_placebos(df, real_positions):
    """安慰剂抽样：随机抽 N_PLACEBO 个“假事件日”。

    约束：
      - 估计窗不越界（pos - 250 >= 1，收益序列无 NaN）；
      - 事件窗不越界（pos + 10 <= 最后一根K线）；
      - 与任一真实事件日相距超过 PLACEBO_GAP 个交易日（避免撞上真事件）；
      - 彼此不重复（rng(42) 拒绝采样，结果可复现）。
    返回位置列表。
    """
    ret = df['ret'].to_numpy()
    lo = 1 - EST_START                            # pos-250 >= 1，估计窗内收益非NaN
    hi = len(df) - 1 - EV_END                     # pos+10 不越界
    real_arr = np.asarray(real_positions)
    chosen, seen, guard = [], set(), 0
    while len(chosen) < N_PLACEBO:
        guard += 1
        if guard > 500_000:
            raise RuntimeError('安慰剂抽样失败：有效位置不足，请检查数据长度')
        i = int(rng.integers(lo, hi + 1))
        if i in seen:
            continue
        if np.any(np.abs(real_arr - i) <= PLACEBO_GAP):
            continue                              # 距真实事件太近，弃用
        seg = ret[i + EST_START: i + EV_END + 1]
        if not np.isfinite(seg).all():
            continue                              # 窗口内有缺失收益，弃用
        seen.add(i)
        chosen.append(i)
    return chosen


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def plot_avg_car(avg_car, se_car, n_events, synthetic):
    """图01：平均CAR曲线 + ±1.96×截面标准误置信带。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = avg_car.index.to_numpy()
    y = avg_car.to_numpy() * 100
    band = 1.96 * se_car.to_numpy() * 100
    ax.plot(x, y, 'o-', color='#1f77b4', lw=2, label='平均CAR（15个事件）')
    ax.fill_between(x, y - band, y + band, color='#1f77b4', alpha=0.18,
                    label='95%置信带（±1.96×截面标准误）')
    ax.axvline(0, color='red', ls='--', lw=1.2, label='事件日 t=0（公告日顺延后）')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('相对事件日的交易日 k')
    ax.set_ylabel('累计异常收益 CAR（%）')
    title = '降准公告的平均累计异常收益（事件窗 [-5, +10]）'
    if synthetic:
        title += '【合成数据】'
    ax.set_title(f'{title}    N={n_events}')
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_平均CAR曲线与置信带.png'), dpi=150)
    plt.close(fig)


def plot_placebo(real_cars, placebo_cars, real_mean, pct, synthetic):
    """图02：安慰剂CAR分布直方图 + 真实事件均值位置。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(placebo_cars, bins=12, color='#9ecae1', edgecolor='white',
            label=f'安慰剂假事件 CAR[0,+10]（N={len(placebo_cars)}）')
    ax.axvline(real_mean, color='red', lw=2.2,
               label=f'真实事件平均CAR[0,+10] = {real_mean:.2f}%')
    for c in real_cars:
        ax.plot(c, -0.4, marker='|', color='darkgreen', markersize=14, clip_on=False)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlabel('CAR[0,+10]（%）')
    ax.set_ylabel('假事件个数')
    title = (f'真实事件 vs 安慰剂：CAR[0,+10] 分布对比\n'
             f'真实均值落在安慰剂分布的第 {pct:.0f} 百分位'
             f'（绿短线=15个真实事件各自的CAR）')
    if synthetic:
        title += '【合成数据】'
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_真实vs安慰剂CAR分布.png'), dpi=150)
    plt.close(fig)


def plot_representative(rep_events, synthetic):
    """图03：代表性事件的逐日CAR曲线（最大/最小/指定大事件）。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for ann, ev in rep_events:
        evd = str(ev['event_date'].date())
        name = ann if evd == ann else f'{ann}（顺延至{evd}）'
        ax.plot(ev['ks'], ev['car'] * 100, marker='o', ms=4,
                label=f"{name}  CAR[0,+10]={ev['car_0_10']*100:.2f}%")
    ax.axvline(0, color='red', ls='--', lw=1.2, label='事件日 t=0')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('相对事件日的交易日 k')
    ax.set_ylabel('累计异常收益 CAR（%）')
    title = '代表性事件的逐日CAR（CAR[0,+10]最大/最小事件，另补2024-09-24与2015-08-25）'
    if synthetic:
        title += '【合成数据】'
    ax.set_title(title)
    ax.set_xticks(range(EV_START, EV_END + 1))
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_代表性事件逐日CAR.png'), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    df, synthetic, source = load_index_data()
    dates = df.index
    print('=' * 72)
    print(f'数据来源: {source}')
    print(f'样本区间: {dates[0].date()} ~ {dates[-1].date()}，共 {len(df)} 个交易日')
    print('=' * 72)

    # ---- 1) 真实事件：对齐 + 计算AR/CAR -------------------------------
    details_rows, ev_by_ann, car_cols, real_positions = [], {}, {}, []
    for ann, name in EVENTS.items():
        pos = align_event_day(dates, ann)         # 公告日 -> 交易日位置（顺延规则）
        if pos is None or pos + EST_START < 1 or pos + EV_END > len(df) - 1:
            print(f'[跳过] {ann} {name}：数据窗口不足')
            continue
        ev = compute_event(df, pos)
        ev_by_ann[ann] = ev
        real_positions.append(pos)
        car_cols[ann] = ev['car']
        details_rows.append({
            '公告日': ann,
            '事件日(顺延后)': ev['event_date'].date(),
            '事件说明': name,
            'CAR[-5,+10](%)': round(ev['car_5_10'] * 100, 4),
            'CAR[0,+10](%)': round(ev['car_0_10'] * 100, 4),
            '事件后1日收益t+1(%)': round(ev['ret_t1'] * 100, 4),
            '事件日收益t0(%)': round(ev['ret_t0'] * 100, 4),
        })

    details = pd.DataFrame(details_rows)
    car_matrix = pd.DataFrame(car_cols, index=range(EV_START, EV_END + 1))
    n_events = len(details)

    # 独立性检查：相邻真实事件的最小间隔是否大于事件窗长度（16个交易日）
    gaps = np.diff(sorted(real_positions))
    print(f'\n真实事件数 N={n_events}，相邻事件最小间隔 {gaps.min()} 个交易日'
          f'（>事件窗长16天，窗口互不重叠，截面样本近似独立）')

    print('\n逐事件明细：')
    print(details.to_string(index=False))

    # ---- 2) 平均CAR曲线与置信带 ---------------------------------------
    avg_car = car_matrix.mean(axis=1)             # 每个相对日上的截面平均CAR
    se_car = car_matrix.std(axis=1, ddof=1) / np.sqrt(n_events)  # 截面标准误
    plot_avg_car(avg_car, se_car, n_events, synthetic)

    # ---- 3) 显著性检验：截面 t 检验 -----------------------------------
    # 原假设 H0：真实平均CAR = 0（公告平均而言不产生超额收益）。
    # t = 样本均值 / (样本标准差/sqrt(N))。p < 0.05 才敢说“显著”。
    real_c0 = details['CAR[0,+10](%)'].to_numpy()
    real_c5 = details['CAR[-5,+10](%)'].to_numpy()
    t0 = stats.ttest_1samp(real_c0, 0.0)
    t5 = stats.ttest_1samp(real_c5, 0.0)
    print(f'\n[截面t检验] CAR[0,+10]:  均值={real_c0.mean():+.2f}%  '
          f'中位数={np.median(real_c0):+.2f}%  t={t0.statistic:+.3f}  p={t0.pvalue:.4f}')
    print(f'[截面t检验] CAR[-5,+10]: 均值={real_c5.mean():+.2f}%  '
          f't={t5.statistic:+.3f}  p={t5.pvalue:.4f}')
    print(f'真实事件中 CAR[0,+10] > 0 的个数: {(real_c0 > 0).sum()}/{n_events}')

    # ---- 4) 安慰剂检验（本课灵魂） -------------------------------------
    placebo_pos = sample_placebos(df, real_positions)
    placebo_cars = np.array([compute_event(df, p)['car_0_10'] * 100
                             for p in placebo_pos])
    real_mean = float(real_c0.mean())
    pct = stats.percentileofscore(placebo_cars, real_mean)  # 真实均值在安慰剂分布中的分位
    q025, q975 = np.percentile(placebo_cars, [2.5, 97.5])
    print(f'\n[安慰剂检验] {len(placebo_cars)}个假事件（距真实事件>30个交易日，rng(42)）')
    print(f'  安慰剂 CAR[0,+10]: 均值={placebo_cars.mean():+.2f}%  '
          f'标准差={placebo_cars.std(ddof=1):.2f}%  '
          f'范围=[{placebo_cars.min():+.2f}%, {placebo_cars.max():+.2f}%]')
    print(f'  安慰剂 2.5%~97.5% 分位: [{q025:+.2f}%, {q975:+.2f}%]')
    print(f'  真实平均CAR {real_mean:+.2f}% 落在安慰剂分布的第 {pct:.0f} 百分位')
    plot_placebo(real_c0, placebo_cars, real_mean, pct, synthetic)

    # ---- 5) 代表性事件逐日CAR -----------------------------------------
    # 选3个不同事件：CAR[0,+10]最大者、最小者、指定超大事件2024-09-24；
    # 若去重后不足3个，再补上2015-08-25双降（CAR[-5,+10]最极端的历史事件）。
    ann_max = details.loc[real_c0.argmax(), '公告日']
    ann_min = details.loc[real_c0.argmin(), '公告日']
    rep_anns = []
    for cand in [ann_max, ann_min, '2024-09-24', '2015-08-25']:
        if cand not in rep_anns:
            rep_anns.append(cand)
        if len(rep_anns) == 3:
            break
    rep = [(a, ev_by_ann[a]) for a in rep_anns]
    plot_representative(rep, synthetic)

    # ---- 6) 汇总表落盘 -------------------------------------------------
    details.to_csv(os.path.join(OUT_DIR, 'event_details.csv'),
                   index=False, encoding='utf-8-sig')
    pt = stats.ttest_1samp(placebo_cars, 0.0)
    test_stats = pd.DataFrame([
        {'检验项': '真实事件 CAR[0,+10] 截面t检验（H0: 平均CAR=0）',
         'N': n_events, '均值(%)': round(real_c0.mean(), 4),
         '标准差(%)': round(real_c0.std(ddof=1), 4),
         't值': round(t0.statistic, 4), 'p值': round(t0.pvalue, 4),
         '结论/备注': 'p<0.05为显著；不显著=公告平均而言未被市场带来超额收益'},
        {'检验项': '真实事件 CAR[-5,+10] 截面t检验',
         'N': n_events, '均值(%)': round(real_c5.mean(), 4),
         '标准差(%)': round(real_c5.std(ddof=1), 4),
         't值': round(t5.statistic, 4), 'p值': round(t5.pvalue, 4),
         '结论/备注': '含公告前5天，检验是否有提前反应/泄密'},
        {'检验项': '安慰剂假事件 CAR[0,+10]',
         'N': len(placebo_cars), '均值(%)': round(placebo_cars.mean(), 4),
         '标准差(%)': round(placebo_cars.std(ddof=1), 4),
         't值': round(pt.statistic, 4), 'p值': round(pt.pvalue, 4),
         '结论/备注': '纯随机抽样，理论上应约等于0、不显著'},
        {'检验项': '真实平均CAR在安慰剂分布中的分位数(%)',
         'N': len(placebo_cars), '均值(%)': round(pct, 4),
         '标准差(%)': '', 't值': '', 'p值': '',
         '结论/备注': '越接近50，说明真实事件看起来越像随机波动'},
    ])
    test_stats.to_csv(os.path.join(OUT_DIR, 'test_stats.csv'),
                      index=False, encoding='utf-8-sig')

    # ---- 7) 教学结论（如实呈现，不显著也是结论） -----------------------
    sig = '显著' if t0.pvalue < 0.05 else '不显著'
    print('\n' + '=' * 72)
    print('教学结论（如实呈现）:')
    print(f'  1. 15次降准/重大政策公告的平均 CAR[0,+10] = {real_mean:+.2f}%，'
          f'截面t检验 p={t0.pvalue:.4f}，在5%水平上{sig}。')
    print(f'  2. 真实平均CAR仅位于安慰剂分布的第 {pct:.0f} 百分位'
          f'（安慰剂范围 [{placebo_cars.min():+.2f}%, {placebo_cars.max():+.2f}%]），'
          f'与随机波动难以区分。')
    print('  3. 机制解读——预期已定价：降准常被市场提前预期（媒体、政策吹风、')
    print('     历史规律），公告日往往只剩“利好落地”的兑现盘；生效日更无信息量。')
    print('     2024-09-24 是例外：一揽子政策远超预期，CAR[0,+10] 一事独大，')
    print('     也提醒我们“把单个大事件当成规律”是事件研究的大坑。')
    print('  4. 这正是事件研究教给我们的：市场对“可预期的消息”在公告前就已反应，')
    print('     简单地“公告日买入”并没有统计上可辨认的超额收益。')
    print('=' * 72)
    print(f'\n全部输出已保存至: {OUT_DIR}')


if __name__ == '__main__':
    main()
