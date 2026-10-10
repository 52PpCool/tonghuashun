# -*- coding: utf-8 -*-
"""
02 双均线策略与向量化回测（单文件自包含教学脚本）
=================================================
这节课用一只真实股票（平安银行 000001，2015 年起前复权日K）讲三件事：

1) SMA(20)/SMA(60) 金叉死叉策略：金叉满仓买入、死叉清仓。
   全程 pandas 向量化（np.where / shift / diff），不用 for 循环逐 bar。
2)【本课灵魂】同一策略写两个版本并排对比：
   - 正确版：今日收盘算出信号 -> shift(1) -> 明日起才按新仓位结算收益（"信号次日执行"）；
   - 前视偏差版：当日信号当日就享受当日涨幅（新手最常见错误，
     "当日收盘价既当裁判——算信号，又当运动员——享受当日涨跌"）。
   两版净值 + 买入持有基准画在一张图上，三版本都计入真实交易成本。
3) 交易成本建模（佣金双边万2.5 + 卖出印花税千0.5 + 滑点双边各0.1%，参数可调），
   以及成本敏感性分析（单边成本率 0 ~ 0.3% 共 10 档）。

输出（outputs/）：
  01_三版本净值对比.png / 02_价格与买卖信号.png / 03_成本敏感性.png
  metrics.csv（三版本绩效表）/ trades.csv（开平仓交易记录）

运行：
  cd 02_双均线策略与向量化回测
  /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py

数据：若 ../data/cache/stock_000001_平安银行.csv 缺失，
      自动用 np.random.default_rng(42) 生成合成数据兜底（所有图标题会标注"合成数据"），
      脚本绝不因缺数据而失败。
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 无界面环境出图
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. 全局设置
# ----------------------------------------------------------------------
SEED = 42                    # 随机种子：策略逻辑本身无随机性，仅合成数据兜底时使用
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'cache', 'stock_000001_平安银行.csv')

TRADING_DAYS = 252   # 年化用：一年约 252 个交易日
RF_ANNUAL = 0.02     # 无风险利率 2%（夏普比率用）

# ---- 交易成本参数（做成可调参数，默认=本课规定值）----
COMMISSION = 2.5e-4        # 佣金：双边各万 2.5
STAMP_TAX = 5.0e-4         # 印花税：仅卖出收千 0.5
SLIPPAGE = 1.0e-3          # 滑点：双边各 0.1%
FEE_BUY = COMMISSION + SLIPPAGE               # 买入单边合计 0.125%
FEE_SELL = COMMISSION + STAMP_TAX + SLIPPAGE  # 卖出单边合计 0.175%（多一项印花税）

FAST, SLOW = 20, 60        # 双均线参数：短均线 20 日、长均线 60 日


# ----------------------------------------------------------------------
# 1. 数据加载（真实缓存优先，缺失则合成兜底）
# ----------------------------------------------------------------------
def load_data():
    """读取本地缓存的真实日K；缺失/损坏时用 rng(42) 造合成数据，绝不失败。"""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', parse_dates=['date'])
            df = df.set_index('date').sort_index()
            df = df[~df.index.duplicated(keep='last')].dropna(subset=['close'])
            if len(df) > SLOW + 20 and {'open', 'high', 'low', 'close', 'volume'} <= set(df.columns):
                return df, False
        except Exception as e:  # noqa: BLE001
            print(f'[警告] 缓存读取失败（{e}），改用合成数据。')
    # ---- 合成数据兜底：分段漂移的几何随机游走（保证有金叉死叉可玩）----
    n = 2800
    idx = pd.bdate_range('2015-01-05', periods=n)
    q = n // 4
    drift = np.concatenate([np.full(q, 0.0006), np.full(q, -0.0004), np.full(n - 2 * q, 0.0002)])
    r = drift + RNG.normal(0.0, 0.018, n)
    close = pd.Series(11.0 * np.exp(np.cumsum(r)), index=idx, name='close')
    spread = np.abs(RNG.normal(0.0, 0.008, n))
    df = pd.DataFrame({
        'open': close * (1 + RNG.normal(0, 0.004, n)),
        'high': close * (1 + spread),
        'low': close * (1 - spread),
        'close': close,
        'volume': RNG.lognormal(18.5, 0.5, n).astype(int),
    }, index=idx)
    print('[警告] 未找到真实缓存，已生成合成数据（图中均标注"合成数据"）。')
    return df, True


# ----------------------------------------------------------------------
# 2. 成本模型：把"仓位变化"翻译成"单边成本扣减"（全向量化）
# ----------------------------------------------------------------------
def apply_costs(pos_ret, pos_exec, fee_buy=FEE_BUY, fee_sell=FEE_SELL):
    """
    pos_ret  : 策略"毛收益"序列 = 当日持仓仓位 × 标的日收益（★★ 必须先按仓位加权！）
    pos_exec : 每日"执行仓位"序列（0/1）。相邻两日仓位之差 = 买/卖的数量（满仓=1）。
    返回     : (策略净收益序列, 每日成本序列)。
      买入成本率 = 佣金 + 滑点；卖出成本率 = 佣金 + 印花税 + 滑点。
    """
    d = pos_exec.diff()
    if len(d) > 0:
        d.iloc[0] = pos_exec.iloc[0]      # 回测起点若直接建仓，也要付一次买入成本
    buy_amt = d.clip(lower=0.0)           # 加仓量（0->1 = 买入 1 份）
    sell_amt = (-d).clip(lower=0.0)       # 减仓量（1->0 = 卖出 1 份）
    cost = buy_amt * fee_buy + sell_amt * fee_sell
    return pos_ret - cost, cost


# ----------------------------------------------------------------------
# 3. 绩效统计
# ----------------------------------------------------------------------
def perf_stats(name, strat_ret, pos_cost, trades_df, freq=TRADING_DAYS):
    """由净收益序列计算绩效表所需全部指标。pos_cost 用于估计换手率。"""
    nav = (1.0 + strat_ret).cumprod()
    years = len(strat_ret) / freq
    total_ret = nav.iloc[-1] - 1.0
    ann_ret = nav.iloc[-1] ** (1.0 / years) - 1.0
    ann_vol = strat_ret.std() * np.sqrt(freq)          # 样本 std(ddof=1) 年化
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 1e-12 else np.nan
    max_dd = (nav / nav.cummax() - 1.0).min()

    d = pos_cost.diff()
    if len(d) > 0:
        d.iloc[0] = pos_cost.iloc[0]
    turnover_ann = d.abs().sum() / years               # 单边换手：每年买+卖的满仓次数

    closed = trades_df[trades_df['事件'].astype(str).str.startswith('平仓')] if len(trades_df) else trades_df
    win_rate = (closed['净收益率(%)'] > 0).mean() if len(closed) else np.nan

    return {
        '版本': name,
        '期末净值': round(float(nav.iloc[-1]), 4),
        '累计收益率(%)': round(total_ret * 100, 2),
        '年化收益率(%)': round(ann_ret * 100, 2),
        '年化波动率(%)': round(ann_vol * 100, 2),
        '夏普比率(rf=2%)': round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        '最大回撤(%)': round(max_dd * 100, 2),
        '胜率(按笔,%)': round(win_rate * 100, 1) if not np.isnan(win_rate) else np.nan,
        '交易次数(开+平)': int(len(trades_df)),
        '年化单边换手(倍)': round(float(turnover_ann), 2),
    }


# ----------------------------------------------------------------------
# 4. 交易记录提取（只对"事件"做循环，几十次，不是逐 bar）
# ----------------------------------------------------------------------
def extract_trades(sig, lag_effective, close, version):
    """
    sig           : 0/1 信号序列（收盘后才可知）。
    lag_effective : 生效滞后交易日数。正确版=1（信号次日生效）；前视偏差版=0（当日生效，不可能！）。
    成交价口径    : 信号日收盘价（shift(1) 口径的隐含成交价，近似"收盘集合竞价下单"）。
    生效日        : 新仓位的收益自该交易日起计入。
    """
    idx = sig.index
    arr = sig.to_numpy()
    px = close.to_numpy()
    prev = np.concatenate([[0.0], arr[:-1]])                     # 回测起点前视为空仓
    up_pos = np.where((arr == 1) & (prev == 0))[0]               # 金叉：0 -> 1
    dn_pos = np.where((arr == 0) & (prev == 1))[0]               # 死叉：1 -> 0
    events = sorted([(p, '开仓') for p in up_pos] + [(p, '平仓') for p in dn_pos])

    def effect_day(i):
        return idx[i + lag_effective] if i + lag_effective < len(idx) else pd.NaT

    rows, open_p = [], None
    for p, ev in events:
        if ev == '开仓':
            open_p = p
            note = '信号次日生效' if lag_effective == 1 else '当日信号当日生效（现实中不可能）'
            rows.append({'版本': version, '事件': '开仓(金叉买入)', '信号日': idx[p],
                         '生效日': effect_day(p), '成交价': round(float(px[p]), 4),
                         '净收益率(%)': np.nan, '备注': note})
        else:
            if open_p is None:
                continue
            net = (px[p] * (1 - FEE_SELL)) / (px[open_p] * (1 + FEE_BUY)) - 1.0
            rows.append({'版本': version, '事件': '平仓(死叉卖出)', '信号日': idx[p],
                         '生效日': effect_day(p), '成交价': round(float(px[p]), 4),
                         '净收益率(%)': round(net * 100, 2),
                         '备注': f'整轮往返净收益(含双边成本)，持仓{p - open_p}个交易日'})
            open_p = None
    if open_p is not None:  # 期末仍持仓：按最后收盘价估值
        net = (px[-1] * (1 - FEE_SELL)) / (px[open_p] * (1 + FEE_BUY)) - 1.0
        rows.append({'版本': version, '事件': '平仓(期末估值)', '信号日': idx[open_p],
                     '生效日': idx[-1], '成交价': round(float(px[-1]), 4),
                     '净收益率(%)': round(net * 100, 2),
                     '备注': '期末仍持仓，按最后收盘价估值(未实际平仓)'})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    pd.set_option('display.unicode.east_asian_width', True)

    df, IS_SYNTH = load_data()
    close = df['close']
    idx = close.index
    ret = close.pct_change().fillna(0.0)          # 日收益率（close-to-close）

    tag = '【合成数据】' if IS_SYNTH else ''
    print(f'数据：{idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}，共 {len(idx)} 个交易日'
          f'{"（合成数据）" if IS_SYNTH else "（真实前复权日K）"}')
    print(f'成本参数：佣金万{COMMISSION*1e4:.1f}(双边) + 印花税千{STAMP_TAX*1e3:.1f}(卖) '
          f'+ 滑点{SLIPPAGE*100:.1f}%(双边) => 买 {FEE_BUY*100:.3f}%/卖 {FEE_SELL*100:.3f}% 每边\n')

    # ================================================================
    # 5. 信号生成（向量化）：金叉=1，死叉=0。注意：信号在当日收盘后才可知！
    # ================================================================
    sma_fast = close.rolling(FAST).mean()
    sma_slow = close.rolling(SLOW).mean()
    signal = pd.Series(np.where(sma_fast > sma_slow, 1.0, 0.0), index=idx)  # NaN 比较为 False => 预热期空仓

    # ================================================================
    # 6.【本课灵魂】两个版本的仓位序列，只差一个 shift(1)
    # ================================================================
    # 正确版：t 日收盘算出信号 -> shift(1) -> t+1 日起才按新仓位结算收益（"信号次日执行"）
    pos_ok = signal.shift(1).fillna(0.0)
    # 前视偏差版：t 日信号直接乘 t 日收益 = 用当天收盘价既当裁判又当运动员（新手最常见错误）
    pos_bad = signal.copy()

    sr_ok, cost_ok = apply_costs(pos_ok * ret, pos_ok)     # 正确版净收益（含成本）
    sr_bad, cost_bad = apply_costs(pos_bad * ret, pos_bad)  # 前视版净收益（含成本）

    # 买入持有基准：第一天收盘买入、最后一天清仓，同样计入双边成本
    pos_bh_cost = pd.Series(1.0, index=idx)
    pos_bh_cost.iloc[-1] = 0.0                    # 仅为计成本构造期末卖出（收益仍按满仓计）
    sr_bh, cost_bh = apply_costs(ret, pos_bh_cost)

    nav_ok = (1.0 + sr_ok).cumprod()
    nav_bad = (1.0 + sr_bad).cumprod()
    nav_bh = (1.0 + sr_bh).cumprod()

    # ================================================================
    # 7. 绩效表 + 交易记录
    # ================================================================
    tr_ok = extract_trades(signal, lag_effective=1, close=close, version='正确版(信号次日执行)')
    tr_bad = extract_trades(signal, lag_effective=0, close=close, version='前视偏差版(当日成交)')

    tr_bh = pd.DataFrame([
        {'版本': '买入持有', '事件': '开仓(建仓买入)', '信号日': idx[0], '生效日': idx[0],
         '成交价': round(float(close.iloc[0]), 4), '净收益率(%)': np.nan, '备注': '第一天收盘买入后一直持有'},
        {'版本': '买入持有', '事件': '平仓(期末清仓)', '信号日': idx[-1], '生效日': idx[-1],
         '成交价': round(float(close.iloc[-1]), 4),
         '净收益率(%)': round(((close.iloc[-1] * (1 - FEE_SELL)) / (close.iloc[0] * (1 + FEE_BUY)) - 1) * 100, 2),
         '备注': '最后一天收盘清仓'},
    ])

    metrics = pd.DataFrame([
        perf_stats('正确版(信号次日执行)', sr_ok, pos_ok, tr_ok),
        perf_stats('前视偏差版(当日成交)', sr_bad, pos_bad, tr_bad),
        perf_stats('买入持有', sr_bh, pos_bh_cost, tr_bh),
    ])

    metrics_path = os.path.join(OUT_DIR, 'metrics.csv')
    trades_path = os.path.join(OUT_DIR, 'trades.csv')
    metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    pd.concat([tr_ok, tr_bad, tr_bh], ignore_index=True).to_csv(trades_path, index=False, encoding='utf-8-sig')

    print('================ 三版本绩效对比（均含交易成本） ================')
    print(metrics.to_string(index=False))
    print(f'\n已保存: {os.path.relpath(metrics_path, BASE_DIR)} / {os.path.relpath(trades_path, BASE_DIR)}')

    # ================================================================
    # 8. 成本敏感性：单边成本率 c 从 0 到 0.3% 共 10 档（佣金+税+滑点合并简化）
    # ================================================================
    cost_grid = np.linspace(0.0, 0.003, 10)
    d_ok = pos_ok.diff()
    d_ok.iloc[0] = pos_ok.iloc[0]
    turn_units = d_ok.abs().to_numpy()            # 每档成本下每单位换手付一次 c

    ann_strategy = []
    for c in cost_grid:
        sr = pos_ok * ret - c * turn_units
        nav = (1.0 + sr).cumprod()
        ann_strategy.append(nav.iloc[-1] ** (TRADING_DAYS / len(idx)) - 1.0)
    ann_strategy = np.array(ann_strategy)

    d_bh = pos_bh_cost.diff()
    d_bh.iloc[0] = pos_bh_cost.iloc[0]
    bh_units = d_bh.abs().to_numpy()              # 买入持有：建仓1次+清仓1次
    ann_bh = np.array([
        ((1.0 + (ret - c * bh_units)).cumprod().iloc[-1]) ** (TRADING_DAYS / len(idx)) - 1.0
        for c in cost_grid
    ])

    # 盈亏平衡成本（年化收益穿过 0 的位置）
    breakeven = None
    for k in range(len(cost_grid) - 1):
        a, b = ann_strategy[k], ann_strategy[k + 1]
        if a == 0:
            breakeven = cost_grid[k]
            break
        if a * b < 0:  # 线性插值找过零点
            breakeven = cost_grid[k] + (cost_grid[k + 1] - cost_grid[k]) * a / (a - b)
            break

    print('\n================ 成本敏感性（单边成本率 -> 策略年化收益） ================')
    sens = pd.DataFrame({'单边成本率(%)': np.round(cost_grid * 100, 3),
                         '策略年化收益(%)': np.round(ann_strategy * 100, 2),
                         '买入持有年化(%)': np.round(ann_bh * 100, 2)})
    print(sens.to_string(index=False))
    if breakeven is not None:
        print(f'>>> 策略盈亏平衡成本 ≈ {breakeven*100:.3f}%/边（高于此成本策略年化转亏）')
    else:
        zero_line = '全档为正' if (ann_strategy > 0).all() else '全档为负'
        print(f'>>> 0~0.3% 档内年化收益{zero_line}，未出现穿零点')

    # ================================================================
    # 9. 图 1：三版本净值对比
    # ================================================================
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.plot(idx, nav_ok, lw=1.8, color='#d62728',
            label=f'正确版：信号次日执行（期末 {nav_ok.iloc[-1]:.2f}）')
    ax.plot(idx, nav_bad, lw=1.8, color='#1f77b4',
            label=f'前视偏差版：当日信号当日成交（期末 {nav_bad.iloc[-1]:.2f}）')
    ax.plot(idx, nav_bh, lw=1.4, color='#7f7f7f', ls='--',
            label=f'买入持有基准（期末 {nav_bh.iloc[-1]:.2f}）')
    ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
    ax.set_title(f'双均线 SMA({FAST})/SMA({SLOW}) 三版本净值对比（含交易成本）{tag}')
    ax.set_ylabel('净值（期初=1）')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, '01_三版本净值对比.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ================================================================
    # 10. 图 2：最近 2 年价格 + 均线 + 金叉死叉买卖点（正确版口径）
    # ================================================================
    win_years = 2
    while True:  # 极端情况下最近2年无交易则放宽窗口
        t_start = idx[-1] - pd.DateOffset(years=win_years)
        m = idx >= t_start
        n_ev = int(tr_ok['生效日'].notna().sum()) if len(tr_ok) else 0
        in_win = tr_ok[(tr_ok['生效日'] >= t_start)] if len(tr_ok) else tr_ok
        if len(in_win) > 0 or win_years >= 4:
            break
        win_years += 1

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.plot(idx[m], close[m], color='#404040', lw=1.2, label='收盘价')
    ax.plot(idx[m], sma_fast[m], lw=1.1, color='#ff7f0e', label=f'SMA({FAST})')
    ax.plot(idx[m], sma_slow[m], lw=1.1, color='#1f77b4', label=f'SMA({SLOW})')

    if len(tr_ok):
        for _, r_ in in_win.iterrows():
            is_buy = str(r_['事件']).startswith('开仓')
            eff = r_['生效日']
            if pd.isna(eff) or eff not in idx:
                continue
            y_eff = float(close.loc[eff])
            # 灰色细箭头：信号日收盘价 -> 次日生效（直观展示 shift(1) 的"信号次日执行"）
            ax.annotate('', xy=(eff, y_eff), xytext=(r_['信号日'], float(r_['成交价'])),
                        arrowprops=dict(arrowstyle='->', color='#999999', lw=0.9))
            ax.scatter([eff], [y_eff], marker='^' if is_buy else 'v',
                       s=90, color='#d62728' if is_buy else '#2ca02c', zorder=5)
        from matplotlib.lines import Line2D
        handles = ax.get_legend_handles_labels()[0]
        handles += [Line2D([], [], marker='^', ls='', color='#d62728', label='买入（信号次日执行）'),
                    Line2D([], [], marker='v', ls='', color='#2ca02c', label='卖出（信号次日执行）')]
        ax.legend(handles=handles, loc='best')
    else:
        ax.legend(loc='best')

    ax.set_title(f'价格与双均线买卖信号（最近 {win_years} 年，正确版口径，灰色箭头=信号日->次日执行）{tag}')
    ax.set_ylabel('价格（前复权）')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, '02_价格与买卖信号.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ================================================================
    # 11. 图 3：成本敏感性衰减曲线
    # ================================================================
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.plot(cost_grid * 100, ann_strategy * 100, 'o-', color='#d62728', lw=1.8,
            label=f'双均线策略（正确版，约{turn_units.sum():.0f}次单边换手）')
    ax.plot(cost_grid * 100, ann_bh * 100, 's--', color='#7f7f7f', lw=1.4,
            label='买入持有（仅建仓+清仓各1次）')
    ax.axhline(0, color='k', lw=0.9, alpha=0.8)
    ax.axvline(0.15, color='#1f77b4', ls=':', lw=1.3)
    ax.annotate('基准成本≈0.15%/边\n（买0.125% / 卖0.175%的中值）',
                xy=(0.15, float(np.interp(0.15, cost_grid * 100, ann_strategy * 100))),
                xytext=(0.16, float(np.interp(0.15, cost_grid * 100, ann_strategy * 100)) - 1.2),
                fontsize=9, color='#1f77b4')
    if breakeven is not None:
        ax.scatter([breakeven * 100], [0], marker='*', s=160, color='#9467bd', zorder=5)
        ax.annotate(f'盈亏平衡 ≈ {breakeven*100:.2f}%/边', xy=(breakeven * 100, 0),
                    xytext=(breakeven * 100 + 0.015, 0.35), fontsize=10, color='#9467bd')
    ax.set_title(f'成本敏感性：单边成本率 0~0.3% 时策略年化收益的衰减{tag}')
    ax.set_xlabel('单边成本率（%，佣金+印花税+滑点合并简化）')
    ax.set_ylabel('年化收益率（%）')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, '03_成本敏感性.png')
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    print('\n已生成图表:')
    for p in (p1, p2, p3):
        print('  ', os.path.relpath(p, BASE_DIR))
    print('\n完成。')


if __name__ == '__main__':
    main()
