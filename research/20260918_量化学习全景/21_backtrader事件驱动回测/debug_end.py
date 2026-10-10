# -*- coding: utf-8 -*-
"""调试2：检查末端分歧与全部钳制成交（留档）"""
import numpy as np
import pandas as pd
import main as m

df, _ = m.load_data()
hand = m.hand_engine(df)
bt_coc = m.run_bt(df, m.SmaCrossStrategy, dict(fast=20, slow=60, tag='btcoc'), coc=True)

nav_h = hand['nav']
nav_b = bt_coc['nav'].reindex(nav_h.index).ffill()
ratio = (nav_b / nav_h - 1.0) * 100

print('每月末净值比偏差(%) 2024-06 起:')
print(ratio[ratio.index >= '2024-06-01'].resample('ME').last().round(4).to_string())

close = df['close']
print('\n2025-2026 的 bt 成交 vs 收盘×(1±滑点):')
for f in bt_coc['fills']:
    if f['日期'] >= pd.Timestamp('2025-01-01'):
        c0 = float(close.loc[f['日期']])
        expect = c0 * (1 + m.SLIPPAGE) if f['方向'] == '买入' else c0 * (1 - m.SLIPPAGE)
        print(f"  {f['日期'].date()} {f['方向']} {f['股数']}股 fill={f['成交价']:.4f} expect={expect:.4f} "
              f"diff={f['成交价']-expect:+.6f} close={c0:.4f} high={df.loc[f['日期'],'high']:.4f} low={df.loc[f['日期'],'low']:.4f}")

# 手写最后一轮的信号
sma_f, sma_s = close.rolling(20).mean(), close.rolling(60).mean()
signal = pd.Series(np.where(sma_f > sma_s, 1.0, 0.0), index=close.index)
arr = signal.to_numpy(); prev = np.concatenate([[0.0], arr[:-1]])
opens = np.where((arr == 1) & (prev == 0))[0]
closes_ = np.where((arr == 0) & (prev == 1))[0]
print('\n手写最后3个开仓日:', [str(df.index[i].date()) for i in opens[-3:]])
print('手写最后3个平仓日:', [str(df.index[i].date()) for i in closes_[-3:]])
print('期末信号:', signal.iloc[-1], ' 最后5日信号:', signal.iloc[-5:].tolist())

# bt 最后两笔成交
print('\nbt 最后3笔成交:')
for f in bt_coc['fills'][-3:]:
    print(f"  {f['日期'].date()} {f['方向']} {f['股数']}股 @{f['成交价']:.4f} comm={f['佣金']:.2f}")

# 手写与 bt 在最后一个开仓日的投入规模
last_open = df.index[opens[-1]]
c0 = float(close.loc[last_open])
nav_at_open_hand = float(nav_h.loc[last_open])
nav_at_open_bt = float(nav_b.loc[last_open])
print(f'\n最后开仓日 {last_open.date()}: close={c0:.4f} 手写NAV={nav_at_open_hand:.4f} btNAV={nav_at_open_bt:.4f}')
print(f'期末: 手写NAV={nav_h.iloc[-1]:.4f} btNAV={nav_b.iloc[-1]:.4f}')
print(f'最后开仓日->期末: 手写涨 {nav_h.iloc[-1]/nav_at_open_hand-1:.4%}, bt涨 {nav_b.iloc[-1]/nav_at_open_bt-1:.4%}')
