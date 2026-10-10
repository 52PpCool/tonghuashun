# -*- coding: utf-8 -*-
"""调试：定位手写引擎与 bt coc 净值路径的第一处分歧（留档，不参与教学输出）"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import main as m  # noqa: E402

df, IS_SYNTH = m.load_data()
hand = m.hand_engine(df)
bt_coc = m.run_bt(df, m.SmaCrossStrategy, dict(fast=20, slow=60, tag='btcoc'), coc=True)

nav_h = hand['nav']
nav_b = bt_coc['nav'].reindex(nav_h.index).ffill()
ratio = nav_b / nav_h - 1.0
bad = ratio[ratio.abs() > 0.002]
print('偏离>0.2%%的天数:', len(bad))
if len(bad):
    print('第一处大偏离:', bad.index[0].date(), f'{bad.iloc[0]*100:.3f}%')
    print('最后一处:', bad.index[-1].date(), f'{bad.iloc[-1]*100:.3f}%')
    print('偏离随时间分布（每500天抽样）:')
    print(ratio.iloc[::500].round(5).to_string())

# 逐轮对比
close = df['close']
sma_f, sma_s = close.rolling(20).mean(), close.rolling(60).mean()
signal = pd.Series(np.where(sma_f > sma_s, 1.0, 0.0), index=close.index)
arr = signal.to_numpy(); prev = np.concatenate([[0.0], arr[:-1]])
opens = list(np.where((arr == 1) & (prev == 0))[0]); closes_ = list(np.where((arr == 0) & (prev == 1))[0])
events = sorted([(i, 'B') for i in opens] + [(i, 'S') for i in closes_])
print('\n手写事件日 vs bt 成交日（前10轮）:')
fills = bt_coc['fills']
hand_days = [df.index[i] for i, _ in events]
bt_days = [f['日期'] for f in fills]
for k in range(min(12, len(events))):
    hd = hand_days[k].date() if k < len(hand_days) else None
    bd = bt_days[k].date() if k < len(bt_days) else None
    flag = 'OK' if hd == bd else '<<< 不同!'
    print(f'  事件{k}: 手写 {hd} vs bt {bd}  {flag}')

# 每轮净收益对比
hi = 0
rows = []
for k, (i, ev) in enumerate(events):
    if ev == 'B':
        bi = i
    else:
        if bi is None:
            continue
        hnet = close.iloc[i] * (1 - m.FEE_SELL) / (close.iloc[bi] * (1 + m.FEE_BUY)) - 1
        rows.append((df.index[bi].date(), df.index[i].date(), hnet * 100))
        bi = None
bnet = []
bi = None
for f in fills:
    if f['方向'] == '买入':
        bi = f
    elif bi is not None:
        net = f['成交价'] * (1 - 0.00075) / (bi['成交价'] * (1 + 0.00025)) - 1
        bnet.append((bi['日期'].date(), f['日期'].date(), net * 100))
        bi = None
print('\n逐轮净收益对比（手写 vs bt，前12轮）:')
for k in range(min(12, len(rows))):
    hb, hs, hnet = rows[k]
    same = bnet[k] if k < len(bnet) else (None, None, float('nan'))
    print(f'  轮{k}: 手写 {hb}~{hs} {hnet:+.3f}% | bt {same[0]}~{same[1]} {same[2]:+.3f}% | 差 {same[2]-hnet:+.3f}pp')
