# -*- coding: utf-8 -*-
"""探针脚本：验证 backtrader 关键行为（运行完可留档，不参与教学输出）"""
import os
import numpy as np
import pandas as pd
import backtrader as bt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'data', 'cache', 'stock_000001_平安银行.csv')

df = pd.read_csv(DATA, encoding='utf-8-sig', parse_dates=['date']).set_index('date').sort_index()
print('bars:', len(df), df.index[0].date(), df.index[-1].date())


class CommInfo(bt.CommInfoBase):
    params = dict(stocklike=True, commtype=bt.CommInfoBase.COMM_PERC, percabs=True)

    def _getcommission(self, size, price, pseudoexec):
        comm = abs(size) * price * 2.5e-4
        if size < 0:
            comm += abs(size) * price * 5.0e-4
        return comm


class S(bt.Strategy):
    params = dict(n=3, coc_mode='')

    def __init__(self):
        self.i = 0

    def next(self):
        self.i += 1
        if self.i == self.p.n and not self.position:
            print(f'  next@{self.data.datetime.date(0)} close={self.data.close[0]:.4f} -> buy')
            self.buy()

    def notify_order(self, order):
        if order.status in (order.Completed, order.Margin, order.Rejected):
            d = order.executed.dt
            print(f'  [{self.p.coc_mode}] {order.getstatusname()} size={order.executed.size} '
                  f'px={order.executed.price:.6f} comm={order.executed.comm:.2f} dt={pd.Timestamp(d).date()}')


def run(coc, slip_open):
    cere = bt.Cerebro()
    cere.adddata(bt.feeds.PandasData(dataname=df))
    cere.broker.setcash(1_000_000)
    cere.broker.addcommissioninfo(CommInfo())
    if slip_open:
        cere.broker.set_slippage_perc(perc=1e-3, slip_open=True)
    if coc:
        cere.broker.set_coc(True)
    cere.addstrategy(S, coc_mode=f'coc={coc} slip_open={slip_open}')
    cere.addanalyzer(bt.analyzers.Returns, _name='ret')
    cere.addanalyzer(bt.analyzers.SharpeRatio_A, _name='shrp')
    cere.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cere.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cere.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days, _name='tr')
    res = cere.run(maxcpus=1)[0]
    print('  value_end=%.2f  rnorm=%.6f shrpA=%s dd_max=%s' % (
        cere.broker.getvalue(), res.analyzers.ret.rets['rnorm'],
        res.analyzers.shrp.rets['sharperatio'], res.analyzers.dd.rets['max']['drawdown']))
    ta = res.analyzers.ta.rets
    print('  ta total=%s closed=%s open=%s won=%s lost=%s' % (
        ta['total']['total'], ta['total'].get('closed', 0), ta['total'].get('open', 0),
        ta.get('won', {}).get('total', 0), ta.get('lost', {}).get('total', 0)))
    tr = res.analyzers.tr.rets
    print('  TimeReturn n=%d' % len(tr))
    return res


print('--- run1: coc=True slip_open=True (信号当日收盘成交+滑点) ---')
r1 = run(coc=True, slip_open=True)
print('--- run2: coc=False slip_open=True (次日开盘成交+滑点) ---')
r2 = run(coc=False, slip_open=True)
print('--- run3: coc=False slip_open=False (次日开盘成交, 无滑点) ---')
r3 = run(coc=False, slip_open=False)

# 检查成交价与 close*1.001 的关系（验证滑点与 slip_match 钳制）
d0 = df.index[2]
print('day3 close=%.4f  1.001x=%.6f' % (df.loc[d0, 'close'], df.loc[d0, 'close'] * 1.001))

# 快速检查优化在 maxcpus=1 下 optreturn 返回结构
class S2(S):
    params = dict(n=3, coc_mode='', p1=20)

cere = bt.Cerebro(maxcpus=1)
cere.adddata(bt.feeds.PandasData(dataname=df))
cere.broker.setcash(1_000_000)
cere.broker.addcommissioninfo(CommInfo())
cere.optstrategy(S2, p1=(5, 7))
cere.addanalyzer(bt.analyzers.Returns, _name='ret')
res = cere.run(maxcpus=1)
print('opt results:', len(res), type(res[0]).__name__)
for rlist in res:
    for o in rlist:
        print('  params p1=', o.params.p1, 'rnorm=', o.analyzers[0].rets['rnorm'])
print('PROBE DONE')
