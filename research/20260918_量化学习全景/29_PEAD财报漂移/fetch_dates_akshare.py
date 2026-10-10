# -*- coding: utf-8 -*-
"""
fetch_dates_akshare.py —— 第29课「PEAD财报漂移」取数第二步（akshare 补充源）

背景（真实调试留痕，详见 README「数据管道说明」）：
  iFinD 的 get_stock_financials 用自然语言问法查"公告日期"，经 4 次试探（raw/01_financials_batch*_v*.json）：
    v1 "公告日期"     → 指标被解析器丢弃，只回报告期+利润；
    v2 "财报公告日期" → 解析成"定期报告实际披露日期/预计披露日期"，但每股只回 1 行标量，无报告期映射；
    v3 每期单独点名   → 仍每股 1 行，且返回的是最新年报（2026-03）的日期，期次参数被忽略；
    v4 引擎指标名+期次 → 每股 1 行标量，实测映射到"第一个提到的报告期"，只够覆盖 1 个期次；
  且 v4 问法的利润表不含去年同期分母（20231231/20240630/20240930），而 iFinD 免费 key 总调用预算
  只剩 1 次（已用 7/8），无法补 4 批。故公告日期与去年同期分母改用 akshare 东财接口补全：
    1) stock_profit_sheet_by_report_em：每只股票按报告期的利润表，自带 NOTICE_DATE（公告日期）
       和 PARENT_NETPROFIT（归母净利润）—— 公告日期主数据源 + 分母补全 + 与 iFinD 利润交叉核验；
    2) stock_yysj_em：全市场"预约披露时间"表（首次预约/实际披露）—— 公告日期第二交叉核验源。

用法:
    .venv/bin/python fetch_dates_akshare.py          # 已缓存的文件自动跳过（幂等）
    .venv/bin/python fetch_dates_akshare.py --fresh  # 忽略缓存重取

留痕: raw/akshare_profit_{code}.csv（30只，全期次）、raw/akshare_yysj_{period}.csv（3期全市场表）
"""
import argparse
import os
import time

import akshare as ak
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'raw')
CACHE_DIR = '/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/data/cache'
os.makedirs(RAW, exist_ok=True)

# 事件期（报告期 -> 展示名）与 yosy 接口的 date 参数（=报告期末日）
EVENT_PERIODS = [('20241231', '2024年报'), ('20250630', '2025半年报'), ('20250930', '2025三季报')]
KEEP_PERIODS = ['2023-12-31', '2024-06-30', '2024-09-30', '2024-12-31', '2025-06-30', '2025-09-30']


def log(msg):
    print(f'[akshare] {msg}', flush=True)


def stock_list():
    rows = []
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.startswith('stock_') and f.endswith('.csv'):
            parts = f.replace('.csv', '').split('_')
            rows.append((parts[1], parts[2]))
    return sorted(rows)


def fetch_profit_sheet(code, name, fresh):
    """单只股票按报告期利润表 → 留痕 CSV（保留全部历史期次，裁剪列）"""
    fpath = os.path.join(RAW, f'akshare_profit_{code}.csv')
    if not fresh and os.path.exists(fpath):
        log(f'{name}({code}) 已缓存，跳过')
        return
    symbol = ('SH' if code.startswith('6') else 'SZ') + code
    df = None
    for attempt in (1, 2):  # 失败重试 1 次
        try:
            df = ak.stock_profit_sheet_by_report_em(symbol=symbol)
            break
        except Exception as e:  # noqa: BLE001
            log(f'{name}({code}) 第{attempt}次失败: {e}')
            time.sleep(2.0)
    if df is None or df.empty:
        log(f'{name}({code}) 获取失败（主流程会如实降级：该股用 iFinD 利润+yysj 日期）')
        return
    cols = [c for c in ('REPORT_DATE', 'REPORT_TYPE', 'NOTICE_DATE', 'PARENT_NETPROFIT') if c in df.columns]
    out = df[cols].copy()
    out.to_csv(fpath, index=False, encoding='utf-8-sig')
    log(f'{name}({code}) 利润表 {len(out)} 期已存 {os.path.basename(fpath)}')


def fetch_yysj(period, label, fresh):
    """全市场预约披露时间表（含实际披露时间）→ 留痕 CSV"""
    fpath = os.path.join(RAW, f'akshare_yysj_{period}.csv')
    if not fresh and os.path.exists(fpath):
        log(f'yysj {label} 已缓存，跳过')
        return
    df = ak.stock_yysj_em(symbol='沪深A股', date=period)
    df.to_csv(fpath, index=False, encoding='utf-8-sig')
    n_actual = df['实际披露时间'].notna().sum() if '实际披露时间' in df.columns else 0
    log(f'yysj {label}: 全市场 {len(df)} 行（实际披露时间非空 {n_actual} 行）已存 {os.path.basename(fpath)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    stocks = stock_list()
    log(f'股票清单 {len(stocks)} 只（来自 data/cache 文件名）')

    for period, label in EVENT_PERIODS:
        fetch_yysj(period, label, args.fresh)
        time.sleep(0.4)

    for i, (code, name) in enumerate(stocks):
        fetch_profit_sheet(code, name, args.fresh)
        if i < len(stocks) - 1:
            time.sleep(0.4)  # 对数据源友好，避免触发东财限流

    ok = sum(1 for c, _ in stocks if os.path.exists(os.path.join(RAW, f'akshare_profit_{c}.csv')))
    log(f'完成: 利润表 {ok}/{len(stocks)} 只, 预约披露表 {len(EVENT_PERIODS)} 期')
    # 教学规范：永不非0退出
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
