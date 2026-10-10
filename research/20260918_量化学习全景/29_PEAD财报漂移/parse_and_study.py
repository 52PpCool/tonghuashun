# -*- coding: utf-8 -*-
"""
parse_and_study.py —— 第29课「PEAD财报漂移」第二~六步：解析 → 事件分组 → 事件研究 → 回测 → 检验

输入:  raw/01_financials_batch*.json   (fetch_ifind.js 留痕的 iFinD 原始响应，含 v1~v4 措辞试探)
       raw/akshare_profit_{code}.csv   (fetch_dates_akshare.py 留痕，NOTICE_DATE=公告日期)
       raw/akshare_yysj_{period}.csv   (东财预约披露/实际披露时间表，第二日期核验源)
       ../../data/cache/stock_*.csv    (30只A股日线，前复权)

输出:  outputs/01_正负面组CAR曲线.png
       outputs/02_PEAD策略净值.png
       outputs/03_安慰剂对比.png
       outputs/04_单只股票事件示例.png
       outputs/event_details.csv / pead_metrics.csv (均 utf-8-sig)

运行:  .venv/bin/python parse_and_study.py 2>&1 | tee run_log.txt
教学规范：任何数据缺失都降级处理并在日志中如实说明，脚本永不非0退出。
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'raw')
OUT = os.path.join(BASE, 'outputs')
CACHE_DIR = '/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/data/cache'
os.makedirs(OUT, exist_ok=True)

COST = 3e-4                  # 双边各万3
EST_LO, EST_HI = -120, -20   # 估计窗（均值调整模型）
EV_LO, EV_HI = -5, 40        # 事件窗
TEST_LO, TEST_HI = 2, 40     # 漂移检验子窗 CAR[+2,+40]
HOLD = 60                    # 策略持有交易日数（约一个季度）
RNG_SEED = 42
N_PLACEBO = 500

# 事件期 -> (展示名, yoy分母报告期)；分母期 = 上年同期，用于自己算同比
PERIODS = {'20241231': '2024年报', '20250630': '2025半年报', '20250930': '2025三季报'}
DENOM = {'20241231': '20231231', '20250630': '20240630', '20250930': '20240930'}
# iFinD 日期标量与报告期的映射（经验规则：实际披露日期 = query里第一个提到的报告期。
# 该规则是否成立由后文交叉核验检验——核验结果不改变研究本身，事件日一律用 akshare NOTICE_DATE）
IFIND_DATE_PERIOD = {
    '01_financials_batch1_v1.json': None,           # v1 问法：无日期表
    '01_financials_batch2_v2.json': '20231231',     # v2：首提期=2023年报（仅作交叉核验）
    '01_financials_batch3_v3test.json': '20251231', # v3：期次参数被忽略，返回最新年报（不采用，留痕）
    '01_financials_batch3_v4test.json': '20241231',
    '01_financials_batch4.json': '20241231',
    '01_financials_batch5.json': '20241231',
    '01_financials_batch6.json': '20241231',
}
XCHECK_PERIODS = set(PERIODS) | set(DENOM.values())  # 交叉核验覆盖的报告期


def log(step, msg):
    print(f'[{step}] {msg}', flush=True)


# ================================================================
# 1. 解析：拆"俄罗斯套娃" + markdown 表格
# ================================================================
def load_ifind_answer(path):
    """读 iFinD 原始响应，拆三层（MCP->业务->answer），返回 markdown 正文。
    兼容两种落盘格式：新格式 {tool,params,response} 与旧格式（response 本体）。"""
    obj = json.load(open(path, encoding='utf-8'))
    resp = obj.get('response', obj)
    if not (isinstance(resp, dict) and resp.get('ok')):
        return None
    text = resp['data']['result']['content'][0]['text']
    for _ in range(2):
        try:
            o = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            break
        if isinstance(o, dict) and isinstance(o.get('data'), str):
            text = o['data']
            continue
        if isinstance(o, dict) and isinstance(o.get('answer'), str):
            text = o['answer']
        break
    return text


_SEP_CELL = re.compile(r'^:?-{2,}:?$')


def parse_markdown_tables(text):
    """markdown 正文里的 '|' 表格 -> DataFrame 列表（处理分隔行/单元格空格/行长短不齐）"""
    tables, current = [], []
    for line in text.split('\n'):
        if line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if cells and all(_SEP_CELL.match(c) for c in cells):
                continue
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    frames = []
    for rows in tables:
        if len(rows) < 2:
            continue
        header, body = rows[0], rows[1:]
        ncol = len(header)
        body = [r[:ncol] + [''] * (ncol - len(r)) if len(r) < ncol else r[:ncol] for r in body]
        frames.append(pd.DataFrame(body, columns=header))
    return frames


def to_profit(s):
    """'378.8338亿' / '-28015835676.94' / '-' -> 元（float）；脏值 -> NaN"""
    s = str(s).strip().replace(',', '')
    if s in ('', '-', '--', 'nan', 'None', '无', 'NaT'):
        return np.nan
    try:
        if s.endswith('万亿'):
            return float(s[:-2]) * 1e12
        if s.endswith('亿'):
            return float(s[:-1]) * 1e8
        return float(s)
    except ValueError:
        return np.nan


def to_yyyymmdd(s):
    """'20250930' / '2025-09-30' / '2025-09-30 00:00:00' -> '20250930'；脏值 -> None"""
    m = re.search(r'(\d{4})-?(\d{2})-?(\d{2})', str(s))
    return f'{m.group(1)}{m.group(2)}{m.group(3)}' if m else None


def code6(s):
    m = re.search(r'\d{6}', str(s))
    return m.group(0) if m else None


# ================================================================
# 2. 读取全部数据源
# ================================================================
def read_ifind():
    """iFinD 原始响应 -> 利润 {(code,period): (profit, api_growth, name)} + 日期标量 {(code,period): yyyymmdd}"""
    profit, dates, batch_stat = {}, {}, []
    for f in sorted(glob.glob(os.path.join(RAW, '01_financials_batch*.json'))):
        fname = os.path.basename(f)
        text = load_ifind_answer(f)
        if text is None:
            batch_stat.append((fname, 0, 0, '响应无效'))
            continue
        n_profit_rows = n_date_rows = 0
        for t in parse_markdown_tables(text):
            cols = list(t.columns)
            if any('归属于母公司所有者的净利润' in str(c) for c in cols) and '日期' in cols:
                profit_col = next(c for c in cols if str(c).startswith('归属于母公司所有者的净利润'))
                growth_col = next((c for c in cols if '归属母公司股东的净利润' in str(c) and '同比增长率' in str(c)), None)
                for _, r in t.iterrows():
                    key = (code6(r['证券代码']), to_yyyymmdd(r['日期']))
                    if key[0] and key[1]:
                        profit[key] = (to_profit(r[profit_col]),
                                       to_profit(r[growth_col]) if growth_col else np.nan,
                                       str(r['证券简称']))
                        n_profit_rows += 1
            elif any('定期报告实际披露日期' in str(c) for c in cols):
                period_tag = IFIND_DATE_PERIOD.get(fname)
                if period_tag is None:   # 该批次日期无报告期映射（v1无日期表 / v3歧义），不采用
                    continue
                for _, r in t.iterrows():
                    c = code6(r['证券代码'])
                    d = to_yyyymmdd(r['定期报告实际披露日期'])
                    if c and d:
                        dates[(c, period_tag)] = d
                        n_date_rows += 1
        batch_stat.append((fname, n_profit_rows, n_date_rows, 'ok'))
    return profit, dates, batch_stat


def read_akshare():
    """akshare 利润表 -> {(code,period): (parent_netprofit, notice_date)}"""
    akp = {}
    n_files = 0
    for f in sorted(glob.glob(os.path.join(RAW, 'akshare_profit_*.csv'))):
        c = code6(os.path.basename(f))
        df = pd.read_csv(f, encoding='utf-8-sig')
        n_files += 1
        for _, r in df.iterrows():
            p = to_yyyymmdd(r.get('REPORT_DATE'))
            if not c or not p:
                continue
            notice = to_yyyymmdd(r.get('NOTICE_DATE'))
            akp[(c, p)] = (float(r['PARENT_NETPROFIT']) if pd.notna(r.get('PARENT_NETPROFIT')) else np.nan, notice)
    return akp, n_files


def read_yysj():
    """东财预约披露表 -> {(code,period): (实际披露, 首次预约)}"""
    yysj = {}
    for period in PERIODS:
        f = os.path.join(RAW, f'akshare_yysj_{period}.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, encoding='utf-8-sig')
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
        for _, r in df.iterrows():
            yysj[(r['股票代码'], period)] = (to_yyyymmdd(r.get('实际披露时间')), to_yyyymmdd(r.get('首次预约时间')))
    return yysj


def load_prices():
    """data/cache 30只日线 -> {code: DataFrame(date indexed)}"""
    px = {}
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, 'stock_*.csv'))):
        c = code6(os.path.basename(f))
        df = pd.read_csv(f, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates('date').set_index('date')
        px[c] = df
    return px


# ================================================================
# 3. 事件表构建（yoy 超预期代理 + 公告日期）
# ================================================================
def build_events(ifind_profit, ifind_dates, ak_profit, yysj):
    rows = []
    codes = sorted({c for c, _ in ifind_profit} | {c for c, _ in ak_profit})
    for code in codes:
        name_of = ifind_profit.get((code, '20241231'), ('',))[2] if (code, '20241231') in ifind_profit \
            else next((v[2] for (c, _), v in ifind_profit.items() if c == code), '')
        for period, label in PERIODS.items():
            num_ifind = ifind_profit.get((code, period), (np.nan,))[0]
            num_ak = ak_profit.get((code, period), (np.nan, None))[0]
            den_period = DENOM[period]
            den_ifind = ifind_profit.get((code, den_period), (np.nan,))[0]
            den_ak = ak_profit.get((code, den_period), (np.nan, None))[0]
            num = num_ifind if pd.notna(num_ifind) else num_ak
            den = den_ifind if pd.notna(den_ifind) else den_ak
            num_src = 'iFinD' if pd.notna(num_ifind) else 'akshare'
            den_src = 'iFinD' if pd.notna(den_ifind) else 'akshare'
            # 同比增速：分母<=0 或缺失时无法定义（亏损基数同比无意义），该事件剔除出分组
            yoy = (num / den - 1) * 100 if (pd.notna(num) and pd.notna(den) and den > 0) else np.nan
            yoy_ak = (num_ak / den_ak - 1) * 100 if (pd.notna(num_ak) and pd.notna(den_ak) and den_ak > 0) else np.nan
            # 公告日期优先级：akshare NOTICE_DATE > 东财yysj实际披露 > 首次预约（预约日近似）
            notice = ak_profit.get((code, period), (np.nan, None))[1]
            ya, ys = yysj.get((code, period), (None, None))
            ann, ann_src = (notice, 'akshare NOTICE_DATE') if notice else (
                (ya, 'yysj实际披露') if ya else ((ys, 'yysj首次预约(预约日近似)') if ys else (None, '缺失')))
            rows.append({
                'code': code, 'name': name_of, 'period': period, 'period_label': label,
                'profit_num': num, 'profit_den': den, 'num_src': num_src, 'den_src': den_src,
                'yoy_pct': yoy, 'yoy_akshare_pct': yoy_ak,
                'ifind_api_growth_pct': ifind_profit.get((code, period), (np.nan, np.nan))[1],
                'ann_date': ann, 'ann_src': ann_src,
                'ifind_actual_same_period': ifind_dates.get((code, period)),
            })
    return pd.DataFrame(rows)


# ================================================================
# 4. 事件研究（均值调整模型）
# ================================================================
class EventStudy:
    """事件日 t0 = 公告日后首个交易日（非交易日顺延；用每只股票自己的交易日历，
    停牌日自然被"顺延"吸收）。估计窗 [-120,-20] 均值调整模型，事件窗 [-5,+40]。"""

    def __init__(self, px):
        self.px = px
        self.ret = {c: df['close'].pct_change().to_numpy() for c, df in px.items()}
        self.dates = {c: df.index.normalize() for c, df in px.items()}

    def align(self, code, ann_yyyymmdd):
        if code not in self.px or ann_yyyymmdd is None:
            return None
        pos = int(self.dates[code].searchsorted(pd.Timestamp(ann_yyyymmdd)))
        return pos if pos < len(self.dates[code]) else None

    def mu(self, code, t0):
        r = self.ret[code]
        if t0 + EST_LO < 0:
            return None
        return float(np.nanmean(r[t0 + EST_LO: t0 + EST_HI + 1]))

    def car_curve(self, code, t0):
        """[-5..40] 的 (AR, CAR, mu)；窗口越界返回 None"""
        r = self.ret[code]
        if t0 + EV_LO - 1 < 0 or t0 + EV_HI >= len(r) or t0 + EST_LO < 0:
            return None
        mu = self.mu(code, t0)
        ar = r[t0 + EV_LO: t0 + EV_HI + 1] - mu
        return ar, np.cumsum(ar), mu

    def car_slice(self, code, t0, a, b):
        """CAR[a,b]（相对t0的偏移）；窗口越界返回 None"""
        r = self.ret[code]
        if t0 + a - 1 < 0 or t0 + b >= len(r) or t0 + EST_LO < 0:
            return None
        return float(np.nansum(r[t0 + a: t0 + b + 1] - self.mu(code, t0)))


def classify(ev_df):
    """每期内按 yoy 排序：前1/3正面、后1/3负面、中间中性（floor(n/3)，n<9 则不分组）"""
    ev_df['group'] = '无法分类(yoy缺失)'
    for period, sub in ev_df.groupby('period'):
        ok = sub.dropna(subset=['yoy_pct']).index
        k = len(ok) // 3
        if k < 3:
            continue
        order = ev_df.loc[ok].sort_values('yoy_pct', ascending=False, kind='mergesort').index
        ev_df.loc[order[:k], 'group'] = '正面'
        ev_df.loc[order[len(order) - k:], 'group'] = '负面'
        ev_df.loc[order[k:len(order) - k], 'group'] = '中性'
    return ev_df


# ================================================================
# 5. 图表
# ================================================================
def fig01_car_curves(curves, out):
    taus = np.arange(EV_LO, EV_HI + 1)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    styles = [('正面', '#c0392b', '正面组(增速前1/3)'), ('负面', '#1e8449', '负面组(增速后1/3)'),
              ('中性', '#7f8c8d', '中性组(中间1/3)')]
    for grp, color, label in styles:
        cs = curves.get(grp)
        if cs is None or len(cs) == 0:
            continue
        m = np.mean(cs, axis=0)
        se = np.std(cs, axis=0, ddof=1) / np.sqrt(len(cs))
        ls = '-' if grp != '中性' else '--'
        ax.plot(taus, m * 100, color=color, lw=2, ls=ls, label=f'{label}（n={len(cs)}）')
        if grp != '中性':
            ax.fill_between(taus, (m - 1.96 * se) * 100, (m + 1.96 * se) * 100, color=color, alpha=0.15)
    ax.axvline(0, color='black', lw=1, ls=':')
    ax.axhline(0, color='black', lw=0.6)
    ax.axvspan(TEST_LO, TEST_HI, color='#f39c12', alpha=0.08)
    ax.text(TEST_LO + 0.5, ax.get_ylim()[0], ' 漂移检验窗 CAR[+2,+40]', fontsize=9, color='#b9770e', va='bottom')
    ax.set_xlabel('事件日相对偏移 τ（交易日，0=公告后首个交易日）')
    ax.set_ylabel('平均累计异常收益 CAR(%)（均值调整模型，阴影=±1.96×标准误）')
    ax.set_title(f'图1 盈余公告后漂移：正面/负面组的平均CAR曲线（估计窗[{EST_LO},{EST_HI}]，事件窗[{EV_LO},{EV_HI}]）')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig02_strategy(nav_df, cohort_marks, out):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(nav_df.index, nav_df['PEAD正面组策略'] * 100, color='#c0392b', lw=2,
            label=f"PEAD正面组策略（终值 {nav_df['PEAD正面组策略'].iloc[-1] * 100:.2f}）")
    ax.plot(nav_df.index, nav_df['等权基准(全部事件股)'] * 100, color='#2471a3', lw=1.8,
            label=f"等权基准（终值 {nav_df['等权基准(全部事件股)'].iloc[-1] * 100:.2f}）")
    for d, label in cohort_marks:
        ax.axvline(pd.Timestamp(d), color='#7d6608', lw=0.8, ls='--', alpha=0.6)
        ax.text(pd.Timestamp(d), ax.get_ylim()[1], f'{label}\n建仓', fontsize=8, color='#7d6608',
                ha='right', va='top')
    ax.axhline(100, color='black', lw=0.6)
    ax.set_xlabel('日期（三个财报期各自建仓持有一个季度，资金三等分，建仓前记为现金）')
    ax.set_ylabel('净值（期初=100，含双边万3成本）')
    ax.set_title(f'图2 简化版PEAD策略 vs 等权基准：公告后第2交易日收盘买入正面组，持有{HOLD}个交易日')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig03_placebo(plac_pos, plac_neg, real_pos, real_neg, pct_pos, pct_neg, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [
        (axes[0], plac_pos, real_pos, pct_pos, '#c0392b', '正面组'),
        (axes[1], plac_neg, real_neg, pct_neg, '#1e8449', '负面组'),
    ]
    for ax, plac, real, pct, color, gname in panels:
        plac = np.array(plac)
        ax.hist(plac * 100, bins=40, color='#aab7b8', edgecolor='white',
                label=f'假公告日组均值分布（{len(plac)}次抽样）')
        ax.axvline(real * 100, color=color, lw=2.5, label=f'真实事件组均值 {real * 100:+.2f}%')
        ax.set_title(f'{gname}：真实均值位于安慰剂分布第 {pct:.1f} 百分位\n'
                     f'（假公告日均值范围 {plac.min() * 100:+.2f}% ~ {plac.max() * 100:+.2f}%）', fontsize=11)
        ax.set_xlabel('CAR[+2,+40] 的组均值(%)')
        ax.set_ylabel('频数')
        ax.legend(fontsize=9)
    fig.suptitle(f'图3 安慰剂对照：随机假公告日（rng={RNG_SEED}，等量抽样）下的组均值分布 vs 真实事件组', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig04_example(es, ev, out):
    """单只股票事件示例：取 yoy 最高的正面事件"""
    code, t0 = ev['code'], int(ev['t0'])
    idx = es.dates[code]
    lo, hi = t0 + EST_LO, t0 + EV_HI
    closes = es.px[code]['close'].iloc[lo: hi + 1]
    curve = es.car_curve(code, t0)
    car = curve[1]
    taus_idx = idx[t0 + EV_LO: t0 + EV_HI + 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax1.plot(idx[lo: hi + 1], closes.to_numpy(), color='#2c3e50', lw=1.2)
    ax1.axvspan(idx[t0 + EST_LO], idx[t0 + EST_HI], color='#3498db', alpha=0.12,
                label=f'估计窗[{EST_LO},{EST_HI}]')
    ax1.axvline(idx[t0], color='#c0392b', lw=1.5,
                label=f'事件日 t0={idx[t0].date()}（公告日 {ev["ann_date"]}）')
    ax1.axvspan(taus_idx[TEST_LO - EV_LO], taus_idx[TEST_HI - EV_LO], color='#f39c12', alpha=0.10)
    ax1.set_ylabel('收盘价（前复权，元）')
    ax1.set_title(f'图4 单只股票事件示例：{ev["name"]}({code}) {ev["period_label"]}，'
                  f'同比 {ev["yoy_pct"]:+.1f}%（{ev["group"]}组），CAR[+2,+40]={ev["car_2_40"] * 100:+.2f}%')
    ax1.legend(loc='upper left', fontsize=9)
    ax2.bar(taus_idx, car * 100, width=1.0, color=np.where(car >= 0, '#c0392b', '#1e8449'))
    ax2.axhline(0, color='black', lw=0.6)
    ax2.set_ylabel('该事件CAR(%)')
    ax2.set_xlabel('日期（事件窗 τ=-5 ~ +40）')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ================================================================
# 6. 策略回测
# ================================================================
def cohort_nav(event_rows, es):
    """event_rows: 同一批建仓事件的 DataFrame（code/name/t0）。
    每个成员：以自身 t0+1 收盘买入、t0+1+HOLD 收盘卖出（持有60个交易日）。
    期内等权买入持有；资金按期三等分，期与期净值并列取均值（重叠期各自独立核算）。
    返回 (pooled_nav Series, 明细list, 每期首个买入日dict)。"""
    detail, nav_cols, first_buy = [], {}, {}
    for period, sub in event_rows.groupby('period'):
        ratios = {}
        for _, r in sub.iterrows():
            c, t0 = r['code'], int(r['t0'])
            e, x = t0 + 1, t0 + 1 + HOLD
            if x >= len(es.ret[c]):
                continue    # 60日持有窗不完整的成员剔除（如实计数）
            ser = es.px[c]['close'].iloc[e: x + 1]
            if len(ser) < 2 or pd.isna(ser.iloc[0]) or ser.iloc[0] <= 0:
                continue
            idx = es.dates[c][e: x + 1]
            ratios[c] = pd.Series(ser.to_numpy() / ser.iloc[0], index=idx)
            gross = ser.iloc[-1] / ser.iloc[0] - 1
            detail.append({'报告期': period, '期': PERIODS[period],
                           '代码': f"{c}.{'SH' if str(c).startswith('6') else 'SZ'}", '名称': r['name'],
                           '买入日': str(idx[0].date()), '卖出日': str(idx[-1].date()),
                           '持有收益(%)': round(gross * 100, 2),
                           '含成本收益(%)': round(((1 + gross) * (1 - COST) ** 2 - 1) * 100, 2)})
        if not ratios:
            continue
        cal = sorted(set().union(*[set(s.index) for s in ratios.values()]))
        mat = pd.DataFrame({c: s.reindex(cal).ffill() for c, s in ratios.items()})  # 停牌日沿用最近收盘价估值
        nav_cols[period] = mat.mean(axis=1)
        first_buy[period] = min(d['买入日'] for d in detail if d['报告期'] == period)
        if not nav_cols:
            return None, detail, first_buy
        # 合并日历：用全部30只股票的交易日并集（ restricted 到首建仓~最晚退出），
        # 否则两期持仓之间的空档会被整段挖掉（净值断线、日收益跨洞计算）
        lo_d = min(s.index[0] for s in nav_cols.values())
        hi_d = max(s.index[-1] for s in nav_cols.values())
        calendar = pd.DatetimeIndex(sorted(
            {d for dd in es.dates.values() for d in dd if lo_d <= d <= hi_d}))
        full = pd.DataFrame(index=calendar)
    for period, nav in nav_cols.items():
        full[period] = nav.reindex(calendar)
        fv = full[period].first_valid_index()
        full.loc[full.index < fv, period] = 1.0     # 建仓前=现金（收益0）
        full[period] = full[period].ffill()          # 卖出后停在退出净值（现金）
    return full.mean(axis=1), detail, first_buy


def perf(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    days = len(nav) - 1
    ann = (1 + total) ** (252 / days) - 1 if days > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(252)
    RF = 0.02  # 无风险利率（年化2%，与本系列课01-10统一口径）
    sharpe = float((r.mean() * 252 - RF) / (r.std(ddof=1) * np.sqrt(252))) if r.std(ddof=1) > 0 else np.nan
    mdd = float((1 - nav / nav.cummax()).max())
    return {'总收益率(%)': round(total * 100, 2), '年化收益率(%)': round(ann * 100, 2),
            '年化波动率(%)': round(vol * 100, 2), '夏普比率': round(sharpe, 3),
            '最大回撤(%)': round(mdd * 100, 2), '交易日数': days}


# ================================================================
# 主流程
# ================================================================
def main():
    meta = {}

    # ---- 读数据 ----
    ifind_profit, ifind_dates, batch_stat = read_ifind()
    ak_profit, n_ak = read_akshare()
    yysj = read_yysj()
    px = load_prices()
    log('数据', 'iFinD 批次解析统计（文件: 利润行 / 日期行）:')
    for b in batch_stat:
        log('数据', f'   {b[0]}: 利润{b[1]}行 / 日期{b[2]}行 / {b[3]}')
    log('数据', f'akshare 利润表 {n_ak} 只; yysj 披露表 {len(PERIODS)} 期; 价格序列 {len(px)} 只')
    meta['ifind_profit_cells'] = len(ifind_profit)
    meta['ak_profit_files'] = n_ak

    # ---- 交叉核验 1：iFinD 利润 vs akshare 利润（重叠 code×期）----
    diffs = []
    for key in sorted(set(ifind_profit) & set(ak_profit)):
        v1, v2 = ifind_profit[key][0], ak_profit[key][0]
        if pd.notna(v1) and pd.notna(v2) and abs(v2) > 1:
            diffs.append(abs(v1 - v2) / abs(v2))
    diffs = np.array(diffs)
    n_bad = int((diffs > 0.01).sum()) if len(diffs) else 0
    if len(diffs):
        log('核验', f'利润交叉核验：iFinD vs akshare 重叠 {len(diffs)} 格，'
                    f'相对差>1% 的 {n_bad} 格，最大相对差 {diffs.max() * 100:.4f}%')
        meta['profit_xcheck'] = {'n': int(len(diffs)), 'n_gt1pct': n_bad,
                                 'max_rel_diff_pct': round(float(diffs.max() * 100), 4)}

    # ---- 交叉核验 2：iFinD 日期标量 vs akshare NOTICE_DATE（验证"首提期"映射规则）----
    n_match = n_cmp = 0
    mismatch_examples = []
    for (c, p), d_ifind in sorted(ifind_dates.items()):
        if p not in XCHECK_PERIODS:
            continue
        d_ak = ak_profit.get((c, p), (np.nan, None))[1]
        if d_ak:
            n_cmp += 1
            if d_ifind == d_ak:
                n_match += 1
            else:
                # 找出 iFinD 标量实际等于该股哪一期的公告日（诊断期次映射错位）
                hit = [pp for pp in set(q for cc, q in ak_profit if cc == c)
                       if ak_profit.get((c, pp), (np.nan, None))[1] == d_ifind]
                mismatch_examples.append(f"{c}({p}->{d_ifind}, 实为{hit[0] if hit else '未知期'})")
    if n_cmp and n_match == n_cmp:
        verdict = '（首提期映射规则成立，iFinD日期可用作旁证）'
    elif n_cmp:
        verdict = (f'（不一致 {n_cmp - n_match}/{n_cmp} 格——NL解析器的期次映射在不同调用间不一致，'
                   f'错位样例: {"; ".join(mismatch_examples[:4])}。事件研究一律采用 akshare NOTICE_DATE，不受影响）')
    else:
        verdict = '（无可比数据）'
    log('核验', f'日期交叉核验：iFinD"实际披露日期"标量 vs akshare NOTICE_DATE 可比 {n_cmp} 格，一致 {n_match} 格 {verdict}')
    meta['date_xcheck'] = {'n': n_cmp, 'match': n_match, 'mismatch_examples': mismatch_examples[:6]}

    # ---- 交叉核验 3：akshare NOTICE_DATE vs yysj 实际披露 ----
    n_match3 = n_cmp3 = 0
    for (c, p), (ya, _) in sorted(yysj.items()):
        d_ak = ak_profit.get((c, p), (np.nan, None))[1]
        if d_ak and ya:
            n_cmp3 += 1
            n_match3 += int(d_ak == ya)
    log('核验', f'日期交叉核验：akshare NOTICE_DATE vs 东财yysj实际披露 可比 {n_cmp3} 格，一致 {n_match3} 格')
    meta['date_xcheck2'] = {'n': n_cmp3, 'match': n_match3}

    # ---- 构建事件表 ----
    ev = build_events(ifind_profit, ifind_dates, ak_profit, yysj)
    log('事件表', f"共 {len(ev)} 条（{len(PERIODS)}个财报期 × {ev['code'].nunique()}只）；"
                 f"公告日缺失 {int(ev['ann_date'].isna().sum())} 条；"
                 f"yoy不可算（分母<=0或缺失）{int(ev['yoy_pct'].isna().sum())} 条")
    cmp_yoy = ev.dropna(subset=['yoy_pct', 'ifind_api_growth_pct'])
    if len(cmp_yoy):
        dmax = (cmp_yoy['yoy_pct'] - cmp_yoy['ifind_api_growth_pct']).abs().max()
        log('核验', f'自算yoy vs iFinD自带同比增长率：可比 {len(cmp_yoy)} 条，最大绝对差 {dmax:.2f} 个百分点')
        meta['yoy_xcheck_maxdiff_pp'] = round(float(dmax), 2)

    # ---- 对齐事件日 + 每个事件的 CAR[+2,+40] ----
    es = EventStudy(px)
    t0_list, full_cnt, drop_cnt = [], 0, 0
    for i, row in ev.iterrows():
        t0 = es.align(row['code'], row['ann_date'])
        t0_list.append(t0)
        ev.at[i, 't0'] = t0
        ev.at[i, 't0_date'] = str(es.dates[row['code']][t0].date()) if t0 is not None else None
        c240 = es.car_slice(row['code'], t0, TEST_LO, TEST_HI) if t0 is not None else None
        ev.at[i, 'car_2_40'] = c240 if c240 is not None else np.nan
        curve = es.car_curve(row['code'], t0) if t0 is not None else None
        ev.at[i, 'full_window'] = curve is not None
        if curve is None:
            drop_cnt += 1
        else:
            full_cnt += 1
    ev['full_window'] = ev['full_window'].fillna(False).astype(bool)
    data_end = max(df.index.max() for df in px.values()).date()
    log('对齐', f'事件窗完整 {full_cnt} 条；公告日缺失或窗口越界剔除 {drop_cnt} 条（价格数据截至 {data_end}）')

    ev = classify(ev)
    valid = ev[ev['full_window'] & ev['group'].isin(['正面', '负面', '中性'])].copy()
    per_period = '; '.join(
        f"{lbl}: 正面{int((valid[valid['period_label'] == lbl]['group'] == '正面').sum())}/"
        f"负面{int((valid[valid['period_label'] == lbl]['group'] == '负面').sum())}"
        for lbl in PERIODS.values())
    log('分组', f'有效事件 {len(valid)} 条（每期内按yoy前/后1/3分组，floor(n/3)）。各期 {per_period}')

    # ---- 事件研究统计 ----
    curves = {}
    stat_rows = []
    for grp in ['正面', '负面', '中性']:
        sub = valid[valid['group'] == grp]
        if len(sub) == 0:
            continue
        xs = sub['car_2_40'].to_numpy()
        t, p = stats.ttest_1samp(xs, 0.0)
        stat_rows.append({'组': grp, '事件数': len(xs), 'CAR[+2,+40]均值(%)': round(xs.mean() * 100, 2),
                          '标准差(%)': round(xs.std(ddof=1) * 100, 2),
                          't统计量': round(float(t), 3), 'p值': round(float(p), 4),
                          '显著(5%)': '是' if p < 0.05 else '否'})
        cs = [es.car_curve(r['code'], int(r['t0']))[1] for _, r in sub.iterrows()]
        curves[grp] = np.array(cs)
    stat_df = pd.DataFrame(stat_rows)
    print('\n===== CAR[+2,+40] 截面 t 检验（真实事件）=====')
    print(stat_df.to_string(index=False))
    meta['ttest'] = stat_rows

    if '正面' in curves and '负面' in curves:
        x_pos = valid.loc[valid['group'] == '正面', 'car_2_40'].to_numpy()
        x_neg = valid.loc[valid['group'] == '负面', 'car_2_40'].to_numpy()
        tw, pw = stats.ttest_ind(x_pos, x_neg, equal_var=False)
        log('检验', f'正面-负面组 CAR[+2,+40] 差异 Welch t={tw:.3f}, p={pw:.4f}')
        meta['welch'] = {'t': round(float(tw), 3), 'p': round(float(pw), 4)}

    # ---- 安慰剂对照（rng=42，等量假公告日）----
    rng = np.random.default_rng(RNG_SEED)
    real_dates = pd.to_datetime(valid['t0_date'])
    d_lo, d_hi = real_dates.min(), real_dates.max()
    pool_codes, pool_cars = [], []
    for code in px:
        r = es.ret[code]
        n = len(r)
        idx = es.dates[code]
        for t0 in range(-EST_LO, n - TEST_HI):   # 估计窗与+40窗都完整
            d = idx[t0]
            if d < d_lo or d > d_hi:
                continue
            mu = float(np.nanmean(r[t0 + EST_LO: t0 + EST_HI + 1]))
            pool_codes.append((code, t0))
            pool_cars.append(float(np.nansum(r[t0 + TEST_LO: t0 + TEST_HI + 1] - mu)))
    pool_cars = np.array(pool_cars)
    log('安慰剂', f'候选普通交易日(股,日)对 {len(pool_cars)} 个（真实事件日跨度 {d_lo.date()} ~ {d_hi.date()}，窗口完整）')

    def placebo_draw(n_events):
        pick = rng.choice(len(pool_cars), size=n_events, replace=True)
        return float(pool_cars[pick].mean())

    n_pos = int((valid['group'] == '正面').sum())
    n_neg = int((valid['group'] == '负面').sum())
    real_pos = float(valid.loc[valid['group'] == '正面', 'car_2_40'].mean())
    real_neg = float(valid.loc[valid['group'] == '负面', 'car_2_40'].mean())
    plac_pos = [placebo_draw(n_pos) for _ in range(N_PLACEBO)]
    plac_neg = [placebo_draw(n_neg) for _ in range(N_PLACEBO)]
    pct_pos = float(np.mean(np.array(plac_pos) <= real_pos) * 100)
    pct_neg = float(np.mean(np.array(plac_neg) <= real_neg) * 100)
    log('安慰剂', f'正面组真实均值 {real_pos * 100:+.2f}%（n={n_pos}）位于 {N_PLACEBO} 次假公告日分布的第 {pct_pos:.1f} 百分位')
    log('安慰剂', f'负面组真实均值 {real_neg * 100:+.2f}%（n={n_neg}）位于第 {pct_neg:.1f} 百分位')
    meta['placebo'] = {'n_pos': n_pos, 'n_neg': n_neg, 'pct_pos': round(pct_pos, 1), 'pct_neg': round(pct_neg, 1),
                       'plac_pos_range': [round(min(plac_pos) * 100, 2), round(max(plac_pos) * 100, 2)],
                       'plac_neg_range': [round(min(plac_neg) * 100, 2), round(max(plac_neg) * 100, 2)]}

    # ---- 策略回测：公告后第2交易日收盘买入正面组，持有60个交易日 ----
    pos_rows = valid[valid['group'] == '正面']
    all_rows = valid[valid['group'].isin(['正面', '负面', '中性'])]
    strat_nav, strat_detail, strat_first = cohort_nav(pos_rows, es)
    bench_nav, bench_detail, bench_first = cohort_nav(all_rows, es)

    print('\n===== PEAD策略回测（双边各万3成本）=====')
    ps = pb = None
    if strat_nav is not None and bench_nav is not None:
        ps, pb = perf(strat_nav), perf(bench_nav)
        perf_df = pd.DataFrame([
            {'组合': f'PEAD正面组策略（{len(strat_detail)}个持仓事件）', **ps},
            {'组合': f'等权基准（全部{len(bench_detail)}个事件，同期同口径）', **pb},
        ])
        print(perf_df.to_string(index=False))
        meta['strategy'] = {'pead': ps, 'pead_events': len(strat_detail), 'bench': pb,
                            'bench_events': len(bench_detail)}
        # 两条净值起点不同（首个正面事件 vs 全部事件的首个建仓日），统一到并集日历，起点前视为现金
        common_idx = strat_nav.index.union(bench_nav.index)
        s_al = strat_nav.reindex(common_idx).ffill().fillna(1.0)
        b_al = bench_nav.reindex(common_idx).ffill().fillna(1.0)
        nav_df = pd.DataFrame({'PEAD正面组策略': s_al, '等权基准(全部事件股)': b_al})
        marks = [(strat_first[p], PERIODS[p]) for p in PERIODS if p in strat_first]
        fig02_strategy(nav_df, marks, os.path.join(OUT, '02_PEAD策略净值.png'))
        det = pd.DataFrame(strat_detail)
        print('\n正面组各期持仓明细（含成本）:')
        print(det.to_string(index=False))
    else:
        log('策略', '有效正面事件不足，策略未构建（如实记录）')

    # ---- 图1 / 图3 / 图4 ----
    fig01_car_curves(curves, os.path.join(OUT, '01_正负面组CAR曲线.png'))
    fig03_placebo(plac_pos, plac_neg, real_pos, real_neg, pct_pos, pct_neg,
                  os.path.join(OUT, '03_安慰剂对比.png'))
    pos_full = valid[valid['group'] == '正面']
    if len(pos_full):
        ex = pos_full.loc[pos_full['yoy_pct'].idxmax()]
        fig04_example(es, ex, os.path.join(OUT, '04_单只股票事件示例.png'))
        meta['example'] = {'code': ex['code'], 'name': ex['name'],
                           'yoy': round(float(ex['yoy_pct']), 1), 'period': ex['period_label']}

    # ---- 落盘 CSV ----
    ev_out = ev.copy()
    # 代码加交易所后缀（防止Excel/CSV读取时 000002 -> 2 丢前导零）
    ev_out['code'] = ev_out['code'].map(
        lambda c: f"{c}.{'SH' if str(c).startswith('6') else 'SZ'}")
    ev_out['car_2_40'] = (ev_out['car_2_40'] * 100).round(2)   # 小数 -> %
    for c in ['yoy_pct', 'yoy_akshare_pct', 'ifind_api_growth_pct']:
        ev_out[c] = ev_out[c].round(2)
    ev_out = ev_out.rename(columns={'period': '报告期', 'period_label': '财报期', 'ann_date': '公告日',
                                    'ann_src': '公告日来源', 't0_date': '事件日t0', 'group': '分组',
                                    'yoy_pct': '同比增速yoy(%)', 'yoy_akshare_pct': 'yoy(纯akshare口径,%)',
                                    'ifind_api_growth_pct': 'iFinD自带增速(%)', 'car_2_40': 'CAR[+2,+40](%)',
                                    'full_window': '事件窗完整', 'num_src': '分子来源', 'den_src': '分母来源',
                                    'profit_num': '归母净利润(元)', 'profit_den': '去年同期(元)',
                                    'ifind_actual_same_period': 'iFinD同报期披露日(核验)'})
    cols = ['code', 'name', '报告期', '财报期', '公告日', '公告日来源', '事件日t0', '分组', '同比增速yoy(%)',
            'yoy(纯akshare口径,%)', 'iFinD自带增速(%)', '归母净利润(元)', '去年同期(元)', '分子来源', '分母来源',
            'CAR[+2,+40](%)', '事件窗完整', 'iFinD同报期披露日(核验)']
    ev_out[cols].to_csv(os.path.join(OUT, 'event_details.csv'), index=False, encoding='utf-8-sig')

    mrows = []
    for _, r in stat_df.iterrows():
        mrows.append({'类别': '事件研究', '指标': f"{r['组']}组 CAR[+2,+40] 均值",
                      '数值': f"{r['CAR[+2,+40]均值(%)']}%",
                      '说明': f"n={r['事件数']}, t={r['t统计量']}, p={r['p值']}, 事件窗完整"})
    if 'welch' in meta:
        mrows.append({'类别': '事件研究', '指标': '正面-负面差异(Welch)', '数值': f"t={meta['welch']['t']}",
                      '说明': f"p={meta['welch']['p']}"})
    mrows.append({'类别': '安慰剂', '指标': '正面组均值百分位', '数值': f"{pct_pos:.1f}%",
                  '说明': f'{N_PLACEBO}次假公告日(rng={RNG_SEED}), 每次等量{n_pos}个'})
    mrows.append({'类别': '安慰剂', '指标': '负面组均值百分位', '数值': f"{pct_neg:.1f}%",
                  '说明': f'{N_PLACEBO}次假公告日(rng={RNG_SEED}), 每次等量{n_neg}个'})
    if ps and pb:
        for name, pf, n_ev in [('PEAD策略', ps, len(strat_detail)), ('等权基准', pb, len(bench_detail))]:
            for k, v in pf.items():
                mrows.append({'类别': f'回测-{name}', '指标': k, '数值': str(v),
                              '说明': f'双边各万3成本, {n_ev}个持仓事件'})
    pd.DataFrame(mrows).to_csv(os.path.join(OUT, 'pead_metrics.csv'), index=False, encoding='utf-8-sig')

    print('\n===== 运行摘要（供 README 引用）=====')
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    log('输出', 'outputs/ 已写入: 4张PNG + event_details.csv + pead_metrics.csv')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        log('致命', '主流程异常（完整堆栈已打印），按教学规范仍以退出码0结束，请根据堆栈修复后重跑')
        sys.exit(0)
