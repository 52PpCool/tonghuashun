# -*- coding: utf-8 -*-
"""
课程 24：从六组合复现 Fama-French 因子（不看黑箱，亲手算 SMB / HML）
====================================================================

这节课做一件大多数教科书只讲不做的事：
  1. 下载（或读缓存）Kenneth French 官网 2x3 六组合月度收益数据，
     亲手解析"说明文字 + 多个数据段"的学术脏格式，取出 Value-Weighted 月度段；
  2. 按官方定义，从六个组合把 SMB、HML 逐月算出来；
  3. 与官方三因子文件（data/cache/ff3_monthly.csv）逐月对照：
     相关系数、平均绝对偏差、最大偏差 —— 验证"你算的就是官方算的"；
  4. 画因子画像：累计收益（对数轴）、分年代均值、2010 年后 HML 的失效段；
  5. A股迁移思考：用 30 只 A 股按价格粗略分组，构造一个"粗糙的" SMB 代理因子；
  6. 长期视角：六组合 1926 年至今的年化收益对比，检验"小盘价值神话"最近 15 年是否还成立。

数据来源：https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/6_Portfolios_2x3_CSV.zip
随机种子 42；总运行时间 < 2 分钟；任何一步失败都会降级兜底，绝不以非 0 退出。
"""

import os
import io
import re
import glob
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示环境，纯出图文件
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 全局设置
SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')                    # 本课自己的数据/缓存
OUT_DIR = os.path.join(BASE_DIR, 'outputs')                  # 输出目录
CACHE_DIR = os.path.join(BASE_DIR, '..', 'data', 'cache')    # 工作区公共缓存（只读，不修改）

FRENCH_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/6_Portfolios_2x3_CSV.zip'
ZIP_PATH = os.path.join(DATA_DIR, '6_Portfolios_2x3_CSV.zip')          # 原始 zip
RAW_CACHE = os.path.join(DATA_DIR, 'six_portfolios_2x3.csv')           # 解压后的原始 CSV 缓存

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 中文显示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 六组合列名映射：French 文件里按 BM（账面市值比）命名，LoBM=成长、HiBM=价值
COL_MAP = {
    'SMALL LoBM': 'SG',   # 小盘成长 Small Growth
    'ME1 BM2':    'SN',   # 小盘中性 Small Neutral
    'SMALL HiBM': 'SV',   # 小盘价值 Small Value
    'BIG LoBM':   'BG',   # 大盘成长 Big Growth
    'ME2 BM2':    'BN',   # 大盘中性 Big Neutral
    'BIG HiBM':   'BV',   # 大盘价值 Big Value
}


def log(msg):
    print(msg, flush=True)


# ================================================================
# 第 1 步：获取原始数据（缓存 -> 本地 zip -> 下载 -> 合成兜底）
# ================================================================
def fetch_raw_text():
    """按优先级拿到 2x3 六组合的原始 CSV 文本；全部失败则返回 (合成文本, True)。"""
    # 1) 已解压的原始 CSV 缓存（第二遍运行走这里：离线可复现）
    if os.path.exists(RAW_CACHE):
        with open(RAW_CACHE, 'r', encoding='utf-8', errors='replace') as f:
            log(f'[数据] 命中缓存：{os.path.basename(RAW_CACHE)}')
            return f.read(), False

    # 2) 本地 zip（第一次运行常见：zip 在，但还没解压）
    if os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH) as z:
                name = z.namelist()[0]  # 6_Portfolios_2x3.csv
                text = z.read(name).decode('utf-8', errors='replace')
            with open(RAW_CACHE, 'w', encoding='utf-8') as f:   # 写缓存供下次离线用
                f.write(text)
            log(f'[数据] 从本地 zip 解出 {name} 并写入缓存')
            return text, False
        except Exception as e:
            log(f'[警告] 读取本地 zip 失败：{e}')

    # 3) 在线下载（教学点：学术数据站偶尔超时，必须设超时并兜底）
    try:
        log('[数据] 在线下载 French 2x3 zip ...')
        req = urllib.request.Request(FRENCH_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            blob = resp.read()
        with open(ZIP_PATH, 'wb') as f:
            f.write(blob)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            text = z.read(z.namelist()[0]).decode('utf-8', errors='replace')
        with open(RAW_CACHE, 'w', encoding='utf-8') as f:
            f.write(text)
        return text, False
    except Exception as e:
        log(f'[警告] 下载失败：{e}')

    # 4) 合成兜底（显著标注！）
    log('[警告] 全部获取方式失败 —— 改用随机种子42的合成六组合数据（仅保流程能跑，结论不可当真）')
    rng = np.random.default_rng(SEED)
    months = pd.period_range('1926-07', '2026-07', freq='M')
    yyyymm = [p.year * 100 + p.month for p in months]
    n = len(yyyymm)
    # 人为构造"小盘价值最赚、大盘成长最差"的结构，便于教学演示
    base = rng.normal(0.6, 5.0, n)
    tilt = {'SG': -0.2, 'SN': 0.1, 'SV': 0.5, 'BG': -0.5, 'BN': -0.1, 'SV_': 0.0}
    cols = {}
    for code, t in [('SG', tilt['SG']), ('SN', tilt['SN']), ('SV', tilt['SV']),
                    ('BG', tilt['BG']), ('BN', tilt['BN']), ('BV', tilt['SV'])]:
        cols[code] = base + t + rng.normal(0, 2.0, n)
    df = pd.DataFrame({'yyyymm': yyyymm, **cols})
    return df.to_csv(index=False), True


# ================================================================
# 第 2 步：解析"说明文字 + 多个数据段"的学术脏格式（教学点！）
# ================================================================
def parse_value_weighted_monthly(raw_text):
    """
    French 文件结构（以 2x3 为例）：
        头部说明文字（十几行）
        "  Average Value Weighted Returns -- Monthly"   <- 段标题
        ",SMALL LoBM,ME1 BM2,..."                       <- 列头（首列为空 = 日期列）
        192607,   1.0866, ...                           <- 数据行（yyyymm 开头）
        （空行）
        "  Average Equal Weighted Returns -- Monthly"   <- 下一段（容易拿错！）
        ...后面还有年度、公司数、平均市值等十余个数据段

    策略：定位段标题 -> 读列头 -> 只接收"6位年月开头"的数据行，遇到其他行即停。
    缺失值约定：-99.99 / -999 表示缺失，转成 NaN。
    """
    lines = raw_text.replace('\r\n', '\n').split('\n')

    # 找到 Value-Weighted Monthly 段标题所在行（容忍大小写与空格差异）
    seg_idx = None
    for i, ln in enumerate(lines):
        if re.search(r'Average\s+Value\s+Weighted\s+Returns\s+--\s+Monthly',
                     ln, flags=re.IGNORECASE):
            seg_idx = i
            break
    if seg_idx is None:
        raise ValueError('未找到 "Average Value Weighted Returns -- Monthly" 段标题')

    # 段标题后的第一个非空行是列头，形如 ",SMALL LoBM,ME1 BM2,..."
    col_idx = None
    for j in range(seg_idx + 1, min(seg_idx + 6, len(lines))):
        if lines[j].strip().startswith(','):
            col_idx = j
            break
    if col_idx is None:
        raise ValueError('未找到六组合的列头行')

    raw_cols = [c.strip() for c in lines[col_idx].strip().split(',')]
    codes = [COL_MAP.get(c, c) for c in raw_cols[1:]]  # 第一列是日期，跳过
    log(f'[解析] 列头：{raw_cols[1:]} -> 映射为 {codes}')

    # 只接收 yyyymm 开头的数据行；遇到空行/下一段标题/其他文字立即停止
    rows = []
    date_pat = re.compile(r'^\s*(\d{6})\s*,')
    for ln in lines[col_idx + 1:]:
        m = date_pat.match(ln)
        if not m:
            if rows:          # 已经收过数据又出现非数据行 => 本段结束
                break
            continue          # 段标题和列头之间的杂行，跳过
        parts = [p.strip() for p in ln.strip().split(',')]
        vals = []
        for p in parts[1:len(codes) + 1]:
            try:
                v = float(p)
                vals.append(np.nan if (v <= -99.0) else v)   # -99.99/-999 => 缺失
            except ValueError:
                vals.append(np.nan)
        rows.append([int(m.group(1))] + vals)

    df = pd.DataFrame(rows, columns=['yyyymm'] + codes).set_index('yyyymm')
    df = df.sort_index()
    log(f'[解析] Value-Weighted 月度段：{len(df)} 个月，'
        f'{df.index.min()} ~ {df.index.max()}，缺失值 {int(df.isna().sum().sum())} 个')
    return df


# ================================================================
# 第 3 步：本课灵魂 —— 按官方定义从六组合复现 SMB / HML
# ================================================================
def build_factors(six):
    """
    Fama-French (1993) 官方定义（单位：%）：
      SMB = (SV + SN + SG)/3 - (BV + BN + BG)/3
            （三个小盘组合等权平均  减  三个大盘组合等权平均）
      HML = (SV + BV)/2 - (SG + BG)/2
            （两个高BM"价值"组合平均  减  两个低BM"成长"组合平均）
    本质：SMB/HML 各是一个"多空对冲组合"——做多一腿、做空另一腿。
    """
    smb = six[['SV', 'SN', 'SG']].mean(axis=1) - six[['BV', 'BN', 'BG']].mean(axis=1)
    hml = six[['SV', 'BV']].mean(axis=1) - six[['SG', 'BG']].mean(axis=1)
    return pd.DataFrame({'SMB': smb, 'HML': hml})


def compare_with_official(rep, official):
    """逐月对照复现值与官方值，输出相关系数 / 偏差 / 符号一致率。"""
    joined = rep.join(official, how='inner', rsuffix='_官方')
    rows = []
    for fac in ['SMB', 'HML']:
        a, b = joined[fac], joined[f'{fac}_官方']
        diff = (a - b).dropna()
        corr = a.corr(b)
        stats = {
            '因子': fac,
            '对照月份数': int(len(diff)),
            '相关系数': round(float(corr), 8),
            '平均绝对偏差(pp)': round(float(diff.abs().mean()), 4),
            '最大绝对偏差(pp)': round(float(diff.abs().max()), 4),
            '符号一致率(%)': round(float((np.sign(a) == np.sign(b)).mean() * 100), 2),
        }
        rows.append(stats)
    return pd.DataFrame(rows).set_index('因子'), joined


# ================================================================
# 工具：年化、年代分组
# ================================================================
def annualize_pct(ret_pct):
    """月收益(%) 序列 -> 年化收益(%)（几何口径）。总净值<=0 时返回 NaN。"""
    r = pd.Series(ret_pct).dropna() / 100.0
    if len(r) == 0:
        return np.nan
    total = float((1 + r).prod())
    if total <= 0:
        return np.nan
    return (total ** (12.0 / len(r)) - 1.0) * 100.0


def decade_of(yyyymm):
    return (yyyymm // 100 // 10) * 10


# ================================================================
# 第 4 步：A 股"粗糙代理"SMB（教学演示，结论不可当真）
# ================================================================
def ashare_proxy_smb():
    """
    用 data/cache/ 的 30 只 A 股构造大小盘代理组合：
      - 按 2018 年首个可得月收盘价排序（注意：价格≠市值！这是刻意的粗糙代理，
        茅台单价高不等于市值大；真实 SMB 排序用的是流通/总市值）；
      - 一半进"大盘组"、一半进"小盘组"（等权）；
      - A股SMB代理 = 小盘组月均收益 - 大盘组月均收益；
      - 与沪深300（缓存为 510300 ETF 序列）对照。
    局限：30 只全是今天的知名公司 => 幸存者偏差；无真实市值/估值数据。
    这里只演示"双重排序 -> 多空因子"的方法骨架。
    """
    files = sorted(glob.glob(os.path.join(CACHE_DIR, 'stock_*.csv')))
    if len(files) == 0:
        log('[A股] 找不到 stock_*.csv，跳过 A 股代理部分')
        return None

    closes = {}
    for f in files:
        name = os.path.basename(f).replace('stock_', '').replace('.csv', '')
        s = pd.read_csv(f, parse_dates=['date'], index_col='date')['close'].dropna()
        closes[name] = s
    panel_daily = pd.DataFrame(closes)
    try:
        m_close = panel_daily.resample('ME').last()
    except ValueError:  # 兼容旧版 pandas 的 'M' 别名
        m_close = panel_daily.resample('M').last()

    # 2018 年初的价格：优先取 2018-01；当年晚些才上市（如迈瑞医疗 2018-10）的，
    # 用其 2018 年首个可得月价 —— 这本身就是"脏数据要显式交代"的教学点
    m2018 = m_close.loc['2018-01-01':'2018-12-31']
    price0 = m2018.iloc[0].copy()
    late_listed = []
    for c in price0.index[price0.isna()]:
        s = m2018[c].dropna()
        if len(s) > 0:
            price0[c] = s.iloc[0]
            late_listed.append(c)
    price0 = price0.dropna()
    log(f'[A股] 参与排序的股票 {len(price0)} 只；2018年内晚上市改用上市首月价的：{late_listed}')

    ordered = price0.sort_values()
    half = len(ordered) // 2
    small_names = list(ordered.index[:half])   # 低价一半 => "小盘组"（粗糙代理）
    big_names = list(ordered.index[half:2 * half])
    log(f'[A股] 小盘组(低价15只)：{small_names}')
    log(f'[A股] 大盘组(高价15只)：{big_names}')

    m_ret = m_close.pct_change()
    start = m_ret.index >= '2018-02-28'  # 2018-01 作为基期，收益从 2018-02 开始
    small_ret = m_ret.loc[start, small_names].mean(axis=1)
    big_ret = m_ret.loc[start, big_names].mean(axis=1)
    smb_proxy = (small_ret - big_ret).dropna()

    # 沪深300（缓存实际是 510300 ETF 净值，走势可作指数代理；末尾可能有空值行）
    idx_path = os.path.join(CACHE_DIR, 'index_000300_沪深300.csv')
    hs300 = pd.Series(dtype=float)
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, parse_dates=['date'], index_col='date')['close']
        idx = idx.dropna()
        try:
            idx_m = idx.resample('ME').last()
        except ValueError:
            idx_m = idx.resample('M').last()
        hs300 = idx_m.pct_change().reindex(smb_proxy.index).dropna()
        common = smb_proxy.index.intersection(hs300.index)
        smb_proxy = smb_proxy.loc[common]
        hs300 = hs300.loc[common]

    # 统一单位：pct_change() 给的是小数，本课因子/年化工具一律用"百分数"，
    # 这里显式 ×100 —— 教学点：单位混乱是量化最常见的隐形 bug
    smb_proxy = smb_proxy * 100.0
    small_ret = small_ret * 100.0
    big_ret = big_ret * 100.0
    hs300 = hs300 * 100.0

    corr = smb_proxy.corr(hs300) if len(hs300) > 1 else np.nan
    stats = {
        '股票数': len(price0),
        '起始月': str(smb_proxy.index.min().date()) if len(smb_proxy) else '-',
        '结束月': str(smb_proxy.index.max().date()) if len(smb_proxy) else '-',
        '代理SMB年化(%)': round(annualize_pct(smb_proxy), 2),
        '小盘组年化(%)': round(annualize_pct(small_ret.reindex(smb_proxy.index)), 2),
        '大盘组年化(%)': round(annualize_pct(big_ret.reindex(smb_proxy.index)), 2),
        '沪深300年化(%)': round(annualize_pct(hs300), 2),
        '代理SMB与沪深300月收益相关': round(float(corr), 3) if pd.notna(corr) else np.nan,
    }
    log(f'[A股] 代理 SMB 统计：{stats}')
    return {'smb_proxy': smb_proxy, 'small_ret': small_ret, 'big_ret': big_ret,
            'hs300': hs300, 'stats': stats, 'small_names': small_names,
            'big_names': big_names, 'late_listed': late_listed}


# ================================================================
# 主流程
# ================================================================
def main():
    log('=' * 72)
    log('课程24：从 2x3 六组合复现 Fama-French 因子')
    log('=' * 72)

    # ---- 1. 获取并解析六组合数据 ----
    raw_text, is_synthetic = fetch_raw_text()
    six = parse_value_weighted_monthly(raw_text)
    if is_synthetic:
        log('[警告] 当前使用合成数据！以下所有结论仅验证代码流程，不代表真实市场。')
    tag = '（合成数据）' if is_synthetic else ''

    # 输出六组合月度收益表
    six_out = six.reset_index()
    six_out.to_csv(os.path.join(OUT_DIR, 'six_portfolios.csv'),
                   index=False, encoding='utf-8-sig')

    # ---- 2. 复现 SMB / HML 并与官方对照 ----
    rep = build_factors(six)
    ff3_path = os.path.join(CACHE_DIR, 'ff3_monthly.csv')
    official = pd.read_csv(ff3_path).set_index('yyyymm')[['SMB', 'HML']]
    stats_df, joined = compare_with_official(rep, official)
    stats_df.to_csv(os.path.join(OUT_DIR, 'replication_stats.csv'),
                    encoding='utf-8-sig')
    log('\n[复现 vs 官方] 统计表：')
    log(stats_df.to_string())

    # ---- 3. 因子画像：分年代均值表 ----
    df = joined.copy()
    df['年代'] = df.index.map(decade_of)
    decade_rows = []
    for d, g in df.groupby('年代'):
        decade_rows.append({
            '年代': f'{d}s',
            '月数': len(g),
            'SMB月均(%)': round(g['SMB'].mean(), 3),
            'HML月均(%)': round(g['HML'].mean(), 3),
            'SMB年化(%)': round(annualize_pct(g['SMB']), 2),
            'HML年化(%)': round(annualize_pct(g['HML']), 2),
            'SV小盘价值年化(%)': round(annualize_pct(six.loc[g.index, 'SV']), 2),
            'BG大盘成长年化(%)': round(annualize_pct(six.loc[g.index, 'BG']), 2),
        })
    decade_tab = pd.DataFrame(decade_rows)
    decade_tab.to_csv(os.path.join(OUT_DIR, 'decade_table.csv'),
                      index=False, encoding='utf-8-sig')
    log('\n[分年代均值表]：')
    log(decade_tab.to_string(index=False))

    # ---- 4. A股粗糙代理 SMB ----
    log('')
    ashare = ashare_proxy_smb()

    # ==============================================================
    # 作图
    # ==============================================================
    yrs = pd.to_datetime(df.index.astype(str), format='%Y%m')

    # ---- 图1/图2：复现 vs 官方 散点（45度线）----
    for k, fac in enumerate(['SMB', 'HML'], start=1):
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        lim = max(df[fac].abs().max(), df[f'{fac}_官方'].abs().max()) * 1.08
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='45°线（完全一致）')
        ax.scatter(df[f'{fac}_官方'], df[fac], s=8, alpha=0.35,
                   color='#C0392B' if fac == 'SMB' else '#2471A3', edgecolors='none')
        r = stats_df.loc[fac, '相关系数']
        mad = stats_df.loc[fac, '平均绝对偏差(pp)']
        mx = stats_df.loc[fac, '最大绝对偏差(pp)']
        ax.set_title(f'图{k} {fac}：亲手复现 vs 官方{tag}\n'
                     f'相关系数 r={r:.8f}  平均|偏差|={mad:.3f}pp  最大|偏差|={mx:.3f}pp')
        ax.set_xlabel(f'官方 {fac}（% / 月）')
        ax.set_ylabel(f'复现 {fac}（% / 月）')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'0{k}_复现vs官方{fac}散点.png'), dpi=150)
        plt.close(fig)

    # ---- 图3：SMB/HML 累计收益（左：1926~今 对数轴；右：2010后线性）----
    cum = (1 + df[['SMB', 'HML']] / 100).cumprod()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ax.plot(yrs, cum['SMB'], label='SMB 复现', color='#C0392B', lw=1.2)
    ax.plot(yrs, cum['HML'], label='HML 复现', color='#2471A3', lw=1.2)
    # 官方值虚线叠画，验证两条线重合
    cum_o = (1 + df[['SMB_官方', 'HML_官方']] / 100).cumprod()
    ax.plot(yrs, cum_o['SMB_官方'], '--', color='black', lw=0.7, label='SMB 官方(虚线)')
    ax.plot(yrs, cum_o['HML_官方'], ':', color='gray', lw=0.7, label='HML 官方(点线)')
    ax.set_yscale('log')
    ax.set_title(f'图3a SMB / HML 累计净值（1926-07 起，对数轴）{tag}\n'
                 f'多空因子也能长期复利，但斜率取决于你活在哪个年代')
    ax.set_ylabel('累计净值（起始=1，对数刻度）')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    mask10 = df.index >= 201001
    cum10 = (1 + df.loc[mask10, ['SMB', 'HML']] / 100).cumprod()
    yrs10 = yrs[mask10]
    ax.plot(yrs10, cum10['HML'], color='#2471A3', lw=1.5, label='HML（价值-成长）')
    ax.plot(yrs10, cum10['SMB'], color='#C0392B', lw=1.2, label='SMB（小-大）')
    ax.axhline(1.0, color='black', lw=0.8)
    hml10_end = float(cum10['HML'].iloc[-1])
    ax.annotate(f'HML 2010 以来累计 {hml10_end:.2f}x\n（成长长期跑赢价值）',
                xy=(0.03, 0.06), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round', fc='#FDEBD0', alpha=0.9))
    ax.set_title(f'图3b 2010 年以后：HML 的"失效段"{tag}\n'
                 f'——因子有拥挤与周期，长期均值不等于未来预期')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_SMB_HML累计收益.png'), dpi=150)
    plt.close(fig)

    # ---- 图4：分年代因子均值 ----
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(decade_tab))
    w = 0.38
    ax.bar(x - w / 2, decade_tab['SMB月均(%)'], w, label='SMB 月均', color='#C0392B')
    ax.bar(x + w / 2, decade_tab['HML月均(%)'], w, label='HML 月均', color='#2471A3')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(decade_tab['年代'], rotation=45)
    ax.set_ylabel('月均收益（%）')
    ax.set_title(f'图4 分年代看 SMB / HML：因子轮动一目了然{tag}\n'
                 '小盘/价值并非永远有效——每个年代都有赢家和输家')
    for i, (s, h) in enumerate(zip(decade_tab['SMB月均(%)'], decade_tab['HML月均(%)'])):
        ax.text(i - w / 2, s + (0.03 if s >= 0 else -0.09), f'{s:.2f}',
                ha='center', fontsize=8)
        ax.text(i + w / 2, h + (0.03 if h >= 0 else -0.09), f'{h:.2f}',
                ha='center', fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_分年代因子均值.png'), dpi=150)
    plt.close(fig)

    # ---- 图5：六组合长期年化 vs 最近15年 ----
    full_ann = {c: annualize_pct(six[c]) for c in ['SV', 'SN', 'SG', 'BV', 'BN', 'BG']}
    mask15 = six.index >= 201107           # 最近15年：2011-07 ~ 2026-07
    recent_ann = {c: annualize_pct(six.loc[mask15, c])
                  for c in ['SV', 'SN', 'SG', 'BV', 'BN', 'BG']}
    name_cn = {'SV': '小盘价值', 'SN': '小盘中性', 'SG': '小盘成长',
               'BV': '大盘价值', 'BN': '大盘中性', 'BG': '大盘成长'}
    order = sorted(full_ann, key=full_ann.get, reverse=True)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, [full_ann[c] for c in order], w,
           label='1926-07 ~ 2026-07（百年）', color='#7D6608')
    ax.bar(x + w / 2, [recent_ann[c] for c in order], w,
           label='2011-07 ~ 2026-07（最近15年）', color='#B7950B')
    for i, c in enumerate(order):
        ax.text(i - w / 2, full_ann[c] + 0.15, f'{full_ann[c]:.1f}',
                ha='center', fontsize=8.5)
        ax.text(i + w / 2, recent_ann[c] + 0.15, f'{recent_ann[c]:.1f}',
                ha='center', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{name_cn[c]}\n{c}' for c in order])
    ax.set_ylabel('年化收益（%，几何）')
    ax.set_title(f'图5 六组合年化收益：百年 vs 最近15年{tag}\n'
                 '"小盘价值百年神话"在最近15年还成立吗？——把两组柱子对一对')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '05_六组合长期年化.png'), dpi=150)
    plt.close(fig)

    # ---- 图6（附赠）：A股粗糙代理 SMB vs 沪深300 ----
    if ashare is not None and len(ashare['smb_proxy']) > 0:
        sp, sr, br, hs = (ashare['smb_proxy'], ashare['small_ret'],
                          ashare['big_ret'], ashare['hs300'])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        ax = axes[0]
        idx_common = sr.index.intersection(hs.index)
        ax.plot(idx_common, (1 + sr.reindex(idx_common) / 100).cumprod(),
                label='小盘组（低价15只，等权）', color='#C0392B')
        ax.plot(idx_common, (1 + br.reindex(idx_common) / 100).cumprod(),
                label='大盘组（高价15只，等权）', color='#2471A3')
        ax.plot(idx_common, (1 + hs.reindex(idx_common) / 100).cumprod(),
                label='沪深300（缓存为510300 ETF）', color='black', lw=1.2)
        ax.set_yscale('log')
        ax.set_title(f'图6a A股"价格分组"代理组合（2018 年起，对数轴）\n'
                     '粗糙代理：用股价代替市值分组，仅演示方法')
        ax.set_ylabel('累计净值（起始=1，对数刻度）')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which='both')

        ax = axes[1]
        ax.plot(sp.index, (1 + sp / 100).cumprod(), color='#6C3483', lw=1.5,
                label='A股 SMB 代理（小-大，多空）')
        ax.axhline(1.0, color='black', lw=0.8)
        if len(hs) > 0:
            ax2 = ax.twinx()
            ax2.plot(hs.index, (1 + hs / 100).cumprod(), color='gray', lw=1.0,
                     linestyle='--', label='沪深300（右轴）')
            ax2.legend(fontsize=9, loc='lower right')
        corr = ashare['stats']['代理SMB与沪深300月收益相关']
        ax.set_title(f'图6b A股 SMB 代理因子（2018 起）\n'
                     f'与沪深300月收益相关 r={corr}；30只知名股 => 幸存者偏差严重')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, '06_A股SMB代理.png'), dpi=150)
        plt.close(fig)

    # ==============================================================
    # 汇总打印（README 的数字都来自这里）
    # ==============================================================
    hml10_end = float((1 + df.loc[df.index >= 201001, 'HML'] / 100).cumprod().iloc[-1])
    smb10_end = float((1 + df.loc[df.index >= 201001, 'SMB'] / 100).cumprod().iloc[-1])
    log('\n' + '=' * 72)
    log('[关键结果汇总]')
    log(f'  六组合样本：{six.index.min()} ~ {six.index.max()}，共 {len(six)} 个月{tag}')
    for fac in ['SMB', 'HML']:
        log(f'  {fac} 复现：相关系数 {stats_df.loc[fac, "相关系数"]}, '
            f'平均绝对偏差 {stats_df.loc[fac, "平均绝对偏差(pp)"]}pp, '
            f'最大绝对偏差 {stats_df.loc[fac, "最大绝对偏差(pp)"]}pp, '
            f'符号一致率 {stats_df.loc[fac, "符号一致率(%)"]}%')
    log(f'  2010 以来累计：SMB {smb10_end:.3f}x，HML {hml10_end:.3f}x（1.0 为不赚不亏）')
    best_smb = decade_tab.loc[decade_tab['SMB月均(%)'].idxmax(), '年代']
    worst_hml = decade_tab.loc[decade_tab['HML月均(%)'].idxmin(), '年代']
    log(f'  SMB 月均最高的年代：{best_smb}；HML 月均最低的年代：{worst_hml}')
    log(f'  百年年化：小盘价值 {full_ann["SV"]:.2f}% vs 大盘成长 {full_ann["BG"]:.2f}%')
    log(f'  最近15年年化：小盘价值 {recent_ann["SV"]:.2f}% vs 大盘成长 {recent_ann["BG"]:.2f}%')
    if ashare is not None:
        log(f'  A股代理SMB：年化 {ashare["stats"]["代理SMB年化(%)"]}%，'
            f'与沪深300相关 {ashare["stats"]["代理SMB与沪深300月收益相关"]}')
    log('=' * 72)
    log('全部完成，输出目录：outputs/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
