# -*- coding: utf-8 -*-
"""
parse_and_screen.py —— 第17课「iFinD真实数据管道」第二~四步：解析 → 清洗 → 打分 → 输出

输入:  raw/*.json        (fetch_ifind.js 留痕的接口原始响应)
       ../../data/cache/stock_*.csv (备用对照数据，30只A股日线)
输出:  outputs/all_scores.csv        全部候选打分明细 (utf-8-sig)
       outputs/screen_result.csv     综合分前10名股票池清单 (utf-8-sig)
       outputs/cache_vs_ifind.csv    iFinD vs 本地缓存 同口径动量对照 (utf-8-sig)
       outputs/01_ROE对PE散点.png / 02_综合得分前10条形图.png / 03_数据缺失情况.png

运行:  /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python parse_and_screen.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 基础配置
BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'raw')
OUT = os.path.join(BASE, 'outputs')
CACHE_DIR = os.path.join(os.path.dirname(BASE), 'data', 'cache')
os.makedirs(OUT, exist_ok=True)

# 中文字体（Linux 常见 CJK 字体，按优先级回退）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

FORCE_FALLBACK = os.environ.get('FORCE_FALLBACK') == '1'  # 测试降级路径用


def log(step, msg):
    print(f'[{step}] {msg}')


# ================================================================
# 第 1 步：解析 —— 拆"俄罗斯套娃" + markdown 表格转 DataFrame
# ================================================================
def unwrap_text(raw):
    """iFinD 的返回是三层套娃:
    MCP层 data.result.content[0].text
      -> 业务层 {"code":1,"msg":"success","data":"<内层还是JSON字符串>"}
      -> 最内层 {"answer":"# 标题...\n| 表格 |...","selectedSecuritiesCount":262}
    本函数把三层全部拆开，返回最内层的 markdown 正文。"""
    text = raw['data']['result']['content'][0]['text']
    for _ in range(2):
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            break
        if isinstance(obj, dict) and isinstance(obj.get('data'), str):
            text = obj['data']      # 还有一层
            continue
        if isinstance(obj, dict) and isinstance(obj.get('answer'), str):
            text = obj['answer']    # 到底了
            break
        break
    return text


_CELL_TRIM = re.compile(r'^\s*|\s*$')
_SEP_CELL = re.compile(r'^:?-{2,}:?$')


def parse_markdown_tables(text):
    """把 markdown 正文里的所有 '|' 分隔表格解析成 DataFrame 列表。
    真实接口的表格常有: 表头下的 |---|---| 分隔行、单元格两端空格、
    多张表挤在同一段返回里 —— 都要在这里处理掉。"""
    tables, current = [], []
    for line in text.split('\n'):
        if line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            # 跳过 |---|---| 这类分隔行
            if cells and all(_SEP_CELL.match(c) for c in cells):
                continue
            current.append(cells)
        else:
            if current:
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


def to_num(s):
    """脏数据数值转换: '12.34%' / '1,234' / '-' / '' / 科学计数法 '2.27E11' 都要能处理"""
    if s is None:
        return np.nan
    s = str(s).strip().replace('%', '').replace(',', '').replace('亿', '')
    if s in ('', '-', '--', 'nan', 'None', '无'):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# 列名规范化：接口每次返回的表头写法可能不同（含单位、含日期），统一映射成标准名
COL_MAP = [
    (r'^证券代码$|^股票代码$', 'code'),
    (r'^证券简称$|^股票简称$', 'name'),
    (r'^市盈率', 'pe'),
    (r'净资产收益率|ROE', 'roe'),
    (r'^归属母公司股东的净利润.*同比增长', 'growth'),      # 优先归母口径
    (r'^净利润.*同比增长', 'profit_growth_alt'),           # 备选口径，留痕
    (r'资产负债率', 'debt_ratio'),
    (r'收盘价', 'close'),
    (r'涨跌幅', 'mom60'),
    (r'总市值', 'mktcap'),
    (r'^日期$', 'date'),
]


def norm_cols(df):
    ren = {}
    for col in df.columns:
        c = _CELL_TRIM.sub('', str(col))
        for pat, std in COL_MAP:
            if re.search(pat, c) and std not in ren.values():
                ren[col] = std
                break
    return df.rename(columns=ren)


# ================================================================
# 主流程：iFinD 数据管道
# ================================================================
def run_ifind_pipeline():
    search_path = os.path.join(RAW, '01_search.json')
    if FORCE_FALLBACK or not os.path.exists(search_path):
        return None

    # ---- 1. 候选池：优先用全量CSV(262只)，否则用返回表格里被截断的前100行 ----
    full_csv = os.path.join(RAW, '01_search_full.csv')
    if os.path.exists(full_csv):
        pool_df = pd.read_csv(full_csv, encoding='utf-8-sig')
        pool_df.columns = ['code', 'name', 'market', 'mktcap']
        pool_df['mktcap'] = pool_df['mktcap'].map(to_num) / 1e8  # 元 -> 亿元
        pool_src = 'raw/01_search_full.csv (接口提供的全量CSV下载，24h有效)'
    else:
        tables = parse_markdown_tables(unwrap_text(json.load(open(search_path, encoding='utf-8'))))
        pool_df = norm_cols(tables[0])[['code', 'name', 'mktcap']].copy()
        pool_df['mktcap'] = pool_df['mktcap'].map(to_num) / 1e8
        pool_src = 'raw/01_search_text.txt (返回被截断，仅前100行)'

    meta = {'pool_source': pool_src, 'pool_size': len(pool_df)}
    log('候选池', f"共 {len(pool_df)} 只 (来源: {pool_src})")

    # ---- 2. 深挖名单（fetch 阶段按市值取前20，这是调用预算约束下的分层取数）----
    pool_json = os.path.join(RAW, 'pool.json')
    detail_pool = json.load(open(pool_json, encoding='utf-8'))
    detail_codes = {s['code'] for s in detail_pool}
    meta['detail_size'] = len(detail_codes)
    log('深挖名单', f"{len(detail_codes)} 只 (候选池按总市值降序前{len(detail_codes)}，受接口调用预算约束)")

    # ---- 3. 解析财务与行情批次 ----
    fin_frames, perf_frames = [], []
    for f in sorted(glob.glob(os.path.join(RAW, '02_financials_*.json'))):
        text = unwrap_text(json.load(open(f, encoding='utf-8')))
        for t in parse_markdown_tables(text):
            t = norm_cols(t)
            if 'pe' in t.columns:
                fin_frames.append(('pe', t))
            elif 'roe' in t.columns:
                fin_frames.append(('fund', t))
    for f in sorted(glob.glob(os.path.join(RAW, '03_performance_*.json'))):
        text = unwrap_text(json.load(open(f, encoding='utf-8')))
        for t in parse_markdown_tables(text):
            if 'mom60' in norm_cols(t).columns:
                perf_frames.append(norm_cols(t))

    pe_df = pd.concat([t for k, t in fin_frames if k == 'pe'], ignore_index=True)
    fund_df = pd.concat([t for k, t in fin_frames if k == 'fund'], ignore_index=True)
    perf_df = pd.concat(perf_frames, ignore_index=True)

    # ---- 关键清洗: "同一查询两次调用返回结构都不同" ----
    # 实测: 行情接口一次返回每股1行的最新快照(880字节)，另一次返回每股多行的时间序列
    # (300条/批，6505字节)。所以合并前必须执行"每个代码只保留日期最新一行"，
    # 不能指望接口返回形状稳定。
    def dedup_latest(df, label):
        n0 = len(df)
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = df['date'].astype(str)          # YYYYMMDD 字典序=时间序
            df = df.sort_values('date', ascending=False)
        df = df.drop_duplicates(subset='code', keep='first')
        log('清洗', f'{label}: 原始{n0}行 -> 每代码保留最新1行后 {len(df)}行')
        return df

    pe_df = dedup_latest(pe_df, 'PE表')
    fund_df = dedup_latest(fund_df, '财务表')
    perf_df = dedup_latest(perf_df, '行情表')
    meta['perf_rows_raw'] = int(len(perf_frames) and sum(len(t) for t in perf_frames))

    # 财务增速口径：归母净利润增速优先，缺了才用净利润增速（教学点：同一家公司两个口径会不同）
    if 'growth' not in fund_df.columns:
        fund_df['growth'] = fund_df['profit_growth_alt']
        meta['growth_col'] = '净利润(同比增长率)'
    else:
        meta['growth_col'] = '归属母公司股东的净利润(同比增长率)（接口同时返回两种口径，优先归母）'

    # ---- 4. 合并 + 清洗 ----
    df = (pool_df[pool_df['code'].isin(detail_codes)]
          .merge(fund_df[['code', 'roe', 'growth', 'debt_ratio']], on='code', how='left')
          .merge(pe_df[['code', 'pe']], on='code', how='left')
          .merge(perf_df[['code', 'close', 'mom60']], on='code', how='left'))
    meta['merged_size'] = len(df)
    meta['dup_dropped'] = int(len(df) - len(df.drop_duplicates(subset='code')))
    df = df.drop_duplicates(subset='code').reset_index(drop=True)

    for c in ['mktcap', 'roe', 'growth', 'debt_ratio', 'pe', 'close', 'mom60']:
        df[c] = df[c].map(to_num)

    # 剔除上市不足1年：本次接口未返回上市日期字段，无法执行该过滤 —— 如实记录而非假装过滤
    meta['listing_filter'] = '接口未返回上市日期字段，本步过滤未执行（真实生产中应从基本资料接口补齐）'

    meta['missing'] = {c: int(df[c].isna().sum()) for c in ['mktcap', 'roe', 'growth', 'debt_ratio', 'pe', 'close', 'mom60']}
    valid = df.dropna(subset=['roe', 'growth', 'pe', 'mom60']).reset_index(drop=True)
    meta['valid_size'] = len(valid)
    meta['pool_codes'] = [c.split('.')[0] for c in pool_df['code']]  # 6位代码，供缓存对照用
    log('清洗', f"合并后 {len(df)} 只; 剔重 {meta['dup_dropped']} 只; 关键字段缺失 {sum(meta['missing'].values())} 处; 可打分 {len(valid)} 只")

    if len(valid) == 0:
        return None

    # ---- 5. 打分：质量(ROE、增速) + 估值(PE低好) + 动量(60日涨幅)，等权合成 ----
    for c, newc in [('roe', 'z_roe'), ('growth', 'z_growth'), ('pe', 'z_pe'), ('mom60', 'z_mom')]:
        valid[newc] = (valid[c] - valid[c].mean()) / valid[c].std(ddof=0)
    valid['quality'] = (valid['z_roe'] + valid['z_growth']) / 2
    valid['value'] = -valid['z_pe']
    valid['momentum'] = valid['z_mom']
    valid['composite'] = (valid['quality'] + valid['value'] + valid['momentum']) / 3
    valid = valid.sort_values('composite', ascending=False).reset_index(drop=True)
    valid['rank'] = valid.index + 1

    cols = ['rank', 'code', 'name', 'mktcap', 'close', 'mom60', 'roe', 'growth', 'pe',
            'debt_ratio', 'quality', 'value', 'momentum', 'composite']
    all_scores = valid[cols].copy()
    for c in ['mom60', 'roe', 'growth', 'pe', 'debt_ratio', 'quality', 'value', 'momentum', 'composite']:
        all_scores[c] = all_scores[c].round(2)
    all_scores['mktcap'] = all_scores['mktcap'].round(0)   # 亿元，整数位即可
    all_scores = all_scores.rename(columns={'mktcap': '总市值(亿元)', 'close': '收盘价', 'mom60': '近60日涨跌幅(%)',
                                            'roe': 'ROE(%)', 'growth': '归母净利增速(%)', 'pe': 'PE(TTM)',
                                            'debt_ratio': '资产负债率(%)', 'quality': '质量分', 'value': '估值分',
                                            'momentum': '动量分', 'composite': '综合分', 'rank': '排名'})
    all_scores.to_csv(os.path.join(OUT, 'all_scores.csv'), index=False, encoding='utf-8-sig')
    top10 = all_scores.head(10)
    top10.to_csv(os.path.join(OUT, 'screen_result.csv'), index=False, encoding='utf-8-sig')
    log('输出', f"outputs/all_scores.csv ({len(all_scores)}只) + outputs/screen_result.csv (前10名)")
    return valid, all_scores, meta


# ================================================================
# 对照：data/cache 30只股票，同口径动量 + 与 iFinD 重叠股对比
# ================================================================
def _code6(code):
    """统一股票代码口径: iFinD 返回 '600519.SH'，缓存文件名是 '600519' —— 都取6位数字"""
    m = re.search(r'\d{6}', str(code))
    return m.group(0) if m else str(code)


def run_cache_comparison(ifind_valid, pool_codes=None):
    """缓存数据只有日线OHLCV，没有ROE/PE —— 所以同口径打分只做到动量维度。
    教学点: 接口数据时效性(截至今天) vs 缓存数据静态性(截至下载日)。"""
    rows = []
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, 'stock_*.csv'))):
        code = os.path.basename(f).split('_')[1]
        name = os.path.basename(f).split('_')[2].replace('.csv', '')
        try:
            raw = pd.read_csv(f)
            d, px = raw['date'], raw['close'].dropna()
            mom = (px.iloc[-1] / px.iloc[-61] - 1) * 100 if len(px) > 61 else np.nan  # 近60个交易日
            rows.append({'code': code, 'name': name, 'cache_close': px.iloc[-1],
                         'cache_last_date': d.iloc[-1], 'cache_mom60': mom})
        except Exception as e:
            log('对照', f'跳过 {name}: {e}')
    cache_df = pd.DataFrame(rows)
    if cache_df.empty:
        return None
    cache_df['code6'] = cache_df['code'].map(_code6)

    out = cache_df.copy()
    comp_info = {'cache_n': len(cache_df),
                 'cache_last_date': f"{cache_df['cache_last_date'].min()} ~ {cache_df['cache_last_date'].max()}",
                 'cache_in_pool': int(out['code6'].isin(pool_codes or []).sum()) if pool_codes else None}
    if ifind_valid is not None:
        iv = ifind_valid.copy()
        iv['code6'] = iv['code'].map(_code6)
        ifz = iv.set_index('code6')['mom60']
        ifc = iv.set_index('code6')['close']
        out['ifind_mom60'] = out['code6'].map(ifz)
        out['ifind_close'] = out['code6'].map(ifc)
        overlap = out.dropna(subset=['ifind_mom60', 'cache_mom60']).copy()
        for src in ['cache', 'ifind']:
            m, s = overlap[f'{src}_mom60'].mean(), overlap[f'{src}_mom60'].std(ddof=0)
            overlap[f'{src}_z'] = (overlap[f'{src}_mom60'] - m) / s
        out = out.merge(overlap[['code', 'cache_z', 'ifind_z']], on='code', how='left')
        out['mom_diff_pp'] = (out['ifind_mom60'] - out['cache_mom60']).round(2)   # 百分点差
        out['px_diff_pct'] = ((out['ifind_close'] / out['cache_close'] - 1) * 100).round(2)  # 收盘价差(%)

        from scipy.stats import spearmanr
        rho, p = spearmanr(overlap['cache_mom60'], overlap['ifind_mom60'])
        comp_info.update({'overlap_n': len(overlap),
                          'spearman_rho': round(float(rho), 3), 'spearman_p': round(float(p), 4),
                          'cache_mom_mean': round(float(overlap['cache_mom60'].mean()), 2),
                          'ifind_mom_mean': round(float(overlap['ifind_mom60'].mean()), 2)})
        log('对照', f"缓存30只中 {comp_info['cache_in_pool']} 只在候选池内; 与深挖名单重叠 {len(overlap)} 只; "
                    f"动量 Spearman rho={comp_info['spearman_rho']} (p={comp_info['spearman_p']})")
    out.to_csv(os.path.join(OUT, 'cache_vs_ifind.csv'), index=False, encoding='utf-8-sig')
    return out, comp_info


# ================================================================
# 图表
# ================================================================
def annotate_no_overlap(ax, fig, points, texts, marker_points=(), fontsize=9):
    """给散点加名字标签，并做简单的"防重叠"处理：
    1) 先按默认偏移(6,5)放置；
    2) 用渲染器量出标签真实包围盒，若与【已放置的标签】或【任何数据点圆斑】
       重叠，就按候选偏移序列换位置重画，直到不撞或试完为止。
    真实机构做图也绕不开这一步 —— 20个点挤在左下角是常态，
    matplotlib 自己不会帮你把标签让开，必须自己写碰撞检测。"""
    obstacles = []
    canvas = fig.canvas
    trans = ax.transData.transform
    from matplotlib.transforms import Bbox
    for x, y, s in marker_points:            # 数据点圆斑也算障碍物
        cx, cy = trans((x, y))
        r = (s ** 0.5) / 2 + 3
        obstacles.append(Bbox([[cx - r, cy - r], [cx + r, cy + r]]))
    placed = []
    offset_candidates = [(6, 5), (6, -13), (6, 17), (6, -25), (-32, 5), (-32, -13), (-32, 17), (-32, 23)]
    for (x, y), label in zip(points, texts):
        ann = None
        for dx, dy in offset_candidates:
            if ann is None:
                ann = ax.annotate(label, (x, y), textcoords='offset points',
                                  xytext=(dx, dy), fontsize=fontsize, color='#333333', zorder=5)
            else:
                ann.xyann = (dx, dy)
            canvas.draw()
            bbox = ann.get_window_extent(renderer=canvas.get_renderer())
            if not any(bbox.overlaps(b) for b in obstacles + placed):
                break
        # 兜底: 8个候选位都撞的话，就在当前位置继续向下小步挪，挪出一条干净的竖直通道
        for _ in range(14):
            bbox = ann.get_window_extent(renderer=canvas.get_renderer())
            if not any(bbox.overlaps(b) for b in obstacles + placed):
                break
            ann.xyann = (ann.xyann[0], ann.xyann[1] - 7)
            canvas.draw()
        placed.append(ann.get_window_extent(renderer=canvas.get_renderer()))
    return placed


def make_charts(valid, all_scores, meta, cache_cmp):
    top10 = all_scores.head(10)

    # 图1: ROE 对 PE 散点，标注前10名
    fig, ax = plt.subplots(figsize=(10, 7))
    is_top = all_scores['排名'] <= 10
    ax.scatter(all_scores.loc[~is_top, 'PE(TTM)'], all_scores.loc[~is_top, 'ROE(%)'],
               s=80, c='#bbbbbb', alpha=0.7, label='前10名以外', edgecolors='white')
    sc = ax.scatter(all_scores.loc[is_top, 'PE(TTM)'], all_scores.loc[is_top, 'ROE(%)'],
                    s=140, c=all_scores.loc[is_top, '综合分'], cmap='RdYlGn',
                    edgecolors='#cc2222', linewidths=1.5, label='前10名', zorder=3)
    # PE 取对数轴: 银行股 PE 挤在 6~9、成长股上百，线性轴下前者会缩成一团。
    # 注意: 必须先设好坐标轴变换，再做标签防重叠，否则量出的包围盒全是错的。
    ax.set_xscale('log')
    ax.set_xlim(left=max(4.0, float(valid['pe'].min()) * 0.85), right=float(valid['pe'].max()) * 1.25)
    ax.set_ylim(bottom=float(valid['roe'].min()) - 2.5, top=float(valid['roe'].max()) + 3.0)
    ax.axhline(valid['roe'].median(), ls='--', lw=0.8, c='#888888')
    ax.axvline(valid['pe'].median(), ls='--', lw=0.8, c='#888888')
    # "理想区"(高ROE且便宜)用浅绿色块表示，比文字注释更直观
    ax.fill_betweenx([valid['roe'].median(), ax.get_ylim()[1]], ax.get_xlim()[0],
                     float(valid['pe'].median()), color='#2ca02c', alpha=0.06, zorder=0)
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('市盈率 PE(TTM, 对数轴) —— 越靠左越便宜; 虚线=中位数; 左上浅绿区=高ROE且便宜(理想区)')
    all_pts = list(zip(all_scores['PE(TTM)'], all_scores['ROE(%)'], all_scores['排名'].le(10).map({True: 140, False: 80})))
    annotate_no_overlap(ax, fig,
                        list(zip(all_scores.head(10)['PE(TTM)'], all_scores.head(10)['ROE(%)'])),
                        list(all_scores.head(10)['name']),
                        marker_points=[(p[0], p[1], p[2]) for p in all_pts])
    ax.set_ylabel('净资产收益率 ROE(%) —— 越靠上越赚钱')
    ax.set_title(f"图1 ROE-PE 散点（候选池{meta['pool_size']}只，深挖{meta['valid_size']}只，红框=综合分前10）")
    fig.colorbar(sc, ax=ax, label='综合分')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '01_ROE对PE散点.png'), dpi=150)
    plt.close(fig)

    # 图2: 综合得分前10条形图
    fig, ax = plt.subplots(figsize=(9, 6))
    t = top10.sort_values('综合分')
    colors = plt.cm.RdYlGn((t['综合分'] - t['综合分'].min()) / max(t['综合分'].max() - t['综合分'].min(), 1e-9) * 0.8 + 0.1)
    ax.barh(t['name'] + ' (' + t['code'] + ')', t['综合分'], color=colors, edgecolor='#555555')
    for i, (score, (q, v, m)) in enumerate(zip(t['综合分'], zip(t['质量分'], t['估值分'], t['动量分']))):
        ax.text(score, i, f'  {score:.2f} (质{q:+.1f}/估{v:+.1f}/动{m:+.1f})', va='center', fontsize=9)
    ax.set_xlabel('综合分 = (质量分 + 估值分 + 动量分) / 3（三项均为等权z-score）')
    ax.set_title("图2 综合得分前10名股票池（括号内: 质量/估值/动量分）")
    ax.axvline(0, c='#888888', lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '02_综合得分前10条形图.png'), dpi=150)
    plt.close(fig)

    # 图3: 数据缺失情况 + 数据覆盖漏斗
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    miss = meta['missing']
    fields_cn = {'mktcap': '总市值', 'roe': 'ROE', 'growth': '净利增速', 'debt_ratio': '资产负债率',
                 'pe': 'PE(TTM)', 'close': '收盘价', 'mom60': '60日涨跌幅'}
    bars = ax1.bar([fields_cn[k] for k in miss], list(miss.values()),
                   color=['#d62728' if v > 0 else '#2ca02c' for v in miss.values()])
    ax1.bar_label(bars)
    ax1.set_ylabel('缺失条数')
    ax1.set_ylim(0, max(2, max(miss.values()) * 1.4))
    from matplotlib.ticker import MaxNLocator
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title(f"图3a 各字段缺失统计（深挖{meta['merged_size']}只，合计缺失 {sum(miss.values())} 处）")
    ax1.tick_params(axis='x', labelrotation=20)
    ax1.axhline(0, c='black', lw=0.5)
    if sum(miss.values()) == 0:
        ax1.text(0.5, 0.5, '本次接口返回无缺失\n（缺失检查机制仍必须保留，\n因为下次运行不一定如此幸运）',
                 transform=ax1.transAxes, ha='center', va='center', fontsize=11, color='#2ca02c')
    stages = ['智能选股\n候选池', '调用预算内\n深挖名单', '字段齐全\n可打分']
    vals = [meta['pool_size'], meta['detail_size'], meta['valid_size']]
    bars2 = ax2.bar(stages, vals, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.bar_label(bars2)
    ax2.set_ylabel('股票数')
    ax2.set_title('图3b 数据覆盖漏斗：每一层都有损耗')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '03_数据缺失情况.png'), dpi=150)
    plt.close(fig)
    log('图表', '3张PNG已写入 outputs/')


# ================================================================
# 降级方案：iFinD 不可用时，用 data/cache 完成动量单因子打分
# ================================================================
def run_fallback():
    print('=' * 60)
    print('[降级模式] raw/ 缺失或 FORCE_FALLBACK=1 —— 本次未使用 iFinD，')
    print('[降级模式] 改用 data/cache 的30只A股日线完成"动量单因子"打分。')
    print('[降级模式] 注意: 缓存无财务数据，质量分/估值分无法计算。')
    print('=' * 60)
    rows = []
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, 'stock_*.csv'))):
        code = os.path.basename(f).split('_')[1]
        name = os.path.basename(f).split('_')[2].replace('.csv', '')
        px = pd.read_csv(f)['close'].dropna()
        if len(px) > 61:
            rows.append({'code': code, 'name': name, 'close': px.iloc[-1],
                         'mom60': (px.iloc[-1] / px.iloc[-61] - 1) * 100})
    df = pd.DataFrame(rows)
    df['momentum'] = (df['mom60'] - df['mom60'].mean()) / df['mom60'].std(ddof=0)
    df['composite'] = df['momentum']
    df = df.sort_values('composite', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    df['排名'], df['股票代码'], df['股票简称'] = df['rank'], df['code'], df['name']
    df['收盘价'], df['近60日涨跌幅(%)'], df['动量分'], df['综合分'] = df['close'], df['mom60'].round(2), df['momentum'].round(2), df['composite'].round(2)
    cols = ['排名', '股票代码', '股票简称', '收盘价', '近60日涨跌幅(%)', '动量分', '综合分']
    df[cols].to_csv(os.path.join(OUT, 'all_scores.csv'), index=False, encoding='utf-8-sig')
    df[cols].head(10).to_csv(os.path.join(OUT, 'screen_result.csv'), index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(9, 6))
    t = df.head(10).sort_values('综合分')
    ax.barh(t['股票简称'] + ' (' + t['股票代码'] + ')', t['综合分'], color='#1f77b4')
    ax.set_title('图2(降级版) 动量单因子前10名（data/cache，无财务字段）')
    ax.set_xlabel('动量z-score')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '02_综合得分前10条形图.png'), dpi=150)
    plt.close(fig)
    log('输出', f"降级模式完成: 候选池{len(df)}只(缓存30只中可计算{len(df)}只), 前10名已写出")
    return df


# ================================================================
def main():
    result = run_ifind_pipeline()
    cache_cmp = None
    if result is None:
        df_fb = run_fallback()
        print('\n[完成] 降级模式输出见 outputs/，请在 README 顶部注明本次未使用 iFinD')
        sys.exit(0)

    valid, all_scores, meta = result

    print('\n===== 综合分前10名（真实运行结果）=====')
    show = all_scores.head(10)[['排名', 'code', 'name', 'ROE(%)', '归母净利增速(%)', 'PE(TTM)',
                                '近60日涨跌幅(%)', '质量分', '估值分', '动量分', '综合分']]
    print(show.to_string(index=False))

    # 对照实验（辅助环节，失败不阻断主流程）
    try:
        cache_cmp = run_cache_comparison(valid, pool_codes=meta.get('pool_codes'))
        if cache_cmp is not None:
            cmp_df, comp_info = cache_cmp
            ov = cmp_df.dropna(subset=['ifind_mom60', 'cache_mom60'])
            if len(ov):
                print('\n===== iFinD vs data/cache 重叠股同口径对照（动量/收盘价）=====')
                print(ov[['code6', 'name', 'cache_last_date', 'cache_close', 'ifind_close', 'px_diff_pct',
                          'cache_mom60', 'ifind_mom60', 'mom_diff_pp']].round(2).to_string(index=False))
    except Exception as e:
        log('对照', f'对照环节失败(不影响主流程): {e}')

    # 图表（核心产出；出错要把完整堆栈打出来，但按教学规范不非0退出）
    try:
        make_charts(valid, all_scores, meta, cache_cmp)
    except Exception:
        import traceback
        log('图表', '出图失败，完整堆栈如下（请排查后重跑）:')
        traceback.print_exc()

    print('\n===== 运行摘要（供 README 引用）=====')
    meta_print = {k: v for k, v in meta.items() if k != 'pool_codes'}
    print(json.dumps(meta_print, ensure_ascii=False, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()
