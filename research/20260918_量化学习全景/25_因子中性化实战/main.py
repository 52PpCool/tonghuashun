# -*- coding: utf-8 -*-
"""
25_因子中性化实战 —— 把"沾了别的因子光"的收益剥干净
================================================================

量化研究员每天都在做的活：你挖了一个"动量因子"，IC 看起来不错，
但它可能只是在替"低波动效应"或某个行业抱团打工。本课用同一个月度截面，
对动量因子做三种处理并对比 IC：

  a) 原始动量（什么都不剥）；
  b) 低波动正交化：每月末在横截面上回归 动量 ~ 低波动，取残差——
     回答"动量里有多少只是低波动效应"；
  c) 行业中性化：每月末对行业哑变量回归取残差（等价于行业内去均值）——
     回答"动量是不是只在某些行业里灵"。

再演示两件进阶的事：
  - 正交化的顺序敏感性：先剥A再剥B ≠ 先剥B再剥A，顺序反映研究假设；
  - 中性化前后因子相关性热力图 + 多空组合行业暴露对比
    （2024-09 大反弹月），直观展示"中性化 = 卸掉行业赌注"。

数据：data/cache/ 30只真实A股日K（前复权）。若缓存缺失则用 seed=42
生成合成数据兜底（会在输出中显著标注）。随机种子 42，运行 < 2 分钟。

严谨性声明：只有 30 只股票的窄截面（幸存者池：都是活到今天的大公司），
没有真实市值/估值数据（"规模"只是价格水平的粗糙代理）。
本课重点是"中性化方法论"本身，而不是数字的绝对大小。
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示环境出图
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- 全局配置 ----------------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, '..', 'data', 'cache')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
START = '2018-01-01'          # 月末截面从2018年起
T_START = pd.Timestamp('2018-01-01')

# ---------------- 行业归属（硬编码） ----------------
# 说明：这是基于公开行业分类（申万/证监会门类）整理的【30只股票的粗略近似】。
# 为了让"行业内z-score/哑变量回归"每组至少有2只股票，合并了部分小行业
# （例如"水电+光伏制造"合并为"电力与新能源"、"石油石化+有色"合并为"能源资源"）。
# 真实研究中应使用官方一级行业分类 + 历史时点的行业归属（见 README"常见的坑"）。
INDUSTRY_MAP = {
    # 白酒：贵州茅台 / 五粮液 / 山西汾酒
    '600519': '白酒', '000858': '白酒', '600809': '白酒',
    # 大消费：伊利股份 / 海天味业 / 中国中免（食品饮料+免税零售合并）
    '600887': '消费', '603288': '消费', '601888': '消费',
    # 银行：平安/招商/兴业/工商/建设/中国
    '000001': '银行', '600036': '银行', '601166': '银行',
    '601398': '银行', '601939': '银行', '601988': '银行',
    # 非银金融：中国平安(保险) / 中信证券(券商) / 东方财富(互联网金融)
    '601318': '非银金融', '600030': '非银金融', '300059': '非银金融',
    # 家电：美的 / 格力
    '000333': '家电', '000651': '家电',
    # 汽车：比亚迪 / 上汽
    '002594': '汽车', '600104': '汽车',
    # 医药：恒瑞 / 迈瑞
    '600276': '医药', '300760': '医药',
    # 能源资源（上游大宗合并）：中国石化 / 中国石油 / 紫金矿业
    '600028': '能源资源', '601857': '能源资源', '601899': '能源资源',
    # 通信电子：中兴通讯 / 海康威视
    '000063': '通信电子', '002415': '通信电子',
    # 地产建材（地产链合并）：万科A / 海螺水泥
    '000002': '地产建材', '600585': '地产建材',
    # 电力与新能源：长江电力(水电) / 隆基绿能(光伏制造)
    '600900': '电力与新能源', '601012': '电力与新能源',
}

IS_SYNTHETIC = False  # 是否使用了合成数据兜底


# =====================================================================
# 一、数据加载：os.listdir + 前缀 'stock_' 遍历，构成收盘价面板
# =====================================================================
def make_synthetic_panel():
    """缓存缺失时的合成数据兜底（seed=42，显著标注）。带行业结构+动量自相关。"""
    print('!! 警告：未找到 data/cache/stock_*.csv，改用【合成数据】（seed=42）!!')
    dates = pd.bdate_range('2018-01-02', '2026-09-17')
    n = len(dates)
    mkt = rng.normal(0.0002, 0.010, n)                      # 市场因子
    ind_list = sorted(set(INDUSTRY_MAP.values()))
    ind_ret = {k: rng.normal(0.0002, 0.005, n) for k in ind_list}  # 行业因子
    cols = {}
    for code in INDUSTRY_MAP:                                # 沿用真实代码→行业映射
        beta = rng.uniform(0.6, 1.3)
        ret = (0.0002 + beta * mkt + 0.6 * ind_ret[INDUSTRY_MAP[code]]
               + rng.normal(0, 0.016, n))
        # 注入轻微动量自相关（一阶正自相关），让"动量因子"在合成世界里有点信号
        ret = ret + 0.10 * np.concatenate([[0.0], ret[:-1]])
        cols[code] = 15.0 * np.exp(np.cumsum(np.log1p(ret)))
    close = pd.DataFrame(cols, index=dates)
    names = {c: '合成_' + c for c in close.columns}
    return close, names


def load_panel():
    """返回 (日收盘价面板DataFrame[日期×代码], 代码→名称dict)。"""
    global IS_SYNTHETIC
    try:
        files = sorted(f for f in os.listdir(CACHE_DIR)
                       if f.startswith('stock_') and f.endswith('.csv'))
    except OSError:
        files = []
    closes, names = {}, {}
    for f in files:
        code = f.split('_')[1]
        name = f[:-4].split('_', 2)[-1]        # stock_000001_平安银行.csv → 平安银行
        try:
            df = pd.read_csv(os.path.join(CACHE_DIR, f), encoding='utf-8-sig')
            s = pd.Series(df['close'].values.astype(float),
                          index=pd.to_datetime(df['date']))
            s = s[~s.index.duplicated(keep='last')].sort_index()
            closes[code], names[code] = s, name
        except Exception as e:                 # 单个文件坏了不拖垮整体
            print(f'  [跳过] {f}: {e}')
    close = pd.DataFrame(closes)
    if close.shape[1] < 10:                    # 缓存缺失/过少 → 合成数据
        close, names = make_synthetic_panel()
        IS_SYNTHETIC = True
    else:
        print(f'已加载真实缓存：{close.shape[1]} 只股票，'
              f'{close.index.min():%Y-%m-%d} ~ {close.index.max():%Y-%m-%d}')
    # 池子里若有未映射行业的代码，归入"其他"（理论上不会发生）
    for c in close.columns:
        if c not in INDUSTRY_MAP:
            print(f'  [提示] {c} 无行业映射，归入"其他"')
            INDUSTRY_MAP[c] = '其他'
    close = close.loc[close.index >= T_START].sort_index()
    close = close.ffill()                      # 停牌日沿用前收盘（收益记0）
    return close, names


# =====================================================================
# 二、中性化工具函数（本课灵魂）
# =====================================================================
def zscore(s: pd.Series) -> pd.Series:
    """横截面 z-score：减均值、除以标准差（ddof=0）。退化截面返回全NaN。"""
    mu, sd = s.mean(), s.std(ddof=0)
    if (not np.isfinite(sd)) or sd < 1e-12:
        return s * np.nan
    return (s - mu) / sd


def ols_resid(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """横截面 OLS：y ~ 常数项 + X（逐列），返回残差 Series。

    公式（README 核心概念）：beta = (X'X)^(-1) X'y，resid = y - X @ beta。
    残差的含义：y 中【X 解释不掉】的部分——把 X 里沾的光剥干净。
    用 np.linalg.lstsq 而非矩阵求逆，对共线/退化更稳。
    支持链式调用：y 可以是上一步的残差（索引是其样本子集），
    因此先把 X 对齐到 y 的索引再取公共非NaN样本。
    """
    X = X.reindex(y.index)                     # 关键：对齐到y（可能是残差的子集）
    mask = y.notna()
    for c in X.columns:
        mask &= X[c].notna()
    yy = y[mask].to_numpy(dtype=float)
    Xd = np.column_stack([np.ones(int(mask.sum()))]
                         + [X[c][mask].to_numpy(dtype=float) for c in X.columns])
    beta, *_ = np.linalg.lstsq(Xd, yy, rcond=None)
    return pd.Series(yy - Xd @ beta, index=y.index[mask])


def industry_neutral_residual(factor: pd.Series,
                              industry: pd.Series) -> pd.Series:
    """行业中性化·做法一：对【行业哑变量+常数】回归取残差。

    X = [1, D_2, ..., D_K]（第1个行业作基准组不进X，避免共线）。
    由 Frisch–Waugh–Lovell 定理，残差 = 因子减去各自行业均值
    = "行业内相对强弱"，与做法二（组内z-score）的秩几乎一致。
    """
    mask = factor.notna()
    inds = industry[mask]
    codes = sorted(inds.unique())
    if len(codes) < 2:                         # 当月只有一个行业 → 无从中性化
        return factor * np.nan
    X = np.column_stack(
        [np.ones(int(mask.sum()))]
        + [(inds == c).to_numpy(dtype=float) for c in codes[1:]])
    yy = factor[mask].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return pd.Series(yy - X @ beta, index=factor.index[mask])


def industry_zscore(factor: pd.Series, industry: pd.Series) -> pd.Series:
    """行业中性化·做法二：行业内 z-score（组内减均值除以组内标准差）。

    与哑变量回归的差别：z-score 除以了【组内】标准差（相当于再按行业
    调整波动），哑变量残差只做组内去均值。秩相关通常非常接近。
    """
    def _z(g: pd.Series) -> pd.Series:
        sd = g.std(ddof=0)
        if len(g) < 2 or (not np.isfinite(sd)) or sd < 1e-12:
            return g * np.nan                  # 组内只有1只 → 无法相对化
        return (g - g.mean()) / sd
    return factor.groupby(industry).transform(_z)


def rank_ic(factor: pd.Series, fwd: pd.Series) -> float:
    """单月 RankIC = 因子与下月收益的横截面 Spearman 秩相关。"""
    df = pd.concat([factor.rename('f'), fwd.rename('r')], axis=1).dropna()
    if len(df) < 8:
        return np.nan
    rho, _ = stats.spearmanr(df['f'], df['r'])
    return float(rho) if np.isfinite(rho) else np.nan


def ls_ret(factor: pd.Series, fwd: pd.Series):
    """单月多空收益：因子最高的前1/3做多、最低的1/3做空（等权）。

    返回 (多空收益, 多头名单, 空头名单)。中性化后因子重新排秩，
    多空名单会变——交易成本在【两边】都要重新付（README 常见的坑）。
    """
    df = pd.concat([factor.rename('f'), fwd.rename('r')], axis=1).dropna()
    n = len(df)
    if n < 12:
        return np.nan, None, None
    k = max(3, n // 3)
    df = df.sort_values('f', ascending=False)
    longs, shorts = list(df.index[:k]), list(df.index[-k:])
    return float(df['r'].iloc[:k].mean() - df['r'].iloc[-k:].mean()), longs, shorts


def ind_net_exposure(longs, shorts) -> pd.Series:
    """多空组合的净行业暴露 = (多头中该行业只数 - 空头中该行业只数)/多头只数。"""
    net = {ind: 0 for ind in sorted(set(INDUSTRY_MAP.values()))}
    for c in longs:
        net[INDUSTRY_MAP[c]] += 1
    for c in shorts:
        net[INDUSTRY_MAP[c]] -= 1
    return pd.Series(net, dtype=float) / len(longs)


# =====================================================================
# 三、主流程：月度循环 → 因子 → 中性化 → IC/多空
# =====================================================================
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    close, names = load_panel()
    ind = pd.Series(INDUSTRY_MAP).reindex(close.columns)

    # ---- 月末截面：每月最后一个交易日；剔除不足15个交易日的"残月" ----
    # （实测会剔除 2026-09——数据只到9月中的未收完月份，以及 2026-02
    #   ——春节假期当月交易日过少的残月。残月当作月末会扭曲月度收益。）
    per = close.index.to_period('M')
    month_last = close.index.to_series().groupby(per).max()
    month_cnt = close.index.to_series().groupby(per).count()
    month_last = month_last[month_cnt >= 15]
    M = close.loc[month_last.values]           # 月末收盘价面板（日期×代码）

    # ---- 因子面板（月末截面，2018起） ----
    # 1) 动量(12-1月)：t-12月 到 t-1月 的累计收益（剔除最近1个月，
    #    A股最近1月常含反转效应；用月末价：M[t-1]/M[t-12]-1）
    mom_panel = M.shift(1) / M.shift(12) - 1.0
    # 2) 下月收益（IC 的预测目标，严格用未来月末价，无前视）
    fwd1 = M.shift(-1) / M - 1.0
    R = close.pct_change()                     # 日收益（供低波动因子）

    versions = ['原始动量', '低波动正交后', '行业中性(哑变量回归)', '行业中性(组内zscore)']
    ic_rec, ls_rec = [], []
    corr_before, corr_mid, corr_after = [], [], []   # 中性化前/低波正交/行业中性的月度因子相关矩阵
    order_rec = []                             # 正交化顺序敏感性记录
    mom_lowcorr = []                           # 动量与低波动的月度相关
    snap = {}                                  # 2024-09 反弹月的持仓快照

    idx = M.index
    for i in range(len(M)):
        t = idx[i]
        if i < 12:                             # 动量需要12个月历史
            continue
        fwd = fwd1.iloc[i]
        if fwd.notna().sum() < 8:              # 最后一个月没有"下月"可预测
            continue
        mom = mom_panel.iloc[i].dropna()
        if len(mom) < 20:                      # 截面太窄（历史不足）跳过
            continue
        pos = close.index.get_loc(t)

        # ---- 当月截面因子（全部做横截面z-score，便于回归比较） ----
        C = close.iloc[pos]
        rev = -(C / close.iloc[pos - 5] - 1.0)             # 短期反转(-5日收益)
        w = R.iloc[pos - 59:pos + 1]                        # 低波动(-60日波动率)
        sd60 = w.std()
        lowvol = -sd60.where(w.count() >= 45)               # 样本<45天视为缺失
        size = np.log(C.copy())                 # 规模代理=log(价格水平)(粗糙!)
        # 注1：价格高低与市值正相关但股本各异，只是"真代理、不精确"。
        # 注2：取对数缓解价格极端偏斜（茅台1400元 vs 银行股十几元），
        #      否则Pearson回归会被高价股一两家主导（README 常见的坑）。
        F = pd.DataFrame({'动量原始': mom, '反转': rev,
                          '低波动': lowvol, '规模代理': size})
        Fz = F.apply(zscore)

        # ----【本课灵魂】对动量的三种处理 ----
        f_raw = Fz['动量原始']
        f_ll = zscore(ols_resid(Fz['动量原始'], Fz[['低波动']]))   # b) 剥低波动
        f_in = zscore(industry_neutral_residual(Fz['动量原始'], ind))  # c) 剥行业
        f_iz = industry_zscore(Fz['动量原始'], ind)               # c') 组内zscore对照
        # 中性化之后【重新标准化】——不改秩，但让不同月可比（README 坑2）

        # ---- RankIC 与多空 ----
        ic_row = {'date': t}
        ls_row = {'date': t}
        for name, f in zip(versions, [f_raw, f_ll, f_in, f_iz]):
            ic_row[name] = rank_ic(f, fwd)
            r, lg, sh = ls_ret(f, fwd)
            ls_row[name] = r
        ic_rec.append(ic_row)
        ls_rec.append(ls_row)

        # ---- 2024-09 大反弹月的持仓快照（形成于2024-08月末） ----
        if t.year == 2024 and t.month == 8:
            _, lg_raw, sh_raw = ls_ret(f_raw, fwd)
            _, lg_in, sh_in = ls_ret(f_in, fwd)
            snap = {'date': t, 'fwd': fwd,
                    'raw': (lg_raw, sh_raw), 'ind': (lg_in, sh_in)}

        # ---- 相关性热力图素材：三种动量版本的月度因子相关矩阵 ----
        Fbase = F[['反转', '低波动', '规模代理']]
        for store, col, vec in [(corr_before, '动量原始', Fz['动量原始']),
                                (corr_mid, '低波动正交动量', f_ll),
                                (corr_after, '行业中性动量', f_in)]:
            Fm = Fbase.copy()
            Fm.insert(0, col, vec)             # 按代码索引自动对齐，缺失为NaN
            store.append(Fm.corr(method='spearman'))
        cm = pd.concat([Fz['动量原始'], Fz['低波动']],
                       axis=1, keys=['m', 'v']).dropna()
        if len(cm) >= 10:
            mom_lowcorr.append(stats.spearmanr(cm['m'], cm['v'])[0])

        # ---- 正交化顺序敏感性：先剥A再剥B vs 先剥B再剥A vs 联合回归 ----
        # A=低波动，B=规模代理（本面板月均|corr|最强的一对风格因子，约0.4）。
        # r_ab 最后剥B → corr(r_ab, B)=0（B被剥得最干净），但对A残留暴露；
        # r_ba 反之。残差化【不可交换】，顺序反映研究假设（见README）。
        # r_jt：把A、B放进同一个回归 → 同时正交于A和B，不依赖顺序，
        # 这是正规风险模型（Barra纯因子组合）的做法，作为对照。
        rA = ols_resid(Fz['动量原始'], Fz[['低波动']])
        r_ab = ols_resid(rA, Fz[['规模代理']])          # 先低波 → 后规模
        rB = ols_resid(Fz['动量原始'], Fz[['规模代理']])
        r_ba = ols_resid(rB, Fz[['低波动']])            # 先规模 → 后低波
        r_jt = ols_resid(Fz['动量原始'],
                         Fz[['低波动', '规模代理']])     # 联合正交化
        d = pd.concat([r_ab.rename('ab'), r_ba.rename('ba'),
                       r_jt.rename('jt'),
                       Fz['低波动'].rename('A'), Fz['规模代理'].rename('B')],
                      axis=1).dropna()
        if len(d) >= 10:
            cc = lambda p, q: float(np.corrcoef(d[p], d[q])[0, 1])
            order_rec.append({
                'date': t, 'corr_AB': cc('A', 'B'),
                'corr_AB_sp': float(stats.spearmanr(d['A'], d['B'])[0]),
                'corr_ab_ba': cc('ab', 'ba'),
                'corr_jt_ab': cc('jt', 'ab'), 'corr_jt_ba': cc('jt', 'ba'),
                'ab_A': cc('ab', 'A'), 'ab_B': cc('ab', 'B'),
                'ba_A': cc('ba', 'A'), 'ba_B': cc('ba', 'B'),
                'jt_A': cc('jt', 'A'), 'jt_B': cc('jt', 'B'),
                'ic_ab': rank_ic(zscore(r_ab), fwd),
                'ic_ba': rank_ic(zscore(r_ba), fwd),
                'ic_jt': rank_ic(zscore(r_jt), fwd)})

    ic_df = pd.DataFrame(ic_rec).set_index('date')
    ls_df = pd.DataFrame(ls_rec).set_index('date')
    n_m = len(ic_df)
    print(f'\n有效月度截面：{ic_df.index.min():%Y-%m} ~ {ic_df.index.max():%Y-%m}，'
          f'共 {n_m} 个月（动量需12个月历史，故自2019-01起）')

    # ---- 汇总对比表 → neutralization_compare.csv ----
    def agg(ic: pd.Series, ls: pd.Series):
        ic, ls = ic.dropna(), ls.dropna()
        icir = ic.mean() / ic.std(ddof=1)
        return {'RankIC均值': ic.mean(), 'IC标准差': ic.std(ddof=1), 'ICIR': icir,
                'IC_t值': icir * np.sqrt(len(ic)), '月份数': len(ic),
                '多空年化收益': ls.mean() * 12, '多空年化波动': ls.std(ddof=1) * np.sqrt(12),
                '多空夏普': (ls.mean() / ls.std(ddof=1) * np.sqrt(12)) if ls.std(ddof=1) > 0 else np.nan}

    note = {'原始动量': '什么都不剥，IC可能被风格/行业暴露垫高',
            '低波动正交后': '每月回归 动量~低波动 取残差，剥掉风格光',
            '行业中性(哑变量回归)': '对行业哑变量回归取残差 = 行业内相对强弱',
            '行业中性(组内zscore)': '做法二对照：行业内z-score，秩几乎同哑变量法'}
    cmp_rows = []
    for v in versions:
        row = {'处理方式': v, **agg(ic_df[v], ls_df[v]), '说明': note[v]}
        cmp_rows.append(row)
    compare = pd.DataFrame(cmp_rows).set_index('处理方式').round(4)
    compare.to_csv(os.path.join(OUT_DIR, 'neutralization_compare.csv'),
                   encoding='utf-8-sig')
    print('\n===== 三种处理的动量因子对比（月度截面） =====')
    print(compare[['RankIC均值', 'ICIR', 'IC_t值', '多空年化收益',
                   '多空夏普', '月份数']].to_string())

    # ---- 动量里有多少只是"低波动效应"？ ----
    # 注：正交化残差与低波动的Pearson相关按构造恒为0，故只需报告
    # 原始动量与低波动的月度Spearman相关的分布（均值/分位数/平均R²）。
    mlc = pd.Series(mom_lowcorr, dtype=float)
    c_raw = mlc.mean()
    print('\n动量 vs 低波动的月度Spearman相关：'
          f'均值 = {c_raw:+.3f}，10%分位 = {mlc.quantile(0.1):+.3f}，'
          f'中位数 = {mlc.quantile(0.5):+.3f}，90%分位 = {mlc.quantile(0.9):+.3f}')
    print(f'平均R²（相关系数平方）≈ {(mlc ** 2).mean():.3f} —— '
          '原始动量的截面排序里有相当一部分与"低波动"风格共线，')
    print('正交化（做法b）按构造把这部分完全剥掉（残差与低波动的Pearson相关=0）。')

    # ---- 行业中性化到底剥掉了什么：多空组合的行业暴露（2024-09反弹月） ----
    print('\n===== 2024-09 大反弹月的行业暴露对比（持仓形成于 %s） ====='
          % (snap['date'].strftime('%Y-%m-%d') if snap else 'N/A'))
    if snap:
        exp_raw = ind_net_exposure(*snap['raw'])
        exp_ind = ind_net_exposure(*snap['ind'])
        ls_raw_m = ls_df.loc[snap['date'], '原始动量']
        ls_ind_m = ls_df.loc[snap['date'], '行业中性(哑变量回归)']
        print(f'该月（2024-09）原始动量多空收益 = {ls_raw_m:+.2%}，'
              f'行业中性动量多空收益 = {ls_ind_m:+.2%}')
        print('原始动量多头:', '、'.join(names[c] for c in snap['raw'][0]))
        print('原始动量空头:', '、'.join(names[c] for c in snap['raw'][1]))
        print('中性动量多头:', '、'.join(names[c] for c in snap['ind'][0]))
        print('中性动量空头:', '、'.join(names[c] for c in snap['ind'][1]))
        comp = pd.DataFrame({'原始动量多空净暴露': exp_raw,
                             '行业中性动量多空净暴露': exp_ind}).round(2)
        print(comp.to_string())

    # ---- 正交化顺序敏感性汇总 ----
    od = pd.DataFrame(order_rec).set_index('date')
    if len(od):
        print('\n===== 正交化顺序敏感性（A=低波动，B=规模代理log价格） =====')
        print(f'A与B的条件相关：月均Pearson |corr| = {od["corr_AB"].abs().mean():.3f}，'
              f'月均Spearman |corr| = {od["corr_AB_sp"].abs().mean():.3f}')
        print(f'corr(r_ab, r_ba) 月均 = {od["corr_ab_ba"].mean():.3f}'
              f'（最低月 {od["corr_ab_ba"].min():.3f}）；若可交换应恒为1')
        print(f'corr(联合残差r_jt, r_ab) 月均 = {od["corr_jt_ab"].mean():.3f}，'
              f'corr(r_jt, r_ba) 月均 = {od["corr_jt_ba"].mean():.3f}')
        print(f'暴露签名（月均|corr|） r_ab: 与A={od["ab_A"].abs().mean():.3f}, '
              f'与B={od["ab_B"].abs().mean():.3f}   ← 后剥的B严格归零')
        print(f'                     r_ba: 与A={od["ba_A"].abs().mean():.3f}, '
              f'与B={od["ba_B"].abs().mean():.3f}   ← 后剥的A严格归零')
        print(f'                     r_jt: 与A={od["jt_A"].abs().mean():.3f}, '
              f'与B={od["jt_B"].abs().mean():.3f}   ← 联合回归同时对A、B归零')
        print(f'RankIC均值 r_ab={od["ic_ab"].mean():+.4f}，'
              f'r_ba={od["ic_ba"].mean():+.4f}，r_joint={od["ic_jt"].mean():+.4f}')

    # =================================================================
    # 四、出图
    # =================================================================
    C_RAW, C_LL, C_IN, C_IZ = '#d62728', '#1f77b4', '#2ca02c', '#9467bd'
    VB = versions[:3]

    # ---- 图01：三种处理 IC 对比 ----
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    metrics = [('RankIC均值', '月度RankIC均值', '{:+.3f}', 1),
               ('ICIR', 'ICIR（IC均值/IC标准差）', '{:+.2f}', 1),
               ('多空年化收益', '多空年化收益（%，多top1/3-空bottom1/3，×12）', '{:+.1f}', 100)]
    colors = [C_RAW, C_LL, C_IN]
    for ax, (col, title, fmt, scale) in zip(axes, metrics):
        vals = [compare.loc[v, col] * scale for v in VB]
        bars = ax.bar(range(3), vals, color=colors, width=0.62)
        off = max(abs(min(vals)), abs(max(vals)), 1e-9) * 0.07
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + off * np.sign(v),
                    fmt.format(v), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['原始\n动量', '低波动\n正交后', '行业中性\n(哑变量)'], fontsize=10)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylim(min(0, min(vals) * 1.25), max(max(vals) * 1.25, 1e-9))
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('动量因子的三种处理：中性化前后 IC 对比（30只A股，月度截面）',
                 fontsize=13)
    fig.savefig(os.path.join(OUT_DIR, '01_三种处理IC对比.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ---- 图02：中性化前后因子相关性热力图（月度相关矩阵的简单平均） ----
    # 剥谁、谁与动量的相关就归零：原始(-0.13 vs 低波动) → 低波正交(≈0)
    # → 行业中性(顺带也削掉一部分，因为行业成分承载了部分风格暴露)。
    CB = sum(corr_before) / len(corr_before)
    CM = sum(corr_mid) / len(corr_mid)
    CA = sum(corr_after) / len(corr_after)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0), constrained_layout=True)
    for ax, mat, title in [(axes[0], CB, '中性化前：动量原始'),
                           (axes[1], CM, '低波动正交后（Pearson相关严格=0）'),
                           (axes[2], CA, '行业中性(哑变量回归)后')]:
        im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=30,
                      ha='right', fontsize=9)
        ax.set_yticks(range(len(mat.index)), mat.index, fontsize=9)
        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                ax.text(b, a, f'{mat.values[a, b]:+.2f}',
                        ha='center', va='center', fontsize=9,
                        color='w' if abs(mat.values[a, b]) > 0.55 else 'k')
        ax.set_title(f'月均横截面Spearman相关 — {title}', fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle('中性化前后的因子相关性："剥谁、谁与动量的相关就归零"'
                 '（行业维度的去相关见图04）', fontsize=13)
    fig.savefig(os.path.join(OUT_DIR, '02_中性化前后因子相关性热力图.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图03：正交化顺序敏感性 ----
    if len(od):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
        ax = axes[0]
        x = np.arange(3)
        wdt = 0.35
        vA = [od['ab_A'].abs().mean(), od['ba_A'].abs().mean(), od['jt_A'].abs().mean()]
        vB = [od['ab_B'].abs().mean(), od['ba_B'].abs().mean(), od['jt_B'].abs().mean()]
        b1 = ax.bar(x - wdt / 2, vA, wdt, label='与低波动(A)的月均|corr|', color=C_LL)
        b2 = ax.bar(x + wdt / 2, vB, wdt, label='与规模代理(B)的月均|corr|', color='#ff7f0e')
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                        f'{b.get_height():.3f}', ha='center', fontsize=9)
        ax.set_xticks(x, ['r_ab\n先剥A(低波动)→后剥B(规模)',
                          'r_ba\n先剥B(规模)→后剥A(低波动)',
                          'r_joint\nA、B联合回归'], fontsize=9)
        ax.set_ylabel('残差与条件因子的月均|相关|')
        ax.set_ylim(0, max(0.1, max(vA + vB) * 1.35))
        ax.legend(fontsize=9)
        ax.set_title('暴露签名：最后被正交的因子被严格归零\n（先剥的因子残留暴露"回渗"）', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax = axes[1]
        ax.plot(od.index, od['corr_ab_ba'], color='#555', lw=1.2,
                label='corr(r_ab, r_ba)')
        ax.plot(od.index, od['corr_jt_ab'], color=C_IN, lw=1.0, ls='--',
                label='corr(r_joint, r_ab)')
        ax.axhline(od['corr_ab_ba'].mean(), color=C_RAW, ls='--', lw=1.2,
                   label=f"corr(r_ab,r_ba)月均 = {od['corr_ab_ba'].mean():.3f}")
        ax.axhline(1.0, color='k', ls=':', lw=1, label='若顺序可交换 = 1')
        ax.set_title('两种顺序残差的逐月相关性：残差化不可交换\n'
                     '（差异是二阶小量，条件因子越相关、样本越小差异越大）', fontsize=11)
        ax.set_ylabel('逐月残差相关系数')
        ax.set_ylim(0.8, 1.005)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)
        fig.suptitle('正交化顺序敏感性：顺序反映研究假设（A=低波动，B=规模代理；'
                     '联合回归不依赖顺序）', fontsize=13)
        fig.savefig(os.path.join(OUT_DIR, '03_正交化顺序敏感性.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    # ---- 图04：多空组合行业暴露对比（2024-09反弹月） ----
    if snap:
        exp_raw = ind_net_exposure(*snap['raw'])
        exp_ind = ind_net_exposure(*snap['ind'])
        order = (exp_raw.abs() + exp_ind.abs()).sort_values(ascending=False).index
        x = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(12.5, 5.0), constrained_layout=True)
        ax.bar(x - 0.2, exp_raw[order], 0.4, color=C_RAW,
               label='原始动量多空（净行业暴露）')
        ax.bar(x + 0.2, exp_ind[order], 0.4, color=C_IN,
               label='行业中性动量多空（净行业暴露）')
        ax.set_xticks(x, order, rotation=30, ha='right', fontsize=9)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylabel('净行业暴露 = (多头只数 − 空头只数) / 多空各10只')
        ax.set_title('2024-09 大反弹月的多空组合行业分布（形成于%s收盘）：'
                     '中性化 = 卸掉行业赌注' % snap['date'].strftime('%Y-%m-%d'),
                     fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.text(0.99, 0.02,
                f'该月多空收益：原始 {ls_df.loc[snap["date"], "原始动量"]:+.1%} → '
                f'行业中性 {ls_df.loc[snap["date"], "行业中性(哑变量回归)"]:+.1%}',
                transform=ax.transAxes, ha='right', fontsize=10, color='#333',
                bbox=dict(fc='#f5f5f5', ec='#ccc'))
        fig.savefig(os.path.join(OUT_DIR, '04_多空组合行业暴露对比.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    # ---- 图05：中性动量累计IC ----
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    for v, c in zip(versions[:3], colors):
        cum = ic_df[v].fillna(0).cumsum()
        ax.plot(cum.index, cum.values, color=c, lw=1.6,
                label=f'{v}（累计 {cum.iloc[-1]:+.2f}）')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('三种处理的动量因子累计 RankIC（逐月累加）', fontsize=12)
    ax.set_ylabel('累计 RankIC')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, '05_中性动量累计IC.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ---- industry_map.csv ----
    imap = pd.DataFrame(
        [{'代码': c, '名称': names.get(c, ''), '行业分组(近似)': INDUSTRY_MAP[c]}
         for c in sorted(INDUSTRY_MAP)]).set_index('代码')
    imap.to_csv(os.path.join(OUT_DIR, 'industry_map.csv'), encoding='utf-8-sig')

    # ---- 收尾打印 ----
    dt = time.time() - t0
    print('\n===== 运行完成 =====')
    print(f'数据来源：{"合成数据(seed=42兜底)" if IS_SYNTHETIC else "data/cache 真实A股日K(前复权)"}')
    print(f'输出目录：{OUT_DIR}')
    for f in sorted(os.listdir(OUT_DIR)):
        print(f'  - {f}')
    print(f'耗时 {dt:.1f} 秒')
    print('严谨性声明：30只窄截面、幸存者池、无真实市值/估值数据（规模=价格粗糙代理）；'
          '本课重点是中性化方法论而非数字大小。')


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # 课程规范：绝不允许以非0退出。打印堆栈便于排查（产出可能不完整）。
        print('=' * 70)
        print('[兜底] 出现未捕获异常，按课程规范以退出码0结束，但产出可能不完整：')
        traceback.print_exc()
        print('=' * 70)
        sys.exit(0)
