# -*- coding: utf-8 -*-
"""
14_配对交易与协整 —— 市场中性策略入门：不猜涨跌，赌价差回归。

本课主线（对应 README 教学文档）：
  步骤1  相关 ≠ 协整：5只银行股两两相关系数 vs Engle-Granger 协整检验 p 值矩阵
  步骤2  选出协整 p 值最小的配对，OLS 回归对冲比率 β，画价差
  步骤3  价差平稳性（ADF）+ OU 过程拟合 → 均值回归半衰期
  步骤4  z-score 策略回测（滚动60日标准化，±2 开仓 / |z|<0.5 平仓，单腿单边成本万3）
  步骤5  绩效：年化/夏普/最大回撤/交易次数/平均持仓天数；样本内(2018-2023) vs 样本外(2024起)
  步骤6  反例教学：构造"高相关但协整不成立"的假想配对，同一套策略跑出亏损/被套

运行：cd 14_配对交易与协整 && ../../.venv/bin/python main.py
随机种子 42；数据缺失时自动降级为合成数据并显著标注，脚本绝不以非 0 退出。
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面环境出图
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

warnings.filterwarnings('ignore')

# ----------------------------- 全局配置 -----------------------------
SEED = 42
random.seed(SEED)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / 'data' / 'cache'
OUT_DIR = BASE / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

STOCKS = {                     # 5只银行股（code -> 简称）
    '601398': '工商银行',
    '601939': '建设银行',
    '600036': '招商银行',
    '601166': '兴业银行',
    '000001': '平安银行',
}
TRAIN_END = pd.Timestamp('2023-12-31')   # 样本内截止（样本外=2024-01-01起）
WINDOW = 60          # z-score 滚动窗口（交易日）
Z_IN = 2.0           # 开仓阈值：|z| >= 2
Z_OUT = 0.5          # 平仓阈值：|z| < 0.5
COST = 3e-4          # 单腿单边成本 万3（开/平各收一次，口径见 run_zscore_strategy 注释）
FREQ = 252           # 年化交易日
IS_SYNTHETIC = False # 数据是否为合成（正常情况下为 False）


# ----------------------------- 数据加载 -----------------------------
def load_prices():
    """读取5只银行股收盘价（2018年起，按共同交易日对齐）。
    任一文件缺失 → 用 rng(42) 造合成GBM价格并全局标注“合成数据”。"""
    global IS_SYNTHETIC
    closes = {}
    missing = [c for c, n in STOCKS.items()
               if not (DATA_DIR / f'stock_{c}_{n}.csv').exists()]
    if missing:
        IS_SYNTHETIC = True
        print(f'!! 数据缺失 {missing}，改用 rng(42) 合成数据（合成数据）!!')
        rng = np.random.default_rng(SEED)
        idx = pd.bdate_range('2018-01-02', periods=2100)
        for c, n in STOCKS.items():
            r = rng.normal(0.0002, 0.015, len(idx))
            closes[n] = 10.0 * np.exp(np.cumsum(r))
        px = pd.DataFrame(closes, index=idx)
    else:
        for c, n in STOCKS.items():
            df = pd.read_csv(DATA_DIR / f'stock_{c}_{n}.csv',
                             parse_dates=['date']).set_index('date')
            closes[n] = df['close']
        px = pd.DataFrame(closes).dropna()          # 内连接对齐交易日
        px = px.loc['2018-01-01':]
    return px


px = load_prices()
logp = np.log(px)                                    # 对数价格（本课主角）
ret = px.pct_change().fillna(0.0)                    # 简单收益率（回测计损益用）
logret = np.log(px).diff().fillna(0.0)               # 对数收益率（算相关用）
in_mask = px.index <= TRAIN_END                      # 样本内布尔掩码
names = list(STOCKS.values())
print(f'数据范围：{px.index[0].date()} ~ {px.index[-1].date()}，共 {len(px)} 个交易日'
      + ('（合成数据）' if IS_SYNTHETIC else ''))
print(f'样本内：{px.index[0].date()} ~ {TRAIN_END.date()}（{in_mask.sum()} 天），'
      f'样本外：{(~in_mask).sum()} 天')


# ----------------------------- 统计工具 -----------------------------
def eg_pvalue(y, x):
    """Engle-Granger 两步法协整检验。原假设：不存在协整关系。
    p 值小 → 拒绝原假设 → 两序列存在协整（价差平稳）。"""
    return float(coint(y, x, trend='c')[1])


def ols_hedge(y, x):
    """OLS 回归 y = a + b·x（对数价格），返回 (a, β, R²)。
    β 即对冲比率：1 单位 A 的头寸需要 β 单位 B 来对冲。"""
    X = sm.add_constant(x.values)
    res = sm.OLS(y.values, X).fit()
    return float(res.params[0]), float(res.params[1]), float(res.rsquared)


def adf_pvalue(s):
    """ADF 单位根检验。原假设：存在单位根（非平稳）。
    p 值小 → 拒绝原假设 → 序列平稳。"""
    return float(adfuller(np.asarray(s, dtype=float), autolag='AIC')[1])


def ou_half_life(s):
    """OU 过程：Δs_t = θ(μ - s_{t-1})Δt + ε，Δt=1个交易日。
    离散化后回归 Δs_t = a + b·s_{t-1}：斜率 b = -θ，截距 a = θμ。
    半衰期 = ln2 / θ = ln2 / (-b)（偏离均值的一半需要多少天）。"""
    s_lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    common = s_lag.index.intersection(ds.index)
    X = sm.add_constant(s_lag.loc[common].values)
    res = sm.OLS(ds.loc[common].values, X).fit()
    a, b = float(res.params[0]), float(res.params[1])
    if b >= 0:                       # 斜率非负 → 无均值回归 → 半衰期不存在
        return np.nan, b, a
    return float(np.log(2.0) / (-b)), b, a


def pair_stats(y, x):
    """一个配对的完整画像：协整p、β、价差ADF p、半衰期（在给定样本上）。"""
    a, beta, r2 = ols_hedge(y, x)
    spread = y - a - beta * x
    hl, b_slope, _ = ou_half_life(spread)
    return {'协整p值': eg_pvalue(y, x), 'α': a, 'β': beta, 'R2': r2,
            '价差ADF_p': adf_pvalue(spread), '半衰期天': hl}


# ----------------------------- z-score 策略回测 -----------------------------
def run_zscore_strategy(spread, beta, ret_a, ret_b,
                        window=WINDOW, z_in=Z_IN, z_out=Z_OUT, cost=COST):
    """价差 z-score 择时回测（同一套代码也用于步骤6的假想配对）。

    规则（信号当日收盘算出，次日收益生效，避免前视）：
      z < -2        → 买入价差：多 1 单位 A、空 β 单位 B（次日生效）
      z > +2        → 卖出价差：空 1 单位 A、多 β 单位 B
      |z| < 0.5     → 平仓
    成本口径：毛名义归一为 1（多1元A卖β元B → 除以1+β），每次换手
    cost × |Δ仓位|，即开仓 3bp、平仓 3bp，一个来回约 6bp（=双腿各单边万3）。
    持仓期间权重每日再平衡至 (1/(1+β), -β/(1+β))（未计再平衡成本，教学简化）。
    """
    idx = spread.index
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    z = z.replace([np.inf, -np.inf], np.nan)
    ra = ret_a.reindex(idx).ffill().values
    rb = ret_b.reindex(idx).ffill().values
    zv = z.values
    n = len(idx)

    nav = np.empty(n)
    cum = 1.0
    pos, pos_prev = 0.0, 0.0          # 今日生效仓位 / 昨日生效仓位
    open_trade = None
    trades, events = [], []
    for i in range(n):
        # ---- 1) 当日损益：多价差收益 = (rA - β·rB)/(1+β) ----
        spread_ret = (ra[i] - beta * rb[i]) / (1.0 + beta)
        daily = pos * spread_ret - cost * abs(pos - pos_prev)
        cum *= (1.0 + daily)
        nav[i] = cum
        if open_trade is not None:
            open_trade['nav'] *= (1.0 + daily)
        pos_prev = pos

        # ---- 2) 收盘信号 → 明日仓位 ----
        zi = zv[i]
        new_pos = pos
        if not np.isnan(zi):
            if open_trade is None:
                if zi <= -z_in:                       # 买入价差
                    new_pos = 1.0
                    open_trade = {'方向': '多价差(多A空B)', '开仓日': idx[i],
                                  '开仓i': i, '开仓z': zi, 'nav': 1.0, '强平': False}
                    events.append((idx[i], zi, spread.iloc[i], '开多'))
                elif zi >= z_in:                      # 卖出价差
                    new_pos = -1.0
                    open_trade = {'方向': '空价差(空A多B)', '开仓日': idx[i],
                                  '开仓i': i, '开仓z': zi, 'nav': 1.0, '强平': False}
                    events.append((idx[i], zi, spread.iloc[i], '开空'))
            elif abs(zi) < z_out:                     # 价差回归，平仓
                open_trade['平仓日'] = idx[i]
                open_trade['平仓i'] = i
                open_trade['持仓天数'] = i - open_trade['开仓i']
                open_trade['净收益%'] = (open_trade['nav'] - 1.0) * 100
                trades.append(open_trade)
                events.append((idx[i], zi, spread.iloc[i], '平仓'))
                open_trade = None
                new_pos = 0.0
        pos = new_pos

    # ---- 3) 样本结束仍在持仓 → 按最后一天收盘强制平仓（“被套”形态）----
    if open_trade is not None:
        open_trade['平仓日'] = idx[-1]
        open_trade['平仓i'] = n - 1
        open_trade['持仓天数'] = n - 1 - open_trade['开仓i']
        open_trade['净收益%'] = (open_trade['nav'] - 1.0) * 100
        open_trade['强平'] = True
        trades.append(open_trade)
        events.append((idx[-1], zv[-1], spread.iloc[-1], '强平'))

    nav_s = pd.Series(nav, index=idx, name='策略净值')
    strat_ret = nav_s.pct_change().fillna(nav_s.iloc[0] - 1.0)
    z_s = pd.Series(zv, index=idx, name='z-score')
    return {'nav': nav_s, 'ret': strat_ret, 'z': z_s,
            'trades': pd.DataFrame(trades), 'events': events}


def perf_stats(nav, rets):
    """一段净值的绩效：累计、年化、夏普(rf=0)、最大回撤。"""
    n = len(nav)
    if n == 0:
        return {'累计收益%': np.nan, '年化收益%': np.nan,
                '夏普比率': np.nan, '最大回撤%': np.nan}
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    ann = (1.0 + total) ** (FREQ / n) - 1.0 if n > 1 else total
    sharpe = rets.mean() / rets.std() * np.sqrt(FREQ) if rets.std() > 0 else np.nan
    mdd = float((nav / nav.cummax() - 1.0).min())
    return {'累计收益%': round(total * 100, 2), '年化收益%': round(ann * 100, 2),
            '夏普比率': round(float(sharpe), 2), '最大回撤%': round(mdd * 100, 2)}


def trade_stats(trades, mask):
    """某时间段内的交易统计：次数、平均持仓天数、胜率、单笔均收益。"""
    t = trades[trades['平仓日'].map(mask)] if len(trades) else trades
    if not len(t):
        return {'交易次数': 0, '平均持仓天数': np.nan, '胜率%': np.nan, '单笔均收益%': np.nan}
    return {'交易次数': int(len(t)),
            '平均持仓天数': round(float(t['持仓天数'].mean()), 1),
            '胜率%': round(float((t['净收益%'] > 0).mean() * 100), 1),
            '单笔均收益%': round(float(t['净收益%'].mean()), 2)}


def report_strategy(tag, res, hl):
    """打印某配对策略的样本内/样本外绩效拆分（供 README 引用）。"""
    nav, rets, trades = res['nav'], res['ret'], res['trades']
    m_in = nav.index <= TRAIN_END
    rows = {}
    rows['全样本'] = {**perf_stats(nav, rets), **trade_stats(trades, lambda d: True)}
    rows['样本内'] = {**perf_stats(nav[m_in], rets[m_in]),
                     **trade_stats(trades, lambda d: d <= TRAIN_END)}
    rows['样本外'] = {**perf_stats(nav[~m_in], rets[~m_in]),
                     **trade_stats(trades, lambda d: d > TRAIN_END)}
    df = pd.DataFrame(rows).T
    print(f'\n—— {tag} 绩效拆分 ——')
    print(df.to_string())
    if not np.isnan(hl):
        print(f'对照：OU半衰期 {hl:.1f} 天 vs 全样本平均持仓 '
              f'{rows["全样本"]["平均持仓天数"]} 天')
    else:
        print('对照：价差未通过平稳性检验，半衰期不存在（这正是亏损的根源）')
    return df


# ============================= 步骤1：相关 ≠ 协整 =============================
print('\n================ 步骤1：相关系数 vs 协整检验 ================')
corr_price = logp.corr()        # 对数价格水平相关（共同趋势会推得很高）
corr_ret = logret.corr()        # 日收益率相关（真实的“联动”）

pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
rows = []
for na, nb in pairs:            # 依次以 na 为 y、nb 为 x（EG 检验对顺序略敏感）
    s_in = pair_stats(logp.loc[in_mask, na], logp.loc[in_mask, nb])    # 样本内：选对/定参
    s_full = pair_stats(logp[na], logp[nb])                            # 全样本：参考
    rows.append({'股票A': na, '股票B': nb,
                 '价格相关系数': round(corr_price.loc[na, nb], 3),
                 '收益相关系数': round(corr_ret.loc[na, nb], 3),
                 '协整p_样本内': round(s_in['协整p值'], 4),
                 '协整p_全样本': round(s_full['协整p值'], 4),
                 'β_样本内': round(s_in['β'], 3),
                 'R2_样本内': round(s_in['R2'], 3),
                 '价差ADFp_样本内': round(s_in['价差ADF_p'], 4),
                 '半衰期天_样本内': (round(s_in['半衰期天'], 1)
                                if not np.isnan(s_in['半衰期天']) else np.nan)})
pair_tbl = pd.DataFrame(rows).sort_values('协整p_样本内')
print(pair_tbl.to_string(index=False))

# p 值矩阵（对角线=自身，填0便于读图；EG 检验方向按上表 A→B）
def build_pmat(col):
    m = pd.DataFrame(0.0, index=names, columns=names)
    for _, r in pair_tbl.iterrows():
        m.loc[r['股票A'], r['股票B']] = m.loc[r['股票B'], r['股票A']] = r[col]
    return m
pmat_in, pmat_full = build_pmat('协整p_样本内'), build_pmat('协整p_全样本')
# coint_pvalues.csv：两张 10x10 的 EG 检验 p 值矩阵（样本内=选对依据，全样本=参考）
with open(OUT_DIR / 'coint_pvalues.csv', 'w', encoding='utf-8-sig', newline='') as f:
    f.write('Engle-Granger协整检验p值矩阵——样本内2018-2023（选对用，<0.05视为协整）\n')
    pmat_in.to_csv(f)
    f.write('\nEngle-Granger协整检验p值矩阵——全样本2018-2026（参考，仅事后比较）\n')
    pmat_full.to_csv(f)
# pair_metrics.csv：每个配对的完整画像（相关、协整p、β、R²、ADF p、半衰期）
pair_tbl.to_csv(OUT_DIR / 'pair_metrics.csv', index=False, encoding='utf-8-sig')


def draw_heat(mat, title, ax, fmt, cmap, vmin, vmax, cbar_label):
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            v = mat.values[i, j]
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color='white' if (vmax - v) < 0.25 * (vmax - vmin) or
                    (v - vmin) < 0.18 * (vmax - vmin) else 'black')
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.set_ylabel(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=8)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
draw_heat(corr_price, '对数价格相关系数（共同趋势推高，普遍>0.8）',
          axes[0], '{:.2f}', 'RdYlGn', 0.4, 1.0, '相关系数')
draw_heat(corr_ret, '日对数收益率相关系数（真实短线联动）',
          axes[1], '{:.2f}', 'RdYlGn', 0.0, 1.0, '相关系数')
fig.suptitle('图1  银行股两两相关性：价格相关 ≠ 收益相关，更 ≠ 协整', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT_DIR / '01_相关系数热力图.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
draw_heat(pmat_in, 'Engle-Granger 协整检验 p 值（样本内 2018-2023，选对用）',
          axes[0], '{:.3f}', 'RdYlGn_r', 0.0, 0.3, 'p值（<0.05绿=协整）')
draw_heat(pmat_full, 'Engle-Granger 协整检验 p 值（全样本 2018-2026，参考）',
          axes[1], '{:.3f}', 'RdYlGn_r', 0.0, 0.3, 'p值（<0.05绿=协整）')
fig.suptitle('图2  协整 p 值矩阵：相关性高不代表价差平稳（原假设=不协整，p小才好）',
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT_DIR / '02_协整p值矩阵.png', dpi=150)
plt.close(fig)

# ============================= 步骤2：选对并估计 β =============================
print('\n================ 步骤2：最佳配对与对冲比率 β ================')
best = pair_tbl.iloc[0]
A_CODE = [c for c, n in STOCKS.items() if n == best['股票A']][0]
B_CODE = [c for c, n in STOCKS.items() if n == best['股票B']][0]
name_a, name_b = best['股票A'], best['股票B']
alpha, beta, r2 = ols_hedge(logp.loc[in_mask, name_a], logp.loc[in_mask, name_b])
spread = logp[name_a] - alpha - beta * logp[name_b]   # 全样本价差（参数冻结于样本内）
print(f"样本内协整 p 值最小的配对：{name_a}({A_CODE}) vs {name_b}({B_CODE})，"
      f"p = {best['协整p_样本内']:.4f}")
print(f"OLS(样本内)：log({name_a}) = {alpha:.4f} + {beta:.4f} · log({name_b})，"
      f"R² = {r2:.3f}")
print(f"含义：每做多 1 元 {name_a}，需做空 β·元 ≈ {beta:.3f} 元 {name_b}")

# ============================= 步骤3：ADF + OU 半衰期 =============================
print('\n================ 步骤3：价差平稳性与半衰期 ================')
adf_p_in = adf_pvalue(spread[in_mask])
adf_p_full = adf_pvalue(spread)
half_life, ou_b, ou_a = ou_half_life(spread[in_mask])
print(f"价差 ADF 检验：样本内 p = {adf_p_in:.4f}（<0.05 → 平稳），"
      f"全样本 p = {adf_p_full:.4f}")
print(f"OU 回归(样本内)：Δs_t = {ou_a:.2e} + ({ou_b:.4f})·s_{{t-1}} → "
      f"θ = {-ou_b:.4f}/天")
print(f"半衰期 = ln2/θ = {half_life:.1f} 个交易日（约 {half_life/21:.1f} 个月）"
      if not np.isnan(half_life) else '半衰期不存在（斜率非负，无均值回归）')

# ============================= 步骤4/5：回测与绩效拆分 =============================
print('\n================ 步骤4/5：z-score 策略回测（成本万3） ================')
res_real = run_zscore_strategy(spread, beta, ret[name_a], ret[name_b])
perf_real = report_strategy(f'真实配对 {name_a} vs {name_b}', res_real, half_life)

# ============================= 步骤6：反例——协整关系破裂（rng(42) 构造） =============================
print('\n================ 步骤6：反例：样本内协整、2024年起关系破裂（rng(42) 构造） ================'
      '\n构造：假想建设银行X = 真实建设银行路径；2024年（样本外起点）起乘上'
      '\n  exp(漂移·t + rng(42)随机游走)——模拟“重组后业绩爆发”、相对工行持续跑赢，'
      '\n  与工行的相关性依旧很高，但协整关系从此断裂，价差单边发散、永不回归。')
rng_fake = np.random.default_rng(SEED)          # 独立种子实例，结果可复现
BREAK = px.index[px.index > TRAIN_END][0]       # 断裂日 = 2024年第一个交易日
post = np.asarray(px.index >= BREAK)            # 断裂后的布尔掩码（ndarray）
RW_SIGMA = 0.001      # 断裂后叠加的随机游走日波动（rng(42)）
FAKE_DRIFT = 0.0012   # 断裂后的相对漂移：假想建行每年额外跑赢约 e^(0.0012*252)-1 ≈ 35%
rw = np.cumsum(np.where(post, rng_fake.normal(0.0, RW_SIGMA, len(px)), 0.0))
i_break = int(np.argmax(post))                  # 断裂日的位置索引
t_rel = np.maximum(np.arange(len(px)) - i_break, 0)   # 断裂日起算的相对时间（从0爬坡）
fake_log = logp[name_b].values + FAKE_DRIFT * t_rel + rw
fake_px = pd.Series(np.exp(fake_log), index=px.index, name='假想建设银行X')

# 同一套流程：样本内估计 α/β（此时假想X=真实建行，检验“完全通过”）→ 全样本跑策略
alpha_f, beta_f, r2_f = ols_hedge(logp.loc[in_mask, name_a], np.log(fake_px)[in_mask])
spread_f = logp[name_a] - alpha_f - beta_f * np.log(fake_px)
p_coint_f_in = eg_pvalue(logp.loc[in_mask, name_a], np.log(fake_px)[in_mask])
p_coint_f_full = eg_pvalue(logp[name_a], np.log(fake_px))
adf_p_f = adf_pvalue(spread_f)
hl_f, b_f, _ = ou_half_life(spread_f)
corr_px_f = np.corrcoef(logp[name_a], np.log(fake_px))[0, 1]
corr_rt_f = np.corrcoef(logret[name_a],
                        pd.Series(np.log(fake_px), index=px.index).diff().fillna(0))[0, 1]
print(f"假想配对：{name_a} vs 假想建设银行X（全样本价格相关 {corr_px_f:.2f}，"
      f"收益相关 {corr_rt_f:.2f} —— 看相关性会以为和真建行一样可靠）")
print(f"样本内协整 p = {p_coint_f_in:.4f}（完全通过！β = {beta_f:.3f}）——"
      "但检验只保证“历史上协整”")
print(f"断裂后（全样本口径）：协整 p = {p_coint_f_full:.4f}、价差 ADF p = {adf_p_f:.4f}"
      "（双双失效 → 关系已死，价差随机游走+漂移，永不回归）")

res_fake = run_zscore_strategy(spread_f, beta_f, ret[name_a],
                               fake_px.pct_change().fillna(0.0))
perf_fake = report_strategy(f'反例配对 {name_a} vs 假想建设银行X', res_fake, np.nan)

# ============================= 图3：价差与 z-score 开平仓 =============================
def add_event_marker(ax, d, s, k, used):
    """画开/平仓标记，每种类型只进一次图例。"""
    styles = {'开多': ('^', 'green', '开多价差(z<-2)'),
              '开空': ('v', 'red', '开空价差(z>+2)'),
              '平仓': ('x', 'black', '平仓(|z|<0.5)'),
              '强平': ('x', 'black', '样本末强平')}
    m, c, lab = styles[k]
    ax.scatter(d, s, marker=m, s=70, color=c, zorder=5,
               label=None if lab in used else lab)
    used.add(lab)


fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                         gridspec_kw={'height_ratios': [3, 2]})
ax = axes[0]
ax.plot(spread.index, spread.values, lw=0.9, color='#1f4e79', label='价差 = log(A) - α - β·log(B)')
ax.axhline(spread[in_mask].mean(), color='gray', ls='--', lw=0.8, label='样本内均值=0（OLS残差）')
ax.axvline(TRAIN_END, color='red', ls=':', lw=1.2)
ax.text(TRAIN_END, ax.get_ylim()[0], ' ←样本内 | 样本外→ ', color='red', fontsize=9)
used_labels = set()
for d, z, s, k in res_real['events']:
    add_event_marker(ax, d, s, k, used_labels)
ax.set_title(f'图3上  {name_a} vs {name_b} 对数价差与开平仓点（β={beta:.3f}，样本内估计）')
ax.legend(loc='upper left', fontsize=8, ncol=3)
ax.set_ylabel('对数价差')

ax = axes[1]
ax.plot(res_real['z'].index, res_real['z'].values, lw=0.8, color='#555')
for lv, txt in [(2, '+2 开空'), (-2, '-2 开多'), (0.5, '+0.5'), (-0.5, '-0.5')]:
    ax.axhline(lv, color='orange' if abs(lv) == 2 else 'gray', ls='--', lw=0.9)
    ax.text(spread.index[5], lv + 0.08, txt, fontsize=8, color='brown')
ax.axvline(TRAIN_END, color='red', ls=':', lw=1.2)
ax.set_title(f'图3下  价差 z-score（滚动{WINDOW}日标准化）')
ax.set_ylabel('z')
fig.tight_layout()
fig.savefig(OUT_DIR / '03_价差与z-score开平仓.png', dpi=150)
plt.close(fig)

# ============================= 图4：净值对比 =============================
fig, ax = plt.subplots(figsize=(12.5, 6))
nav = res_real['nav']
ax.plot(nav.index, nav.values, lw=1.5, color='#c0392b',
        label=f'价差多空策略（期末 {nav.iloc[-1]:.3f}）')
hold_a = px[name_a] / px[name_a].iloc[0]
hold_b = px[name_b] / px[name_b].iloc[0]
ax.plot(hold_a.index, hold_a.values, lw=1.1, color='#7f8c8d',
        label=f'{name_a} 买入持有（期末 {hold_a.iloc[-1]:.2f}）')
ax.plot(hold_b.index, hold_b.values, lw=1.1, color='#bdc3c7', ls='--',
        label=f'{name_b} 买入持有（期末 {hold_b.iloc[-1]:.2f}）')
ax.axvline(TRAIN_END, color='red', ls=':', lw=1.4)
oos = nav.index > TRAIN_END
ax.axvspan(TRAIN_END, nav.index[-1], alpha=0.06, color='red')
ann_in = perf_real.loc['样本内', '年化收益%']
ann_oos = perf_real.loc['样本外', '年化收益%']
ax.text(0.02, 0.95, f'样本内年化 {ann_in}%  |  样本外年化 {ann_oos}%（参数冻结，如实呈现衰减）',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))
ax.set_title(f'图4  净值对比：{name_a} vs {name_b} 价差多空组合（市场中性） vs 单腿买入持有')
ax.set_ylabel('净值（起点=1）')
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / '04_净值对比.png', dpi=150)
plt.close(fig)

# ============================= 图5：反例假想配对 =============================
fig, axes = plt.subplots(3, 1, figsize=(12.5, 11))
ax = axes[0]
ax.plot(px.index, px[name_b] / px[name_b].iloc[0], lw=1.2,
        color='#1f4e79', label=f'{name_b}（真实）')
ax.plot(fake_px.index, fake_px.values / fake_px.iloc[0], lw=1.2,
        color='#8e44ad', label='假想建设银行X（2024年起叠加rng(42)漂移+随机游走）')
ax.axvline(BREAK, color='red', ls=':', lw=1.4)
ax.text(BREAK, ax.get_ylim()[0], ' 关系断裂 ', color='red', fontsize=9)
ax.set_title(f'图5a  归一化价格：全样本价格相关 {corr_px_f:.2f}、收益相关 {corr_rt_f:.2f}'
             '——断裂后仍“高相关”，但协整已死')
ax.legend(fontsize=9)
ax.set_ylabel('归一化价格')

ax = axes[1]
ax.plot(spread_f.index, spread_f.values, lw=0.9, color='#8e44ad')
ax.plot(spread.index, spread.values, lw=0.7, color='#1f4e79', alpha=0.55,
        label=f'真实价差（对照，{name_a} vs {name_b}）')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(BREAK, color='red', ls=':', lw=1.4)
used_labels_f = set()
for d, z, s, k in res_fake['events']:
    add_event_marker(ax, d, s, k, used_labels_f)
ax.set_title(f'图5b  假想配对的价差：样本内协整 p={p_coint_f_in:.3f}（通过！），'
             f'断裂后 ADF p={adf_p_f:.3f}（非平稳）→ 价差单边发散，抄底=接飞刀')
ax.legend(fontsize=9)
ax.set_ylabel('对数价差')

ax = axes[2]
ax.plot(nav.index, nav.values, lw=1.4, color='#c0392b',
        label=f'真实配对策略（期末 {nav.iloc[-1]:.3f}）')
navf = res_fake['nav']
ax.plot(navf.index, navf.values, lw=1.4, color='#8e44ad', ls='--',
        label=f'假想配对策略（期末 {navf.iloc[-1]:.3f}，最大回撤 '
              f'{perf_fake.loc["全样本", "最大回撤%"]}%）')
ax.axvline(BREAK, color='red', ls=':', lw=1.4)
ax.set_title('图5c  同一套 z-score 策略、同一套参数：协整持续 vs 断裂，结局天差地别')
ax.legend(fontsize=9)
ax.set_ylabel('净值（起点=1）')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / '05_反例假想配对.png', dpi=150)
plt.close(fig)

# ============================= 汇总输出 =============================
print('\n================ 关键结论汇总（README 引用） ================'
      f'\n1) 样本内协整 p 值最小的配对：{name_a} vs {name_b}，p={best["协整p_样本内"]:.4f}，'
      f'β={beta:.4f}，R²={r2:.3f}'
      f'\n2) 价差 ADF p（样本内）={adf_p_in:.4f}；OU 半衰期={half_life:.1f} 个交易日'
      f'\n3) 策略样本内：年化 {perf_real.loc["样本内", "年化收益%"]}%，夏普 '
      f'{perf_real.loc["样本内", "夏普比率"]}，回撤 {perf_real.loc["样本内", "最大回撤%"]}%，'
      f'{perf_real.loc["样本内", "交易次数"]} 次交易，平均持仓 '
      f'{perf_real.loc["样本内", "平均持仓天数"]} 天'
      f'\n   策略样本外：年化 {perf_real.loc["样本外", "年化收益%"]}%，夏普 '
      f'{perf_real.loc["样本外", "夏普比率"]}，回撤 {perf_real.loc["样本外", "最大回撤%"]}%，'
      f'{perf_real.loc["样本外", "交易次数"]} 次交易，平均持仓 '
      f'{perf_real.loc["样本外", "平均持仓天数"]} 天'
      f'\n4) 反例（协整破裂）：样本内协整 p={p_coint_f_in:.4f} 完全通过，断裂后全样本 '
      f'p={p_coint_f_full:.4f}、ADF p={adf_p_f:.4f}；价格相关 {corr_px_f:.2f}/收益相关 '
      f'{corr_rt_f:.2f}；策略期末净值 {res_fake["nav"].iloc[-1]:.3f}，最大回撤 '
      f'{perf_fake.loc["全样本", "最大回撤%"]}%，交易 {perf_fake.loc["全样本", "交易次数"]} 次，'
      f'最长单笔持仓 {res_fake["trades"]["持仓天数"].max() if len(res_fake["trades"]) else np.nan} 天')
print('\n全部图表与CSV已保存至 outputs/，运行完成。')
