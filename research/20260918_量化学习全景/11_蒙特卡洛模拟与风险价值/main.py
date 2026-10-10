# -*- coding: utf-8 -*-
"""
11 蒙特卡洛模拟与风险价值（VaR / ES）
=====================================
用一万种未来回答一个问题："最坏会坏到什么程度？"

单文件自包含，依次完成：
  §1 数据加载与参数估计（贵州茅台 2015 年起，对比沪深300）
  §2 GBM 几何布朗运动：10000条路径 × 一年252天 → outputs/01_一万条路径扇形图.png
  §3 三种分布假设下的一年期 VaR/ES（正态参数法 / 历史模拟法 / t分布 / bootstrap）→ 02_三种方法收益率分布与VaR.png
  §4 组合（50%茅台+50%沪深300）：相关性对组合 VaR 的削弱 → 03_组合回撤分布.png
  §5 压力测试三情景 → outputs/stress_tests.csv
  §6 持有期 1/10/250 天与 √t 法则 → 04_持有期与VaR关系.png
  §7 汇总输出 outputs/var_results.csv 和控制台"README 数字速查"

运行：
  cd 11_蒙特卡洛模拟与风险价值
  /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py

全局随机种子 42；数据缺失时自动降级为合成数据（rng(42)）并标注，绝不失败退出非0。
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use('Agg')  # 无显示器环境出图
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 全局常量 ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.normpath(os.path.join(BASE, '..', 'data', 'cache'))
SEED = 42          # 全局随机种子（每个随机模块内部各自 default_rng(42)，保证可复现）
N_PATHS = 10000    # 蒙特卡洛路径数
DAYS = 252         # 一年交易日
W_MT, W_HS = 0.5, 0.5   # 组合权重：50% 茅台 + 50% 沪深300
CAPITAL = 1_000_000     # 压力测试用：100 万元组合

T0 = time.time()
SPEED_REMARKS = []   # 汇总打印用


# ============================================================
# §1 数据加载与参数估计
# ============================================================
def load_close(path, name):
    """读缓存收盘价（2015年起）；文件缺失则用 rng(42) 造 GBM 合成数据并标注。"""
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8-sig', parse_dates=['date']).set_index('date')
        s = df['close'].loc[df.index >= pd.Timestamp('2015-01-01')].dropna()
        return s, False
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range('2015-01-05', periods=2845)
    r = rng.normal(0.0008, 0.02, size=2845)          # 合成：日μ=0.08%, 日σ=2%
    print(f'[警告] 未找到 {path}，已改用合成数据（rng(42) GBM，μ_d=0.08%/日, σ_d=2%/日）')
    return pd.Series(143.0 * np.exp(np.cumsum(r)), index=idx, name=name), True


px_mt, syn_mt = load_close(os.path.join(DATA_DIR, 'stock_600519_贵州茅台.csv'), '贵州茅台')
px_hs, syn_hs = load_close(os.path.join(DATA_DIR, 'index_000300_沪深300.csv'), '沪深300')
SYNTH = syn_mt or syn_hs

# 按共同交易日对齐（茅台自 2015 年起，故对齐后即 2015 年至今）
df = pd.concat([px_mt.rename('mt'), px_hs.rename('hs')], axis=1, join='inner').dropna()

# 对数收益（跨期可加，是模拟与年化的语言）与组合日收益（截面可加，是权重的语言）
lr_mt = np.log(df['mt']).diff().dropna()             # 茅台日对数收益
lr_hs = np.log(df['hs']).diff().dropna()             # 沪深300日对数收益
rp_hist = W_MT * (np.exp(lr_mt) - 1) + W_HS * (np.exp(lr_hs) - 1)   # 组合日简单收益
lp_hist = np.log1p(rp_hist)                          # 组合日对数收益（用于重采样）

mu_mt, sd_mt = lr_mt.mean(), lr_mt.std(ddof=1)       # 日频参数
mu_hs, sd_hs = lr_hs.mean(), lr_hs.std(ddof=1)
mu_p, sd_p = lp_hist.mean(), lp_hist.std(ddof=1)
mu_log_ann = mu_mt * DAYS                            # 年化对数漂移 = 252 × 日μ
sigma_ann = sd_mt * np.sqrt(DAYS)                    # 年化波动率 = √252 × 日σ
mu_gbm_ann = mu_log_ann + 0.5 * sigma_ann ** 2       # GBM 漂移 μ（Itô修正，E[S_T]=S0·e^{μ}）
rho = float(np.corrcoef(lr_mt, lr_hs)[0, 1])         # 两资产日收益相关系数

print('=' * 72)
print('§1 数据与参数')
print(f'  数据源: {"合成数据(rng42)" if SYNTH else "data/cache 真实缓存"}；'
      f'对齐区间 {df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 个交易日')
print(f'  茅台最新收盘 {df["mt"].iloc[-1]:.2f} 元；日μ={mu_mt:.5%}, 日σ={sd_mt:.4%}')
print(f'  年化: 对数漂移 μ_log={mu_log_ann:.2%}, 波动率 σ={sigma_ann:.2%}, GBM漂移 μ_GBM={mu_gbm_ann:.2%}')
print(f'  沪深300: 日μ={mu_hs:.5%}, 日σ={sd_hs:.4%}, 年化σ={sd_hs * np.sqrt(DAYS):.2%}')
print(f'  组合(50/50): 日σ={sd_p:.4%}；茅台-沪深300日收益相关系数 ρ={rho:.3f}')
print(f'  分散化效果: 组合σ {sd_p:.4%} vs 加权平均σ {0.5 * sd_mt + 0.5 * sd_hs:.4%} '
      f'→ 降 {1 - sd_p / (0.5 * sd_mt + 0.5 * sd_hs):.1%}')

# ---- 厚尾证据：真实极端日 vs 正态模型的预测 ----
sk_mt = stats.skew(lr_mt)
ku_mt = stats.kurtosis(lr_mt)                        # 超额峰度，正态=0
_jb = stats.jarque_bera(lr_mt.values)
jb_stat, jb_p = (_jb[0], _jb[1]) if isinstance(_jb, tuple) else (_jb.statistic, _jb.pvalue)
z999 = stats.norm.ppf(0.001)                         # -3.09
n999_real = int((lr_mt <= mu_mt + z999 * sd_mt).sum())      # 真实跌破正态99.9%分位的天数
n999_exp = len(lr_mt) * 0.001                                # 正态模型的期望天数
n99_real = int((lr_mt <= mu_mt + stats.norm.ppf(0.01) * sd_mt).sum())
n99_exp = len(lr_mt) * 0.01
worst3 = lr_mt.nsmallest(3)
print('  [厚尾证据] 茅台日收益: 偏度={:.2f}, 超额峰度={:.2f}（正态=0）, JB统计量={:.0f}(p={:.1e})'.format(
    sk_mt, ku_mt, jb_stat, jb_p))
print(f'    跌破正态99.9%分位({mu_mt + z999 * sd_mt:.2%})的天数: 真实{n999_real}天 vs 正态预期{n999_exp:.1f}天')
print(f'    跌破正态99%分位({mu_mt + stats.norm.ppf(0.01) * sd_mt:.2%})的天数: 真实{n99_real}天 vs 正态预期{n99_exp:.0f}天')
print('    历史最差3个交易日: ' + '；'.join(f'{d.date()} {v:.2%}' for d, v in worst3.items()))

# μ 的子区间不稳定性（README"常见的坑"素材）
print('  [μ的不稳定性] 茅台年化对数漂移按子区间:')
for tag, a, b in [('2015-2018', '2015-01-01', '2018-12-31'), ('2019-2021', '2019-01-01', '2021-12-31'),
                  ('2022-至今', '2022-01-01', '2026-12-31')]:
    seg = lr_mt.loc[a:b]
    print(f'    {tag}: {(seg.mean() * DAYS):+.1%}（{len(seg)}个交易日）')
print('  [数据核验备注] ①茅台2018-10-29缓存 open=close=前收（疑似行情错位），10-30的"单日-15.21%"实为两日累计-14.1%，'
      '真实最差单日应视为2015-08-24的-9.38%（跌停）；②沪深300缓存个别极端日幅度与公开指数数据有出入'
      '（如2015-01-19缓存-10.53% vs 公开约-7.9%），且量纲非指数点位（疑似ETF复权序列）；'
      '本课与课程01-08保持同一缓存口径，结论以相对比较为主。')


# ============================================================
# 通用工具：VaR / ES 计算（输入=持有期总对数收益样本，输出=简单收益口径的正损失）
# ============================================================
def var_es_from_logsum(sample, cl):
    """给定持有期总对数收益样本，返回 (VaR, ES)。
    VaR  = 1 - exp(收益分布的 α 分位)  —— 有 α 的概率损失超过它；
    ES   = 1 - exp(落入尾部的样本均值) —— 尾部的条件平均损失（一定 ≥ VaR）。"""
    alpha = 1 - cl
    q = np.quantile(sample, alpha)
    tail = sample[sample <= q]
    return 1 - np.exp(q), (1 - np.exp(tail.mean()) if tail.size else 1 - np.exp(q))


def normal_analytic_var_es(mu_d, sd_d, h, cl):
    """正态参数法（解析式）：h 天总对数收益 ~ N(mu_d·h, sd_d·√h)。
    ES 用正态尾部条件均值公式 E[X|X≤q] = μ - σ·φ(z)/α。√t 法则在此精确成立。"""
    alpha = 1 - cl
    z = stats.norm.ppf(alpha)
    q = mu_d * h + z * sd_d * np.sqrt(h)
    es_log = mu_d * h - sd_d * np.sqrt(h) * stats.norm.pdf(z) / alpha
    return 1 - np.exp(q), 1 - np.exp(es_log)


def bootstrap_h(lr, h, n=N_PATHS, seed=SEED):
    """bootstrap 自助重采样：从真实日对数收益中有放回抽 h 个、加总为一条 h 天路径，共 n 条。"""
    rng = np.random.default_rng(seed)
    arr = np.asarray(lr)
    return rng.choice(arr, size=(n, h), replace=True).sum(axis=1)


def shuffle_h(lr, h, n=N_PATHS, seed=SEED):
    """历史模拟法（重排）：把真实日收益随机洗牌，取前 h 天复利为一年，重排 n 次。"""
    rng = np.random.default_rng(seed)
    arr = np.asarray(lr)
    out = np.empty(n)
    for i in range(n):
        out[i] = arr[rng.permutation(len(arr))[:h]].sum()
    return out


def t_sim_h(lr, h, n=N_PATHS, seed=SEED):
    """t分布法：对真实日收益做 MLE 拟合自由度，再从拟合 t 分布抽 h 天模拟（厚尾参数法）。"""
    rng = np.random.default_rng(seed)
    arr = np.asarray(lr)
    df_t, loc_t, sc_t = stats.t.fit(arr)
    return stats.t.rvs(df_t, loc=loc_t, scale=sc_t, size=(n, h), random_state=rng).sum(axis=1), (df_t, loc_t, sc_t)


def real_window_h(lr, h):
    """真实历史重叠窗口：序列上所有长度 h 的真实滚动区间收益（保留波动聚集，但样本重叠）。"""
    return pd.Series(np.asarray(lr)).rolling(h).sum().dropna().values


RES = {}     # (标的, 方法, 持有期, 置信水平) -> (VaR, ES)，§7 汇总写 CSV 用
ROWS = []    # CSV 行


def add_row(asset, method, h, cl, var, es, note=''):
    RES[(asset, method, h, cl)] = (var, es)
    ROWS.append({'标的': asset, '方法': method, '持有期天数': h, '置信水平': f'{cl:.0%}',
                 'VaR': round(var, 6), 'ES': round(es, 6), '备注': note})


# ============================================================
# §2 GBM 几何布朗运动：10000 条路径 × 一年 252 天
# ============================================================
print('\n' + '=' * 72)
print('§2 GBM 蒙特卡洛模拟（茅台，10000路径×252天）')

S0 = float(df['mt'].iloc[-1])          # 从最新真实价格出发
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_PATHS, DAYS))
# 向量化核心：一次生成 (10000, 252) 的标准正态矩阵，日对数收益 = μ_d + σ_d·Z。
# 等价于 GBM dS/S=μdt+σdW 的精确离散解 ln(S_{t+1}/S_t) = (μ-σ²/2)Δt + σ√Δt·Z，
# 其中数据标定给出 μ_GBM = μ_log + σ²/2，代入后漂移项恰为 μ_log（见 mu_gbm_ann）。
log_r = mu_mt + sd_mt * Z
log_S = np.hstack([np.zeros((N_PATHS, 1)), np.cumsum(log_r, axis=1)])   # 第0列=期初
paths = S0 * np.exp(log_S)                                              # (10000, 253) 价格路径
norm_paths = paths / S0                                                 # 归一化净值

# 分位数扇形：沿"路径轴"直接对每个交易日取 5/25/50/75/95 分位 —— 完全向量化，无循环
pct = np.percentile(norm_paths, [5, 25, 50, 75, 95], axis=0)

term = paths[:, -1]
p_loss = float(np.mean(term < S0))
p_loss_theory = stats.norm.cdf(-mu_log_ann / sigma_ann)
print(f'  期末净值分位: 5%={pct[0, -1]:.2f}, 25%={pct[1, -1]:.2f}, 中位={pct[2, -1]:.2f}, '
      f'75%={pct[3, -1]:.2f}, 95%={pct[4, -1]:.2f}')
print(f'  期末价格(元): 5%分位={np.percentile(term, 5):.0f}; 中位={np.median(term):.0f}; '
      f'均值={term.mean():.0f}（理论 S0·e^μ_GBM={S0 * np.exp(mu_gbm_ann):.0f}）; 95%分位={np.percentile(term, 95):.0f}')
print(f'  一年后亏钱概率: 模拟 {p_loss:.1%} vs 理论 Φ(-μ_log/σ)={p_loss_theory:.1%}（漂移为正仍有约1/4概率亏钱）')

fig, ax = plt.subplots(figsize=(12.5, 6.8))
x = np.arange(DAYS + 1)
ax.plot(x, norm_paths[:100].T, color='steelblue', alpha=0.13, lw=0.7)   # 100条样例路径
ax.fill_between(x, pct[0], pct[4], color='steelblue', alpha=0.13, label='5%~95% 分位带')
ax.fill_between(x, pct[1], pct[3], color='steelblue', alpha=0.28, label='25%~75% 分位带')
ax.plot(x, pct[2], color='navy', lw=2.2, label='中位数路径(50%)')
ax.axhline(1.0, color='crimson', ls='--', lw=1.2, label='期初净值 = 1')
for k, txt in [(0, '5%分位'), (2, '中位数'), (4, '95%分位')]:
    ax.text(DAYS * 1.005, pct[k, -1], f'{txt}终值 {pct[k, -1]:.2f}\n({S0 * pct[k, -1]:.0f}元)',
            fontsize=9, color='navy' if k == 2 else 'dimgray', va='center')
ax.set_xlim(0, DAYS * 1.17)
ax.set_xlabel('交易日（0 = 今天）')
ax.set_ylabel('净值（期初 = 1）')
ax.set_title(f'图01 贵州茅台 GBM 蒙特卡洛：{N_PATHS}条路径 × 一年{DAYS}天（灰蓝色细线为100条样例路径）\n'
             f'参数（2015年以来真实数据估计）：日μ={mu_mt:.4%} → 年化对数漂移 {mu_log_ann:.1%}，'
             f'日σ={sd_mt:.2%} → 年化波动率 {sigma_ann:.1%}')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '01_一万条路径扇形图.png'), dpi=150)
plt.close(fig)


# ============================================================
# §3 三种分布假设下的一年期 VaR / ES —— 本课灵魂：厚尾
# ============================================================
print('\n' + '=' * 72)
print('§3 三种分布假设下的一年期 VaR/ES（茅台）')

ASSETS = {'贵州茅台': lr_mt, '沪深300': lr_hs, '组合(50%茅台+50%沪深300)': lp_hist}
samples_1y = {}     # 画图02用：茅台各方法的一年期总对数收益样本
t_fit_mt = None

for asset, lr in ASSETS.items():
    m_d, s_d = lr.mean(), lr.std(ddof=1)
    # (a) 正态参数法（解析公式，h=1 与 h=250）
    for h in (1, DAYS):
        for cl in (0.95, 0.99):
            v, e = normal_analytic_var_es(m_d, s_d, h, cl)
            add_row(asset, '正态参数法(解析)', h, cl, v, e, 'μ-1.65σ/μ-2.33σ型解析式')
    # (b) 正态蒙特卡洛（校验解析式：一年总对数收益 ~ N(250μ, √250·σ)）
    rng_chk = np.random.default_rng(SEED)
    chk = rng_chk.normal(m_d * DAYS, s_d * np.sqrt(DAYS), size=N_PATHS)
    for cl in (0.95, 0.99):
        v, e = var_es_from_logsum(chk, cl)
        add_row(asset, '正态蒙特卡洛(校验)', DAYS, cl, v, e, '应与解析法几乎一致')
    # (c) 历史模拟法：真实日收益重排拼一年
    sh = shuffle_h(lr, DAYS)
    for cl in (0.95, 0.99):
        v, e = var_es_from_logsum(sh, cl)
        add_row(asset, '历史模拟法(重排)', DAYS, cl, v, e, '真实日收益随机重排×252天复利')
    # (d) bootstrap 自助重采样（有放回）
    bs = bootstrap_h(lr, DAYS)
    for cl in (0.95, 0.99):
        v, e = var_es_from_logsum(bs, cl)
        add_row(asset, 'bootstrap重采样', DAYS, cl, v, e, 'rng(42)有放回重采样10000次')
    # (e) t 分布蒙特卡洛（厚尾参数法）
    ts, (df_t, loc_t, sc_t) = t_sim_h(lr, DAYS)
    for cl in (0.95, 0.99):
        v, e = var_es_from_logsum(ts, cl)
        add_row(asset, 't分布蒙特卡洛', DAYS, cl, v, e, f'MLE拟合t自由度={df_t:.1f}')
    # (f) 日频经验分位数 / t拟合分位数（教科书口径的"1天VaR"）
    emp = np.asarray(lr)
    for cl in (0.95, 0.99):
        v, e = var_es_from_logsum(emp, cl)
        add_row(asset, '历史经验分位数', 1, cl, v, e, '真实日收益经验分位数(历史模拟法日频)')
    rng_t1 = np.random.default_rng(SEED)
    dt_, lt_, st_ = stats.t.fit(emp)
    t1 = stats.t.rvs(dt_, loc=lt_, scale=st_, size=N_PATHS, random_state=rng_t1)
    for cl in (0.95, 0.99):
        v, e = var_es_from_logsum(t1, cl)
        add_row(asset, 't分布(拟合)', 1, cl, v, e, f'拟合t自由度={dt_:.1f}')
    if asset == '贵州茅台':
        samples_1y = {'正态蒙特卡洛': chk, '历史模拟(重排)': sh, 'bootstrap重采样': bs, 't分布模拟': ts}
        t_fit_mt = (df_t, loc_t, sc_t)
        SPEED_REMARKS.append(f'茅台t分布自由度={df_t:.1f}（越小尾越厚；正态≈df→∞）')

print('  一年期 VaR/ES（茅台，正数=亏损幅度）:')
hd = f'  {"方法":<16s}{"VaR95":>9s}{"ES95":>9s}{"VaR99":>9s}{"ES99":>9s}'
print(hd)
for m in ['正态参数法(解析)', '正态蒙特卡洛(校验)', '历史模拟法(重排)', 'bootstrap重采样', 't分布蒙特卡洛']:
    v95, e95 = RES[('贵州茅台', m, DAYS, 0.95)]
    v99, e99 = RES[('贵州茅台', m, DAYS, 0.99)]
    print(f'  {m:<16s}{v95:>9.2%}{e95:>9.2%}{v99:>9.2%}{e99:>9.2%}')

# ---- 图02：日收益厚尾 + 一年期分布与 VaR ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.hist(lr_mt, bins=80, density=True, color='lightgray', edgecolor='darkgray', label='真实日对数收益(2015起)')
xs = np.linspace(-0.09, 0.09, 500)
ax1.plot(xs, stats.norm.pdf(xs, mu_mt, sd_mt), 'b-', lw=1.8, label='正态拟合(同μ,σ)')
df_t, loc_t, sc_t = t_fit_mt
ax1.plot(xs, stats.t.pdf(xs, df_t, loc=loc_t, scale=sc_t), 'r--', lw=1.8,
         label=f't分布拟合(自由度={df_t:.1f})')
wd, wv = lr_mt.idxmin(), lr_mt.min()
ax1.annotate(f'历史最差单日\n{wd.date()} {wv:.2%}', xy=(wv, 0.6), xytext=(wv - 0.032, 6.5),
             fontsize=9, color='darkred', arrowprops=dict(arrowstyle='->', color='darkred'))
ax1.set_xlim(-0.085, 0.085)
ax1.set_xlabel('日对数收益')
ax1.set_ylabel('密度')
ax1.set_title(f'图02左 茅台日收益：真实分布 vs 正态\n'
              f'超额峰度={ku_mt:.1f}(正态=0)；跌破正态99.9%分位真实{n999_real}天/预期{n999_exp:.1f}天')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.25)

styles = [('正态蒙特卡洛', 'tab:blue', '-'), ('历史模拟(重排)', 'tab:orange', '-'),
          ('bootstrap重采样', 'tab:green', '--'), ('t分布模拟', 'tab:red', '-')]
for name, color, ls in styles:
    simple_1y = np.exp(samples_1y[name]) - 1
    q99 = np.quantile(simple_1y, 0.01)
    ax2.hist(simple_1y, bins=80, density=True, histtype='step', lw=1.7, color=color, ls=ls,
             label=f'{name}（99%VaR={q99:.0%}）')
    ax2.axvline(q99, color=color, ls=':', lw=1.8)
ax2.axvline(0, color='gray', lw=0.8)
ax2.set_xlabel('一年期总收益率')
ax2.set_ylabel('密度')
ax2.set_title('图02右 一年期收益分布：三种方法对比（虚点线=99% VaR，数值见图例）\n'
              't分布法左移最多=尾部最厚；正态把最坏的1%年份画得太温和')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '02_三种方法收益率分布与VaR.png'), dpi=150)
plt.close(fig)


# ============================================================
# §4 组合（50%茅台 + 50%沪深300）：相关性对组合 VaR 的削弱
# ============================================================
print('\n' + '=' * 72)
print('§4 组合模拟（50/50，含相关性）')

# 分散化的量化：σ_p = √(w'Σw) < w1σ1 + w2σ2（当 ρ<1）
sig_p_cov = float(np.sqrt(W_MT ** 2 * sd_mt ** 2 + W_HS ** 2 * sd_hs ** 2 + 2 * W_MT * W_HS * rho * sd_mt * sd_hs))
div_daily = 1 - sig_p_cov / (W_MT * sd_mt + W_HS * sd_hs)
print(f'  相关性 ρ={rho:.3f} → 组合日σ={sig_p_cov:.4%}，比加权平均({W_MT * sd_mt + W_HS * sd_hs:.4%})低 {div_daily:.1%}')

# 二维相关模拟：对日对数收益协方差矩阵做 Cholesky 分解，把独立正态"掰弯"成相关正态
mu_vec = np.array([mu_mt, mu_hs])
Sigma = np.cov(np.vstack([lr_mt, lr_hs]), ddof=1)
L = np.linalg.cholesky(Sigma)
rng = np.random.default_rng(SEED)
Zc = rng.standard_normal((N_PATHS, DAYS, 2))
rets = mu_vec + Zc @ L.T                      # (10000,252,2) 相关的日对数收益
s_mt = np.exp(rets[:, :, 0]) - 1              # 茅台日简单收益
s_hs = np.exp(rets[:, :, 1]) - 1              # 沪深300日简单收益
rp_sim = W_MT * s_mt + W_HS * s_hs            # 组合日简单收益（权重作用于简单收益）
V = 100 * np.cumprod(1 + rp_sim, axis=1)      # 组合净值路径（期初=100）
V = np.hstack([np.full((N_PATHS, 1), 100.0), V])
p_loss_port = float(np.mean(V[:, -1] < 100))
for cl in (0.95, 0.99):
    v, e = var_es_from_logsum(np.log(V[:, -1] / 100), cl)
    add_row('组合(50%茅台+50%沪深300)', '正态蒙特卡洛(含相关性)', DAYS, cl, v, e, 'Cholesky引入ρ=%0.2f' % rho)

# 相关性削弱的"账面证据"：组合VaR vs 两资产VaR直接加权平均（同正态解析法，一年期）
v_mt99 = RES[('贵州茅台', '正态参数法(解析)', DAYS, 0.99)][0]
v_hs99 = RES[('沪深300', '正态参数法(解析)', DAYS, 0.99)][0]
v_p99 = RES[('组合(50%茅台+50%沪深300)', '正态蒙特卡洛(含相关性)', DAYS, 0.99)][0]
wavg99 = W_MT * v_mt99 + W_HS * v_hs99
print(f'  一年期VaR99: 组合模拟={v_p99:.2%} vs 两资产简单加权平均={wavg99:.2%} → 分散化省下 {1 - v_p99 / wavg99:.1%}')
v_mt95 = RES[('贵州茅台', '正态参数法(解析)', DAYS, 0.95)][0]
v_hs95 = RES[('沪深300', '正态参数法(解析)', DAYS, 0.95)][0]
v_p95 = RES[('组合(50%茅台+50%沪深300)', '正态蒙特卡洛(含相关性)', DAYS, 0.95)][0]
wavg95 = W_MT * v_mt95 + W_HS * v_hs95
print(f'  一年期VaR95: 组合模拟={v_p95:.2%} vs 加权平均={wavg95:.2%} → 省 {1 - v_p95 / wavg95:.1%}')

# 每条路径的最大回撤：running-max 同样向量化
runmax = np.maximum.accumulate(V, axis=1)
mdd_path = (1 - V / runmax).max(axis=1)


def hist_mdd_info(price, index):
    """真实序列的最大回撤，附峰/谷日期（用于核对回撤的真实性）。"""
    rm = np.maximum.accumulate(price)
    mdd = 1 - price / rm
    i = int(np.argmax(mdd))
    peak = int(np.argmax(price[:i + 1]))
    return float(mdd[i]), index[peak].date(), index[i].date()


w_price = (W_MT * df['mt'] + W_HS * df['hs'])                        # 真实组合市值（权重固定50/50）
mdd_real_p, pk_p, tr_p = hist_mdd_info(w_price.values, df.index)
mdd_real_mt, pk_mt, tr_mt = hist_mdd_info(df['mt'].values, df.index)
mdd_real_hs, pk_hs, tr_hs = hist_mdd_info(df['hs'].values, df.index)
print(f'  真实历史最大回撤: 组合={mdd_real_p:.1%}(峰{pk_p}→谷{tr_p}), '
      f'茅台={mdd_real_mt:.1%}(峰{pk_mt}→谷{tr_mt}), 沪深300={mdd_real_hs:.1%}(峰{pk_hs}→谷{tr_hs})')
q95_mdd = float(np.quantile(mdd_path, 0.95))
print(f'  模拟一年最大回撤: 中位={np.median(mdd_path):.1%}, 95%分位={q95_mdd:.1%}, 最深={mdd_path.max():.1%}')

fig, ax = plt.subplots(figsize=(10.5, 6))
ax.hist(mdd_path * 100, bins=60, color='steelblue', alpha=0.8, edgecolor='white')
ax.axvline(np.median(mdd_path) * 100, color='navy', lw=2, label=f'模拟中位数 {np.median(mdd_path):.1%}')
ax.axvline(q95_mdd * 100, color='crimson', ls='--', lw=2, label=f'模拟95%分位（坏年份） {q95_mdd:.1%}')
ax.axvline(mdd_real_p * 100, color='black', ls='-.', lw=2, label=f'真实历史最大回撤 {mdd_real_p:.1%}')
ax.set_xlabel('组合一年内最大回撤（%）')
ax.set_ylabel('路径数')
ax.set_title(f'图03 组合(50%茅台+50%沪深300)一年路径最大回撤分布：{N_PATHS}条模拟路径\n'
             f'ρ={rho:.2f}的相关性让回撤比"两只资产各自回撤的加权"更靠近加权值；真实历史的坑比模拟中位数更深')
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '03_组合回撤分布.png'), dpi=150)
plt.close(fig)


# ============================================================
# §5 压力测试三情景 —— VaR给概率，压力测试给想象
# ============================================================
print('\n' + '=' * 72)
print('§5 压力测试（组合口径，100万元本金）')

stress_rows = []
r1 = np.exp(lr_mt) - 1     # 茅台日简单收益
r2 = np.exp(lr_hs) - 1

# 情景一：-2σ日（合成情景：两资产同日各跌2个自身的日σ）
r1_s, r2_s = np.exp(-2 * sd_mt) - 1, np.exp(-2 * sd_hs) - 1
rp_s = W_MT * r1_s + W_HS * r2_s
loss1 = -rp_s
# 情景二：历史最大单日跌幅（2024-09-30式单日极端行情的反向版）
d2 = rp_hist.idxmin()
r1_d2, r2_d2 = float(r1.loc[d2]), float(r2.loc[d2])
rp_d2 = float(rp_hist.loc[d2])
loss2 = -rp_d2
# 情景三：连续20日历史最差窗口
roll20 = lp_hist.rolling(20).sum()
d3_end = roll20.idxmin()
d3_start = lp_hist.index[lp_hist.index.get_loc(d3_end) - 19]
port20 = float(np.expm1(roll20.loc[d3_end]))
mt20 = float(np.expm1(lr_mt.rolling(20).sum().loc[d3_end]))
hs20 = float(np.expm1(lr_hs.rolling(20).sum().loc[d3_end]))
loss3 = -port20

# 参照系：组合 bootstrap 一年期 VaR99 / 1天 VaR99
ref_1d = var_es_from_logsum(bootstrap_h(lp_hist, 1), 0.99)[0]
ref_250 = var_es_from_logsum(bootstrap_h(lp_hist, DAYS), 0.99)[0]
stress_items = [
    ('情景一：-2σ单日', '茅台、沪深300同日各跌2倍自身日σ（合成情景）', '合成', '合成',
     -r1_s, -r2_s, loss1, ref_1d),
    ('情景二：历史最大单日跌幅', '2024-09-30式单日极端行情的反向版（组合历史最差一天）', str(d2.date()), str(d2.date()),
     -r1_d2, -r2_d2, loss2, ref_1d),
    ('情景三：连续20日最差窗口', '组合历史上最差的连续20个交易日', str(d3_start.date()), str(d3_end.date()),
     -mt20, -hs20, loss3, ref_250),
]
print(f'  {"情景":<18s}{"茅台":>8s}{"沪深300":>9s}{"组合损失":>9s}{"100万元亏":>10s}{"参照VaR99":>10s}{"倍数":>6s}')
for tag, desc, d0, d1, leg_mt, leg_hs, loss, ref in stress_items:
    stress_rows.append({'情景': tag, '描述': desc, '起始日期': d0, '结束日期': d1,
                        '茅台跌幅': round(float(leg_mt), 6), '沪深300跌幅': round(float(leg_hs), 6),
                        '组合损失率': round(float(loss), 6),
                        '100万元组合亏损元': round(float(loss) * CAPITAL, 0),
                        '参照VaR(组合99%)': round(float(ref), 6),
                        '损失/参照VaR': round(float(loss) / float(ref), 2)})
    print(f'  {tag:<18s}{leg_mt:>8.2%}{leg_hs:>9.2%}{loss:>9.2%}{loss * CAPITAL:>10,.0f}{ref:>10.2%}'
          f'{loss / ref:>6.1f}x')
pd.DataFrame(stress_rows).to_csv(os.path.join(OUT, 'stress_tests.csv'), index=False, encoding='utf-8-sig')
print('  已写出 outputs/stress_tests.csv（VaR告诉你会超过的概率，压力测试告诉你超过时大概长什么样）')


# ============================================================
# §6 持有期 1/10/250 天与 √t 法则
# ============================================================
print('\n' + '=' * 72)
print('§6 持有期与 VaR：√t 法则及其失效（茅台 & 组合）')

HORIZONS = [1, 10, DAYS]
boot_var_plot = {0.95: {}, 0.99: {}}
real_var_plot = {0.95: {}, 0.99: {}}
naive_var_plot = {0.95: {}, 0.99: {}}

for asset, lr in [('贵州茅台', lr_mt), ('组合(50%茅台+50%沪深300)', lp_hist)]:
    m_d, s_d = lr.mean(), lr.std(ddof=1)
    bs1 = None
    for h in HORIZONS:
        # (a) 正态参数法：√t 法则精确成立
        for cl in (0.95, 0.99):
            v, e = normal_analytic_var_es(m_d, s_d, h, cl)
            add_row(asset, '正态参数法(解析)', h, cl, v, e, '√t法则精确成立')
        # (b) bootstrap（厚尾）：抽 h 天加总
        bs = bootstrap_h(lr, h)
        for cl in (0.95, 0.99):
            v, e = var_es_from_logsum(bs, cl)
            add_row(asset, 'bootstrap重采样', h, cl, v, e, '厚尾i.i.d.重采样')
            if asset == '贵州茅台':
                boot_var_plot[cl][h] = v
        # (c) 真实历史重叠窗口（厚尾+波动聚集）
        rw = real_window_h(lr, h)
        for cl in (0.95, 0.99):
            v, e = var_es_from_logsum(rw, cl)
            add_row(asset, '真实历史重叠窗口', h, cl, v, e, '真实滚动区间，含波动聚集')
            if asset == '贵州茅台':
                real_var_plot[cl][h] = v
    # (d) naive √t 外推：bootstrap 1天VaR × √h
    bs1 = bootstrap_h(lr, 1)
    for cl in (0.95, 0.99):
        v1, _ = var_es_from_logsum(bs1, cl)
        for h in HORIZONS:
            add_row(asset, '√t naive外推', h, cl, v1 * np.sqrt(h), np.nan, '1天bootstrap VaR×√h（仅VaR）')
            if asset == '贵州茅台':
                naive_var_plot[cl][h] = v1 * np.sqrt(h)

print('  贵州茅台 VaR99 随持有期（第三个期限按252个交易日≈"250天/一年"计）:')
print(f'  {"方法":<18s}{"1天":>9s}{"10天":>9s}{"252天(1年)":>11s}{"10天/√10外推":>13s}{"252天/√252外推":>15s}')
for m, store in [('bootstrap重采样', boot_var_plot[0.99]), ('真实历史重叠窗口', real_var_plot[0.99]),
                 ('√t naive外推', naive_var_plot[0.99])]:
    vals = [store[1], store[10], store[DAYS]]
    r10 = vals[1] / naive_var_plot[0.99][10]
    r250 = vals[2] / naive_var_plot[0.99][DAYS]
    print(f'  {m:<18s}{vals[0]:>9.2%}{vals[1]:>9.2%}{vals[2]:>10.2%}{r10:>12.2f}x{r250:>14.2f}x')

# ---- 图04：VaR 随持有期的增长 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True)
hs_grid = np.arange(1, DAYS + 1)
for ax, cl, ttl in zip(axes, (0.95, 0.99), ('95%', '99%')):
    norm_curve = np.array([normal_analytic_var_es(mu_mt, sd_mt, h, cl)[0] for h in hs_grid])
    naive_curve = np.array([naive_var_plot[cl][1] * np.sqrt(h) for h in hs_grid])
    ax.plot(hs_grid, norm_curve * 100, 'b-', lw=2, label='正态参数法（√t精确成立）')
    ax.plot(hs_grid, naive_curve * 100, 'g--', lw=1.6, label='√t naive外推（1天bootstrap VaR×√h）')
    bh = [boot_var_plot[cl][h] for h in HORIZONS]
    rh = [real_var_plot[cl][h] for h in HORIZONS]
    ax.plot(HORIZONS, np.array(bh) * 100, 'ro', ms=8, label='bootstrap模拟（厚尾）')
    ax.plot(HORIZONS, np.array(rh) * 100, 'k^', ms=8, label='真实历史重叠窗口')
    ax.set_xscale('log')
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels(['1天', '10天', '1年(252天)'])
    ax.set_xlabel('持有期')
    ax.set_ylabel(f'VaR{ttl}（亏损%，简单收益口径）')
    ax.set_title(f'图04{["左", "右"][1 if cl == 0.99 else 0]} 茅台 VaR{ttl} 随持有期的变化')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which='both')
fig.tight_layout()
fig.savefig(os.path.join(OUT, '04_持有期与VaR关系.png'), dpi=150)
plt.close(fig)


# ============================================================
# §7 汇总：var_results.csv + README 数字速查
# ============================================================
pd.DataFrame(ROWS).to_csv(os.path.join(OUT, 'var_results.csv'), index=False, encoding='utf-8-sig')
print('\n' + '=' * 72)
print('§7 已写出 outputs/var_results.csv（%d 行）与全部图表' % len(ROWS))

print('\n' + '#' * 72)
print('# README 数字速查（全部来自本次实际运行）')
print('#' * 72)
mt99 = {m: RES[('贵州茅台', m, DAYS, 0.99)][0] for m in
        ['正态参数法(解析)', '历史模拟法(重排)', 'bootstrap重采样', 't分布蒙特卡洛']}
base99 = mt99['正态参数法(解析)']
print(f'茅台一年期VaR99对比: ' + ' | '.join(f'{m}={v:.1%}' for m, v in mt99.items()))
print(f'  → 历史法/正态 = {mt99["历史模拟法(重排)"] / base99:.2f}倍, bootstrap/正态 = {mt99["bootstrap重采样"] / base99:.2f}倍, '
      f't分布/正态 = {mt99["t分布蒙特卡洛"] / base99:.2f}倍')
daily99 = {m: RES[('贵州茅台', m, 1, 0.99)][0] for m in ['正态参数法(解析)', '历史经验分位数', 't分布(拟合)']}
d99_base = daily99['正态参数法(解析)']
print('茅台1天VaR99对比: ' + ' | '.join(f'{m}={v:.2%}' for m, v in daily99.items()) +
      f'  → 经验/正态={daily99["历史经验分位数"] / d99_base:.2f}倍, t/正态={daily99["t分布(拟合)"] / d99_base:.2f}倍')
print(f'{SPEED_REMARKS[0]}')
mt95 = {m: RES[('贵州茅台', m, DAYS, 0.95)][0] for m in ['正态参数法(解析)', '历史模拟法(重排)', 't分布蒙特卡洛']}
print(f'茅台一年期VaR95对比: ' + ' | '.join(f'{m}={v:.1%}' for m, v in mt95.items()))
es99_n = RES[('贵州茅台', '正态参数法(解析)', DAYS, 0.99)][1]
es99_t = RES[('贵州茅台', 't分布蒙特卡洛', DAYS, 0.99)][1]
print(f'茅台一年期ES99: 正态={es99_n:.1%} vs t分布={es99_t:.1%}（ES永远≥VaR，厚尾下差距更大）')
print(f'组合一年期VaR99: 正态含相关MC={v_p99:.1%}, 加权平均={wavg99:.1%}, 分散化节省={1 - v_p99 / wavg99:.1%}; ρ={rho:.2f}')
print(f'组合一年期VaR95/99与ES（bootstrap口径）: '
      f'VaR95={RES[("组合(50%茅台+50%沪深300)", "bootstrap重采样", DAYS, 0.95)][0]:.1%}, '
      f'VaR99={RES[("组合(50%茅台+50%沪深300)", "bootstrap重采样", DAYS, 0.99)][0]:.1%}, '
      f'ES99={RES[("组合(50%茅台+50%沪深300)", "bootstrap重采样", DAYS, 0.99)][1]:.1%}')
print(f'组合模拟一年亏钱概率={p_loss_port:.1%}; 模拟回撤中位={np.median(mdd_path):.1%}, '
      f'95%分位={q95_mdd:.1%}, 真实历史MDD={mdd_real_p:.1%}')
worst_stress = max(stress_rows, key=lambda r: r['组合损失率'])
print(f'压力测试最痛情景: {worst_stress["情景"]}，组合亏 {worst_stress["组合损失率"]:.1%}'
      f'（100万元亏 {worst_stress["100万元组合亏损元"]:,.0f} 元），'
      f'={worst_stress["损失/参照VaR"]:.1f}倍参照VaR99')
print(f'√t检验(茅台VaR99, 实际/naive√t外推): 10天={boot_var_plot[0.99][10] / naive_var_plot[0.99][10]:.2f}x, '
      f'252天={boot_var_plot[0.99][DAYS] / naive_var_plot[0.99][DAYS]:.2f}x; '
      f'真实窗口/正态参数: 10天={real_var_plot[0.99][10] / RES[("贵州茅台", "正态参数法(解析)", 10, 0.99)][0]:.2f}x, '
      f'252天={real_var_plot[0.99][DAYS] / RES[("贵州茅台", "正态参数法(解析)", DAYS, 0.99)][0]:.2f}x')
print(f'厚尾证据: 超额峰度={ku_mt:.1f}, JB p={jb_p:.1e}, 跌破正态99.9%分位 {n999_real}天(预期{n999_exp:.1f}天), '
      f'最差单日 {wv:.2%}({wd.date()})')
print(f'GBM要点: 中位终值={np.median(term):.0f}元 vs 均值终值={term.mean():.0f}元(均值>中位, 对数正态右偏); '
      f'一年亏钱概率≈{p_loss:.0%}')
print(f'运行耗时 {time.time() - T0:.1f} 秒，退出码 0。')
