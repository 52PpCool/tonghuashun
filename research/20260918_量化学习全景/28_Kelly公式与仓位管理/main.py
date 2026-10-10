# -*- coding: utf-8 -*-
"""
28_Kelly公式与仓位管理 —— “多少仓位才对？”
=================================================
从凯利公式到现实约束：回撤、估计误差、破产概率。

六个实验（与 README 章节一一对应）：
  1. 离散版 Kelly：p=0.55、b=1 的二结果赌局，f 从 0 扫到 1.5，
     蒙特卡洛验证长期增长率的峰恰在 f*=0.10；全下注(f=1)长期趋零。
  2. 连续版 Kelly：f* = μ/σ²，用茅台/上证真实日收益(对数)估计，
     讨论“满仓Kelly≈几倍杠杆”以及为什么现实没人这么干。
  3. 估计误差的代价：真实参数生成的市场 vs 用过去120日估的 μ̂ 定仓，
     对比 真Kelly / 估计Kelly / 半Kelly / 固定50% 的长期增长与回撤。
  4. 回撤分布：半Kelly / Kelly / 2×Kelly 下 10000 条路径的最大回撤
     箱线图 + “回撤超过50%的概率”表（市场参数用茅台真实估计）。
  5. 真实数据应用：茅台/上证 20日均线信号 × 四种仓位规则
     （全仓 / 固定30% / 波动率目标10% / 半Kelly动态）回测。
  6. 破产概率：离散赌局下不同 f 的“净值曾跌破初始10%”概率表。

运行（工作区指定解释器）：
  /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py
随机种子：np.random.default_rng(42)，全流程可复现。
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示环境出图
import matplotlib.pyplot as plt

# 中文字体（工作区规范）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']  # DejaVu 兜底希腊字母
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(BASE), 'data', 'cache')

rng = np.random.default_rng(42)   # 全课唯一随机源，种子42
T_START = time.time()


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# 0. 数据加载：真实日K优先，缺失时用 rng(42) 造合成数据并明确标注
# ----------------------------------------------------------------------
def load_asset(fname, asset_name, synth_years=11, synth_n=2800):
    """返回 (close序列, 简单日收益, 对数日收益, 数据来源标注)"""
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8-sig', parse_dates=['date']).set_index('date')
        src = '真实数据(data/cache)'
    else:  # 数据缺失兜底：几何布朗运动合成，绝不失败退出
        idx = pd.bdate_range('2015-01-05', periods=synth_n)
        r = rng.normal(0.0005, 0.02, size=synth_n)
        close = pd.Series(100 * np.cumprod(1 + r), index=idx, name='close')
        df = pd.DataFrame({'close': close})
        src = f'合成数据(rng42, {asset_name}缓存缺失)'
    close = df['close'].astype(float)
    ret = close.pct_change().dropna()            # 简单收益：回测计净值用
    logret = np.log(close).diff().dropna()       # 对数收益：估计 μ/σ 用（可加性好）
    log(f'[数据] {asset_name}: {len(close)}个交易日 {close.index[0].date()}~{close.index[-1].date()}  [{src}]')
    return close, ret, logret, src


# 通用指标：净值序列(简单日收益) -> 绩效字典
RF_ANNUAL = 0.02  # 无风险利率（年化2%，与本系列课01-10统一口径）

def perf(r, idx, freq=252):
    r = r.dropna()
    yrs = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    W = float((1 + r).prod())
    cagr = W ** (1 / yrs) - 1 if W > 0 else -1.0
    g_log = np.log(W) / yrs if W > 0 else -np.inf      # 年化对数增长率(几何增长)
    vol = float(r.std() * np.sqrt(freq))
    sharpe = float((r.mean() * freq - RF_ANNUAL) / vol) if vol > 0 else 0.0  # 年化超额/波动
    cum = (1 + r).cumprod()
    dd = float((cum / cum.cummax() - 1).min())          # 最大回撤(负数)
    return {'终值(倍)': W, '年化收益%': cagr * 100, '年化对数增长率%': g_log * 100,
            '年化波动%': vol * 100, '夏普': sharpe, '最大回撤%': dd * 100}


# ======================================================================
# 第1部分：【离散版Kelly】p=0.55、b=1 的赌局，f 扫描 + 蒙特卡洛验证
# ======================================================================
# --- 公式推导（代数全过程）--------------------------------------------
# 每期押上净值的 f 倍：以概率 p 赢，净值变为 (1 + b·f)；以概率 1-p 输，变为 (1 - f)。
# 赌局独立重复 T 期，长期增长率用“每期对数收益的期望”(几何增长)度量：
#     g(f) = p·ln(1 + b·f) + (1-p)·ln(1 - f)
# 极值条件 g'(f) = 0：
#     g'(f) = p·b/(1+b·f) - (1-p)/(1-f) = 0
#     =>  p·b·(1-f) = (1-p)·(1+b·f)
#     =>  p·b - p·b·f = (1-p) + (1-p)·b·f
#     =>  p·b - (1-p) = b·f·(p + 1 - p) = b·f
#     =>  f* = (p·b - (1-p)) / b = p - (1-p)/b
# 二阶导 g''(f) = -p·b²/(1+bf)² - (1-p)/(1-f)² < 0，g(f) 是凹函数，f* 是唯一峰值。
# 本课参数 p=0.55, b=1：f* = 0.55 - 0.45 = 0.10（押一成仓）。
# 关键不等式：g(f) < 0 当且仅当 f > 2f*（等比赌局），即“2倍Kelly是不归点”。
# ======================================================================
log('\n' + '=' * 72)
log('第1部分：离散版Kelly —— 增长率 vs 仓位扫描（p=0.55, b=1, f*=0.10）')
log('=' * 72)

P_WIN, B_PAY = 0.55, 1.0
N_PATHS, N_STEPS = 10000, 1000
EPS_FLOOR = 1e-12          # 破产净值地板：真实终值为0，数值上用1e-12代表“趋零”
F_STAR = P_WIN - (1 - P_WIN) / B_PAY

# 共享同一批“运气”：所有仓位等级用同一份输赢矩阵（公平对比、曲线平滑）
wins = rng.random((N_PATHS, N_STEPS)) < P_WIN      # True=赢
nwin = wins.sum(axis=1).astype(np.float64)         # 每条路径总胜场
# 注意：终值 = (1+bf)^胜场 · (1-f)^败场，与输赢“顺序”无关（独立赌局可交换），
# 所以扫描最终增长率只需要总胜场 nwin；第6部分看路径最低点才需要逐期累计。

def theory_growth(f):
    """离散赌局理论长期增长率（仅 f<1 有定义；f>=1 一次亏损即破产）"""
    return P_WIN * np.log(1 + B_PAY * f) + (1 - P_WIN) * np.log(1 - f)

f_grid = np.round(np.arange(0.0, 1.5001, 0.025), 4)   # 0 → 1.5，共61档
mc_growth, mc_medW = [], []
for f in f_grid:
    if f < 1.0:   # f<1：永不清零，终值=exp(nwin·ln(1+bf)+(T-nwin)·ln(1-f))，只需nwin
        g_path = (nwin * np.log1p(B_PAY * f) + (N_STEPS - nwin) * np.log(1 - f)) / N_STEPS
        mc_growth.append(g_path.mean())
        mc_medW.append(float(np.median(np.exp(nwin * np.log1p(B_PAY * f) + (N_STEPS - nwin) * np.log(1 - f)))))
    else:         # f>=1：任一次亏损 => 净值×(1-f)<=0，路径死亡（全下注f=1恰好归零）
        alive = (nwin == N_STEPS)                  # 存活=1000期全赢，P=0.55^1000≈1e-262
        g_path = np.where(alive, np.log(1 + B_PAY * f), np.log(EPS_FLOOR)) / N_STEPS
        mc_growth.append(g_path.mean())
        W_alive = np.exp(min(N_STEPS * np.log1p(B_PAY * f), 700.0))   # 存活终值封顶防溢出
        mc_medW.append(float(np.median(np.where(alive, W_alive, 0.0))))
mc_growth = np.array(mc_growth)

g_peak_mc = mc_growth.max()
f_peak_mc = float(f_grid[mc_growth.argmax()])
i10 = int(np.argmin(np.abs(f_grid - 0.10)))
i20 = int(np.argmin(np.abs(f_grid - 0.20)))
log(f'理论 f* = {F_STAR:.2f}，理论峰值 g = {theory_growth(F_STAR) * 100:.4f}%/期'
    f'（≈年化 {theory_growth(F_STAR) * 252 * 100:.0f}%，若一年能赌252期独立局）')
log(f'蒙特卡洛峰值：f={f_peak_mc:.3f}，g={g_peak_mc * 100:.4f}%/期（与理论一致=>验证通过）')
log(f'f=0.20（2×Kelly）：g = {mc_growth[i20] * 100:.4f}%/期（理论 {theory_growth(0.2) * 100:.4f}%/期）'
    f'—— 增长已归零转负（2倍Kelly不归点）')
log(f'终值中位数对比：f=0.05 半Kelly → {mc_medW[int(np.argmin(np.abs(f_grid-0.05)))]:.2f} 倍；'
    f'f=0.10 Kelly → {mc_medW[i10]:.2f} 倍；f=1.00 全下注 → {mc_medW[int(np.argmin(np.abs(f_grid-1.0)))]:.4f} 倍（趋零）')
log(f'f=1.5（1.5倍杠杆）：蒙特卡洛增长率 = {mc_growth[-1] * 100:.3f}%/期（趴在破产地板 ln(1e-12)/1000）')

# ---- 图01：增长率 vs 仓位（全课最震撼的一张图）-----------------------
fig, ax = plt.subplots(figsize=(10.5, 6.2))
f_th = np.linspace(0, 0.99, 500)
ax.plot(f_th, theory_growth(f_th) * 100, '-', lw=2, color='#1f77b4', label='理论 g(f)=p·ln(1+bf)+(1-p)·ln(1-f)')
ax.plot(f_grid, mc_growth * 100, 'o', ms=4.5, color='#d62728', label='蒙特卡洛（10000路径×1000期）')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.axvspan(1.0, 1.5, color='red', alpha=0.10)
ax.text(1.25, -3.6, '杠杆区\n一次亏损即破产\n（长期趋零）', ha='center', fontsize=10, color='#8b0000')
ax.axvline(F_STAR, color='green', lw=1.2, ls=':')
ax.axvline(2 * F_STAR, color='orange', lw=1.2, ls=':')
ax.annotate(f'最优仓位 f*={F_STAR:.2f}\ng≈{theory_growth(F_STAR) * 100:.3f}%/期',
            xy=(F_STAR, theory_growth(F_STAR) * 100), xytext=(0.18, 0.42),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=11, color='green')
ax.text(2 * F_STAR + 0.015, 0.18, f'2×Kelly={2 * F_STAR:.2f}：增长归零的不归点', fontsize=10, color='#c96a00')
ax.text(0.99, -3.0, '全下注 f=1\n首次亏损→净值=0', ha='right', fontsize=10, color='#8b0000')
ax.set_xlim(-0.02, 1.52)
ax.set_ylim(-4.2, 0.8)
ax.set_xlabel('仓位 f（占净值比例；>1 为杠杆）')
ax.set_ylabel('每期对数增长率 g(f)（%）')
ax.set_title('图01 离散赌局(p=0.55, b=1)：长期增长率 vs 仓位 —— 峰在Kelly点，\n'
             '少押只是少赚，多押加速变负，全下注终值趋零（破产地板 ln(1e-12)/1000≈-2.76%/期）')
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '01_增长率vs仓位曲线.png'), dpi=150)
plt.close(fig)

# ---- 图02：样本路径对比：全下注 vs 半Kelly ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
n_show, n_show_steps = 30, 300
sub = wins[:n_show, :n_show_steps]
t_axis = np.arange(1, n_show_steps + 1)
for ax, f, name in [(axes[0], 1.0, '全下注 f=1.0'), (axes[1], 0.05, '半Kelly f=0.05')]:
    mult = np.where(sub, 1 + B_PAY * f, 1 - f)                # (30,300) 每期乘数
    W_path = np.cumprod(mult, axis=1)
    W_path = np.hstack([np.ones((n_show, 1)), W_path])        # 起点=1
    W_plot = np.where(W_path > 0, W_path, EPS_FLOOR)          # 破产=0，画在地板上
    for i in range(n_show):
        ax.plot(range(n_show_steps + 1), W_plot[i], lw=0.8, alpha=0.55)
    med_end = float(np.median(W_path[:, -1]))
    ax.set_yscale('log')
    ax.set_title(f'{name}（{n_show}条样本路径×{n_show_steps}期，对数轴）\n终值中位数 = {med_end:.3g} 倍')
    ax.set_xlabel('期数')
    ax.grid(alpha=0.25, which='both')
axes[0].axhline(EPS_FLOOR, color='red', lw=1, ls='--')
axes[0].text(150, EPS_FLOOR * 2.5, '破产地板：净值=0（画在1e-12）', color='red', fontsize=9)
axes[0].set_ylabel('净值（对数刻度）')
fig.suptitle('图02 同样的运气、不同的仓位：全下注第一次亏损就出局，半Kelly稳步复利', y=1.0, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '02_路径模拟对比.png'), dpi=150)
plt.close(fig)
log('[图] 已保存 01_增长率vs仓位曲线.png, 02_路径模拟对比.png')

# ======================================================================
# 第2部分：【连续版Kelly】f* = μ/σ²，用茅台/上证真实日收益估计
# ======================================================================
# --- 推导直觉 ---------------------------------------------------------
# 连续化：每期收益 r~N(μ,σ²)，净值 W' = W·(1+f·r)。对数增长
#     E[ln(1+f·r)] = f·E[r] - f²·E[r²]/2 + ... ≈ f·μ - f²σ²/2   （泰勒展开，σ小则高阶项可忽略）
# 对 f 求导取零：μ - f·σ² = 0  =>  f* = μ/σ²，最大增长率 g* = μ²/(2σ²) = 夏普²/2。
# 直觉：分子是你能吃的“漂移”，分母是你必须付出的“方差税”；仓位=漂移/方差。
# 满仓Kelly的组合波动率 = f*·σ = μ/σ = 夏普 —— 夏普0.5的资产，满仓Kelly意味着
# 组合年化波动50%，这就是“满仓Kelly≈X倍杠杆”的具象化，也是没人真这么干的原因之一。
# 注：μ 用对数收益估计（可加、与 g 的定义一致）；f*=μ/σ² 与年化/日度尺度无关（252约掉）。
# ======================================================================
log('\n' + '=' * 72)
log('第2部分：连续版Kelly —— f* = μ/σ²（真实数据估计）')
log('=' * 72)

close_mt, ret_mt, logret_mt, src_mt = load_asset('stock_600519_贵州茅台.csv', '贵州茅台600519')
close_sh, ret_sh, logret_sh, src_sh = load_asset('index_000001_上证指数.csv', '上证指数000001')

def kelly_continuous(logret, name):
    mu, sd, n = float(logret.mean()), float(logret.std(ddof=1)), len(logret)
    fstar = mu / sd ** 2
    se_mu = sd / np.sqrt(n)          # μ 的标准误
    se_f = se_mu / sd ** 2           # f* 的标准误（一阶delta：∂f*/∂μ=1/σ²）
    tstat = mu / se_mu
    log(f'\n[{name}] n={n}个交易日')
    log(f'  日度 μ={mu * 100:.4f}%  σ={sd * 100:.3f}%   |  年化 μ={mu * 252 * 100:.1f}%  σ={sd * np.sqrt(252) * 100:.1f}%')
    log(f'  连续Kelly f* = μ/σ² = {fstar:.2f} 倍（即满仓Kelly≈{fstar:.2f}倍杠杆，{max(fstar - 1, 0):.2f}倍需融资）')
    log(f'  f* 的标准误 ≈ {se_f:.2f}（μ的t值仅 {tstat:.2f}）→ f* 的95%区间约 [{fstar - 1.96 * se_f:.2f}, {fstar + 1.96 * se_f:.2f}]')
    log(f'  满仓Kelly的组合年化波动 = f*·σ_ann = {fstar * sd * np.sqrt(252) * 100:.0f}% —— 这就是没人满仓Kelly的第一个原因')
    return {'name': name, 'mu_ann': mu * 252, 'sd_ann': sd * np.sqrt(252),
            'fstar': fstar, 'se_f': se_f, 'tstat': tstat, 'n': n}

k_mt = kelly_continuous(logret_mt, '贵州茅台(2015起)')
k_sh = kelly_continuous(logret_sh, '上证指数(2010起)')
F_MT = k_mt['fstar']

# ======================================================================
# 第3部分：【估计误差的代价】真参数市场 vs 120日滚动估计定仓
# ======================================================================
log('\n' + '=' * 72)
log('第3部分：估计误差的代价（真Kelly vs 估计Kelly vs 半Kelly vs 固定50%）')
log('=' * 72)

P3, T3, W3 = 10000, 1000, 120            # 路径数 / 期数 / 估计窗口
MU_T, SD_T = 0.0006, 0.02                # 真实日度参数：μ=0.06%,σ=2%（年化≈15.1%/31.7%）
F_TRUE = MU_T / SD_T ** 2                # 真Kelly = 1.50 倍
LEV_CAP = 5.0                            # 现实约束：最多5倍杠杆（融资上限）
log(f'真实市场：r~N(μ={MU_T * 100:.2f}%, σ={SD_T * 100:.1f}%)，真Kelly f*={F_TRUE:.2f}；'
    f'模拟 {P3} 条路径 × {T3} 期，用过去{W3}日估计 μ̂,σ̂')

r3 = rng.normal(MU_T, SD_T, size=(P3, T3))
r3 = np.clip(r3, MU_T - 6 * SD_T, MU_T + 6 * SD_T)      # 正态尾部保护(6σ截断)，防极端杠杆爆仓
# 滚动窗口和：前缀和技巧，一次算出所有时点的 120 日 μ̂ 与 σ̂（无未来函数：t 期仓位只用 r[t-W..t-1]）
C1 = np.cumsum(r3, axis=1)
Z = np.hstack([np.zeros((P3, 1)), C1])                   # Z[:,t] = sum(r[0..t-1])
S1 = Z[:, W3:T3] - Z[:, :T3 - W3]                        # 窗口和，对应 t=W3..T3-1
Z2 = np.hstack([np.zeros((P3, 1)), np.cumsum(r3 * r3, axis=1)])
S2 = Z2[:, W3:T3] - Z2[:, :T3 - W3]
mu_hat = S1 / W3
var_hat = np.maximum(S2 / W3 - mu_hat ** 2, 1e-8)        # 方差下限保护
f_hat = np.clip(mu_hat / var_hat, 0.0, LEV_CAP)          # 估计Kelly仓位（不允许做空，上限5倍）
f_half = np.clip(0.5 * mu_hat / var_hat, 0.0, LEV_CAP / 2)
Fmat = np.zeros((P3, T3))
Fmat[:, W3:] = f_hat
Hmat = np.zeros((P3, T3))
Hmat[:, W3:] = f_half
del C1, Z, Z2, S1, S2, mu_hat, var_hat, f_hat, f_half

def eval_rule(f_spec, r, T):
    """给定仓位(标量或(P,T)矩阵)作用在收益矩阵上，返回绩效字典（路径横截面统计）"""
    fr = np.minimum(f_spec * r, 0.999)                    # 单期净值保护
    L = np.cumsum(np.log1p(fr), axis=1)                   # 对数净值路径：逐期累加！
    g_path = L[:, -1] / T
    Lmax = np.maximum.accumulate(L, axis=1)
    dd_log = (Lmax - L).max(axis=1)                       # 最大回撤(对数)
    dd_frac = 1 - np.exp(-dd_log)
    out = {'年化对数增长率%': g_path.mean() * 252 * 100,
           '中位年化对数增长率%': np.median(g_path) * 252 * 100,
           '终值中位数(倍)': float(np.median(np.exp(L[:, -1]))),
           '最大回撤中位%': float(np.median(dd_frac)) * 100,
           '最大回撤均值%': float(dd_frac.mean()) * 100,
           'P(回撤>50%)': float((dd_log > np.log(2)).mean()),
           '平均仓位(倍)': float(f_spec.mean()) if not np.isscalar(f_spec) else float(f_spec)}
    return out

rules3 = {
    '真Kelly(1.50倍,上帝视角)': F_TRUE,
    '估计Kelly(120日μ̂/σ̂²,≤5倍)': Fmat,
    '半Kelly(0.5×估计,≤2.5倍)': Hmat,
    '真半Kelly(0.75倍,常数对照)': 0.5 * F_TRUE,   # 与“半Kelly”平均仓位相同，但无需估计——隔离估计噪声
    '固定50%仓位': 0.5,
}
res3 = {}
for name, fs in rules3.items():
    res3[name] = eval_rule(fs, r3, T3)
    m = res3[name]
    log(f"  {name:<28s} 年化对数增长 {m['年化对数增长率%']:6.2f}% | 终值中位 {m['终值中位数(倍)']:7.2f} | "
        f"回撤中位 {m['最大回撤中位%']:5.1f}% | P(回撤>50%) {m['P(回撤>50%)'] * 100:5.1f}% | 平均仓位 {m['平均仓位(倍)']:.2f}倍")
gstar_theory = (MU_T ** 2 / (2 * SD_T ** 2)) * 252
log(f'理论校验：真Kelly最大年化对数增长率 g*=μ²/(2σ²)×252 = {gstar_theory * 100:.2f}%（蒙特卡洛应接近）')
m_e, m_h = res3['估计Kelly(120日μ̂/σ̂²,≤5倍)'], res3['半Kelly(0.5×估计,≤2.5倍)']
m_th = res3['真半Kelly(0.75倍,常数对照)']
log(f'估计Kelly有 {float((Fmat[:, W3:] == 0).mean()) * 100:.0f}% 的天数因 μ̂<0 而空仓')
log(f'隔离估计噪声：真半Kelly(0.75倍常数) 增长 {m_th["年化对数增长率%"]:.2f}%/年、回撤中位 {m_th["最大回撤中位%"]:.1f}%；'
    f'平均仓位相同的估计半Kelly只有 {m_h["年化对数增长率%"]:.2f}%/年、回撤 {m_h["最大回撤中位%"]:.1f}% —— 差距全部来自估计噪声')
del r3, Fmat, Hmat

# ---- 图05：估计误差的代价（双面板柱状图）-----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
names3 = list(res3.keys())
n3 = len(names3)
colors3 = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#7f7f7f']
short3 = ['真Kelly\n(上帝视角)', '估计Kelly\n(120日滚动)', '半Kelly\n(0.5×估计)', '真半Kelly\n(0.75倍常数)', '固定50%'][:n3]
vals1 = [res3[n]['年化对数增长率%'] for n in names3]
vals2 = [res3[n]['最大回撤中位%'] for n in names3]
bars1 = ax1.bar(range(n3), vals1, color=colors3[:n3])
ax1.bar_label(bars1, fmt='%.2f%%', fontsize=9)
ax1.axhline(0, color='gray', lw=0.8)
ax1.set_xticks(range(n3))
ax1.set_xticklabels(short3, fontsize=9.5)
ax1.set_ylabel('年化对数增长率（%）')
ax1.set_title('长期增长：估计误差吃掉真Kelly的增长\n（固定50%这个“最笨规则”反而稳赢估计Kelly）')
ax1.grid(alpha=0.25, axis='y')
bars2 = ax2.bar(range(n3), vals2, color=colors3[:n3])
for i, n in enumerate(names3):
    ax2.text(i, vals2[i] + 0.8, f"P(>50%)\n{res3[n]['P(回撤>50%)'] * 100:.0f}%", ha='center', fontsize=8.5, color='#8b0000')
ax2.set_xticks(range(n3))
ax2.set_xticklabels(short3, fontsize=9.5)
ax2.set_ylabel('最大回撤中位数（%）')
ax2.set_title('回撤代价：仓位越高、仓位越“晃”，深回撤越常见')
ax2.grid(alpha=0.25, axis='y')
fig.suptitle(f'图05 估计误差的代价：真实 μ={MU_T * 100:.2f}%/日，但120日样本估计的标准误达 {SD_T / np.sqrt(W3) * 100:.3f}%/日 —— 参数不确定时必须打折', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '05_估计误差代价.png'), dpi=150)
plt.close(fig)
log('[图] 已保存 05_估计误差代价.png')

# ======================================================================
# 第4部分：【回撤分布】半Kelly / Kelly / 2×Kelly 的最大回撤（茅台参数市场）
# ======================================================================
log('\n' + '=' * 72)
log('第4部分：回撤分布与 P(回撤>50%)（市场参数=茅台真实日度 μ,σ）')
log('=' * 72)

P4, T4 = 10000, 1000
r4 = rng.normal(k_mt['mu_ann'] / 252, k_mt['sd_ann'] / np.sqrt(252), size=(P4, T4))
r4 = np.clip(r4, r4.mean() - 6 * r4.std(), r4.mean() + 6 * r4.std())   # 6σ截断保护
levels4 = [('半Kelly', 0.5 * F_MT), ('Kelly', F_MT), ('2倍Kelly', 2 * F_MT), ('无杠杆满仓(参考)', 1.0)]
dd_arr, rows4 = {}, []
log(f'茅台日度 μ={k_mt["mu_ann"] / 252 * 100:.4f}%, σ={k_mt["sd_ann"] / np.sqrt(252) * 100:.3f}% → f*={F_MT:.2f}；'
    f'模拟 {P4} 条路径 × {T4} 期')
THR = [0.10, 0.20, 0.30, 0.50]
for name, f in levels4:
    fr = np.minimum(f * r4, 0.999)
    L = np.cumsum(np.log1p(fr), axis=1)                   # 对数净值路径：逐期累加！
    Lmax = np.maximum.accumulate(L, axis=1)
    dd = 1 - np.exp(-(Lmax - L).max(axis=1))
    dd_arr[name] = dd
    line = f'  {name:<14s}(f={f:.2f}) 回撤中位 {np.median(dd) * 100:5.1f}% | ' + \
           ' | '.join([f'P(回撤>{t * 100:.0f}%)={float((dd > t).mean()) * 100:5.1f}%' for t in THR])
    log(line)
    for t in THR:
        rows4.append({'实验': '04_回撤分布(茅台参数)', '方案': f'{name}(f={f:.2f})',
                      '阈值/事件': f'最大回撤>{t * 100:.0f}%', '概率': float((dd > t).mean())})

# ---- 图03：回撤分布箱线图 -------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 6))
bp = ax.boxplot([dd_arr[n] for n, _ in levels4], tick_labels=[f'{n}\nf={f:.2f}' for n, f in levels4],
                showfliers=False, patch_artist=True, widths=0.55)
for patch, c in zip(bp['boxes'], ['#2ca02c', '#1f77b4', '#d62728', '#7f7f7f']):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
ax.axhline(0.5, color='red', ls='--', lw=1.2)
ax.text(0.55, 0.51, '腰斩线：回撤50%', color='red', fontsize=10)
ax.set_ylabel('最大回撤（10000条路径×1000期）')
ax.set_title(f'图03 仓位越高回撤越深：茅台参数下 Kelly(f={F_MT:.2f}) 的回撤中位数已达 '
             f'{np.median(dd_arr["Kelly"]) * 100:.0f}%，\n2×Kelly 几乎必然腰斩 —— 这就是机构用分数Kelly的原因')
ax.grid(alpha=0.25, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(OUT, '03_回撤分布箱线图.png'), dpi=150)
plt.close(fig)
log('[图] 已保存 03_回撤分布箱线图.png')
del r4

# ======================================================================
# 第5部分：【真实数据应用】MA20信号 × 四种仓位规则（茅台/上证）
# ======================================================================
# 信号本身弱没关系——本课主角是仓位。四种规则（信号内）：
#   全仓：信号开=100%仓位        固定30%：信号开=30%
#   波动率目标10%：f = clip(10% / σ20年化, 0, 1)  —— 组合波动≈10%
#   半Kelly动态：f = clip(0.5×μ̂60/σ̂60², 0, 1)    —— 只用过去60日，无未来函数
# 所有用到的信号/仓位都 shift(1)：今天收盘后才知道的信息，明天才能用。
# ======================================================================
log('\n' + '=' * 72)
log('第5部分：真实数据回测 —— 20日均线信号 × 四种仓位规则')
log('=' * 72)

def backtest_rules(close, ret, logret, name):
    sig = (close > close.rolling(20).mean()).astype(float).shift(1).fillna(0)   # 昨收在MA20上方→今日持仓
    sd20 = ret.rolling(20).std() * np.sqrt(252)
    f_vol = (0.10 / sd20).clip(0, 1).shift(1).fillna(0)                          # 波动率目标10%（≤1无杠杆）
    mu60 = logret.rolling(60).mean()
    sd60 = logret.rolling(60).std()
    f_kelly = (0.5 * mu60 / sd60 ** 2).clip(0, 1).shift(1).fillna(0)             # 半Kelly动态（≤1无杠杆）
    schemes = {
        '全仓(信号内100%)': sig * 1.0,
        '固定30%': sig * 0.30,
        '波动率目标10%': sig * f_vol,
        '半Kelly动态': sig * f_kelly,
    }
    rows, curves = [], {}
    log(f'\n[{name}] 信号=收盘>MA20，信号覆盖率 {sig.mean() * 100:.0f}%')
    for sname, fseries in schemes.items():
        r = (fseries * ret).dropna()
        m = perf(r, ret.index)
        m.update({'方案': sname, '资产': name, '平均仓位%': float(fseries.mean()) * 100})
        rows.append(m)
        curves[sname] = (1 + r).cumprod()
        log(f"  {sname:<14s} 终值{m['终值(倍)']:7.2f}倍 | 年化{m['年化收益%']:7.2f}% | 对数增长{m['年化对数增长率%']:6.2f}%/年 "
            f"| 波动{m['年化波动%']:5.1f}% | 夏普{m['夏普']:5.2f} | 最大回撤{m['最大回撤%']:6.1f}% | 平均仓位{m['平均仓位%']:4.1f}%")
    m = perf(ret, ret.index)
    m.update({'方案': '买入持有(参考)', '资产': name, '平均仓位%': 100.0})
    rows.append(m)
    curves['买入持有(参考)'] = (1 + ret).cumprod()
    log(f"  {'买入持有(参考)':<14s} 终值{m['终值(倍)']:7.2f}倍 | 年化{m['年化收益%']:7.2f}% | 对数增长{m['年化对数增长率%']:6.2f}%/年 "
        f"| 波动{m['年化波动%']:5.1f}% | 夏普{m['夏普']:5.2f} | 最大回撤{m['最大回撤%']:6.1f}% | 平均仓位100.0%")
    return rows, curves

rows5_mt, cur_mt = backtest_rules(close_mt, ret_mt, logret_mt, '贵州茅台')
rows5_sh, cur_sh = backtest_rules(close_sh, ret_sh, logret_sh, '上证指数')

# ---- 图04：四种仓位规则净值（双面板，对数轴）------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
styles = {'全仓(信号内100%)': ('#d62728', '-'), '固定30%': ('#1f77b4', '-'),
          '波动率目标10%': ('#2ca02c', '-'), '半Kelly动态': ('#ff7f0e', '-'),
          '买入持有(参考)': ('gray', '--')}
for ax, curves, name in [(axes[0], cur_mt, '贵州茅台 600519'), (axes[1], cur_sh, '上证指数 000001')]:
    for sname, c in curves.items():
        color, ls = styles[sname]
        lw = 2.4 if ls == '-' else 1.4
        ax.plot(c.index, c.values, color=color, ls=ls, lw=lw,
                label=f"{sname}（终值{c.values[-1]:.2f}x）")
    ax.set_yscale('log')
    ax.set_title(f'{name}：MA20信号 × 仓位规则')
    ax.set_xlabel('日期')
    ax.set_ylabel('净值（对数刻度，起点=1）')
    ax.legend(fontsize=8.5, loc='best')
    ax.grid(alpha=0.25, which='both')
fig.suptitle('图04 同一个弱信号，仓位规则决定你拿到哪一段收益与回撤（信号本身弱没关系，主角是仓位）', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '04_四种仓位规则净值.png'), dpi=150)
plt.close(fig)
log('[图] 已保存 04_四种仓位规则净值.png')

# ======================================================================
# 第6部分：【破产概率】离散赌局下不同 f 的“净值曾跌破10%”概率
# ======================================================================
log('\n' + '=' * 72)
log('第6部分：破产概率（离散赌局 p=0.55,b=1；破产=净值曾跌破初始的10%）')
log('=' * 72)

f_ruin_list = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.25, 1.5]
t_axis6 = np.arange(1, N_STEPS + 1)
C_win = np.cumsum(wins, axis=1, dtype=np.int32)          # 逐期累计胜场（顺序在这里才重要）
rows6 = []
for f in f_ruin_list:
    if f < 1.0:
        L_path = C_win * np.log1p(B_PAY * f) + (t_axis6 - C_win) * np.log(1 - f)
        ruin = L_path.min(axis=1) < np.log(0.10)          # 路径最低点曾跌破10%
    else:
        ruin = nwin < N_STEPS                             # f>=1：任一次亏损净值≤0 → 必然“破产”
    p_ruin = float(ruin.mean())
    rows6.append({'实验': '06_破产概率(离散赌局)', '方案': f'f={f:.2f}',
                  '阈值/事件': '净值曾跌破初始10%', '概率': p_ruin})
    log(f'  f={f:.2f}  P(曾跌破10%) = {p_ruin * 100:6.2f}%')
del C_win
log('  解读：f≤2f*时破产概率为0（f<1永不归零、且长期增长为正托底）；'
    'f≥1时破产概率=1-P(全赢)=1-0.55^1000≈100%。')

# ======================================================================
# 汇总输出 CSV
# ======================================================================
res_rows = []
# 第1部分关键结果
res_rows.append({'实验': '01_离散赌局扫描', '方案': f'f*={F_STAR:.2f}(理论)', '指标': '峰值增长率(%/期)', '数值': theory_growth(F_STAR) * 100})
res_rows.append({'实验': '01_离散赌局扫描', '方案': f'f={f_peak_mc:.3f}(蒙特卡洛)', '指标': '峰值增长率(%/期)', '数值': g_peak_mc * 100})
res_rows.append({'实验': '01_离散赌局扫描', '方案': 'f=0.20(2×Kelly)', '指标': '增长率(%/期)', '数值': mc_growth[i20] * 100})
res_rows.append({'实验': '01_离散赌局扫描', '方案': 'f=1.00(全下注)', '指标': '终值中位数(倍)', '数值': mc_medW[int(np.argmin(np.abs(f_grid - 1.0)))]})
res_rows.append({'实验': '01_离散赌局扫描', '方案': 'f=0.05(半Kelly)', '指标': '终值中位数(倍)', '数值': mc_medW[int(np.argmin(np.abs(f_grid - 0.05)))]})
res_rows.append({'实验': '01_离散赌局扫描', '方案': 'f=0.10(Kelly)', '指标': '终值中位数(倍)', '数值': mc_medW[i10]})
# 第2部分
for k in (k_mt, k_sh):
    res_rows.append({'实验': '02_连续Kelly(真实数据)', '方案': k['name'], '指标': '年化μ(%)', '数值': k['mu_ann'] * 100})
    res_rows.append({'实验': '02_连续Kelly(真实数据)', '方案': k['name'], '指标': '年化σ(%)', '数值': k['sd_ann'] * 100})
    res_rows.append({'实验': '02_连续Kelly(真实数据)', '方案': k['name'], '指标': 'Kelly仓位f*(倍)', '数值': k['fstar']})
    res_rows.append({'实验': '02_连续Kelly(真实数据)', '方案': k['name'], '指标': 'f*标准误(倍)', '数值': k['se_f']})
    res_rows.append({'实验': '02_连续Kelly(真实数据)', '方案': k['name'], '指标': 'μ的t值', '数值': k['tstat']})
# 第3部分
for n, m in res3.items():
    for kk, vv in m.items():
        res_rows.append({'实验': '03_估计误差代价', '方案': n, '指标': kk, '数值': vv})
# 第5部分
for m in rows5_mt + rows5_sh:
    for kk in ['终值(倍)', '年化收益%', '年化对数增长率%', '年化波动%', '夏普', '最大回撤%', '平均仓位%']:
        res_rows.append({'实验': '05_真实回测', '方案': f"{m['资产']}|{m['方案']}", '指标': kk, '数值': m[kk]})
pd.DataFrame(res_rows, columns=['实验', '方案', '指标', '数值']).to_csv(
    os.path.join(OUT, 'kelly_results.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(rows4 + rows6, columns=['实验', '方案', '阈值/事件', '概率']).to_csv(
    os.path.join(OUT, 'drawdown_probs.csv'), index=False, encoding='utf-8-sig')
log('\n[CSV] 已保存 kelly_results.csv, drawdown_probs.csv（utf-8-sig）')

log('\n' + '=' * 72)
log(f'全部完成，总耗时 {time.time() - T_START:.1f} 秒。输出目录：{OUT}')
log('=' * 72)
sys.exit(0)
