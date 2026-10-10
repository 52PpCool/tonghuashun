# -*- coding: utf-8 -*-
"""
第30课：执行算法 —— TWAP 与 VWAP
=================================
回测之外的第二道坎：1个亿的订单，怎么在不砸盘的前提下买进来。

【重要声明】本课的"日内分钟数据"全部是合成模拟（rng(42)），但模拟的刻度用
data/cache/stock_000001_平安银行.csv 的真实日K做了校准：
  - 日成交量规模：近250个交易日的日均成交量、变异系数 → 决定模拟日总量与日间波动
  - 日收益率波动率 → 决定分钟级价格随机游走的幅度
日内分钟成交量的"U型形状"（开收高、中间低）是经验假设——日K数据无法恢复分钟形状。
真实最优执行必须逐票、逐时段重新校准（见 README"如实结论"）。

运行：python main.py    （约10秒，产出 outputs/ 下 5 张图 + is_results.csv）
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 无界面环境出图
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------------------
# 0. 全局常量
# ----------------------------------------------------------------------
RNG_SEED = 42            # 全课程统一随机种子
N_SIMS = 500             # 模拟市场路径条数（同一条路径上跑所有算法，保证可比）
N_MIN = 240              # 一个交易日 = 240 个交易分钟（9:30-11:30, 13:00-15:00）
ORDER_NOTIONAL = 1.0e8   # 基准订单名义金额：1 亿元
HALF_SPREAD = 2.0e-4     # 半个买卖价差 = 2bp：每买入1股都要付一次（流动性成本）
K_TEMP = 2.0e-3          # 临时冲击系数：你吃掉某分钟市场成交量的 x 倍 → 价格瞬时抬升 20bp*x，下一分钟回落
K_PERM = 2.5e-4          # 永久冲击系数：同上，但价格抬升后不回落（Kyle's λ 直觉，见README）
HORIZONS = [10, 30, 60, 120, 240]                        # 时长扫描（分钟）
SIZE_GRID = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]   # 规模扫描：订单占日均成交量的7档
RISK_AVERSION = 0.5      # 风险调整IS = IS均值 + 0.5 * IS标准差（示范性取值）

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'cache', 'stock_000001_平安银行.csv')


# ----------------------------------------------------------------------
# 1. 用真实日K校准模拟参数（平安银行，近250个交易日）
# ----------------------------------------------------------------------
def load_real_calibration(csv_path):
    """返回 (决策价P0, 日波动率σ_d, 日均成交量V_bar, 日成交量变异系数cv, 日均成交额)"""
    df = pd.read_csv(csv_path)          # 列：date, open, high, low, close, volume（前复权价，量为股数）
    df = df.tail(250)                   # 用最近一年代表"现在的流动性"
    ret = df['close'].pct_change().dropna()
    sigma_d = float(ret.std())                          # 日波动率
    v_bar = float(df['volume'].mean())                  # 日均成交量（股）
    cv_vol = float(df['volume'].std() / v_bar)          # 日成交量变异系数（日与日的量能差异）
    turnover = float((df['volume'] * df['close']).mean())  # 日均成交额（元）
    p0 = float(df['close'].iloc[-1])                    # 决策时价格 = 最近收盘价
    return p0, sigma_d, v_bar, cv_vol, turnover


# ----------------------------------------------------------------------
# 2. 模拟市场搭建：U型分钟成交量 + 随机游走价格
# ----------------------------------------------------------------------
def ushape(n_min=N_MIN):
    """日内分钟成交量U型形状（归一化到均值=1）：开盘≈2.0x、午盘≈0.76x、收盘≈2.2x。
    形状是经验假设；"量有多大、日间多不稳"由真实日K校准（见 build_day_pool）。"""
    t = np.arange(n_min)
    u = 1.0 * np.exp(-t / 25.0) + 1.2 * np.exp(-(n_min - 1 - t) / 20.0) + 0.6
    return u / u.mean()


def build_day_pool(rng, u, sigma_d, v_bar, cv_vol, n_sims=N_SIMS):
    """预生成 N_SIMS 条模拟市场路径（同一批路径复用给所有算法，公平对比）。
    每条路径包含：
      v_day: 当日总成交量 ~ 对数正态（均值=真实日均量，离散度=真实变异系数）
      v:     分钟成交量 = 当日总量 × U型形状 × 噪声，再归一化（日内分钟量噪声σ=15%）
      z:     分钟价格随机游走的标准正态新息，分钟波动 σ_m = σ_d / sqrt(240)"""
    sigma_log = np.sqrt(np.log(1.0 + cv_vol ** 2))               # 对数正态参数
    v_day = v_bar * np.exp(sigma_log * rng.standard_normal(n_sims) - 0.5 * sigma_log ** 2)
    xi = np.exp(0.15 * rng.standard_normal((n_sims, N_MIN)) - 0.5 * 0.15 ** 2)
    raw = u[None, :] * xi
    v = raw / raw.sum(axis=1, keepdims=True) * v_day[:, None]    # 每日分钟量加总=当日总量
    z = rng.standard_normal((n_sims, N_MIN))
    return {'v': v, 'z': z, 'v_day': v_day}


# ----------------------------------------------------------------------
# 3. 执行模拟器：给定切片计划，返回每条路径的执行缺口IS（单位bp）
# ----------------------------------------------------------------------
def run_schedule(pool, q_slice, p0, sigma_m, frontrun=False):
    """在500条路径上执行切片计划 q_slice（形状(240,)，每分钟买多少股）。
    每分钟的事件顺序：
      (a) [若 frontrun] 抢跑者按同样的切片提前买入 → 其永久冲击先落地，推高你的成交价
      (b) 你买入：成交价 = 公允价 × (1 + 半价差 + 临时冲击)，临时冲击 ∝ 该分钟参与率
      (c) 公允价演化：F ← F × (1 + 市场随机游走 + 你的永久冲击)
    参与率 rho = 你的分钟买单 / 该分钟市场成交量 → 这就是 Kyle's λ 的直觉：
    λ ∝ 波动率/流动性，你的单子占市场量的比例越高，价格被你推得越高。
    返回 IS（bp）= 加权成交均价 / 决策价 - 1，正数=比决策价买贵了。"""
    v, z = pool['v'], pool['z']
    n = v.shape[0]
    F = np.full(n, p0)          # 公允价格（元）
    cost = np.zeros(n)          # 累计买入金额
    q_total = float(np.sum(q_slice))
    for t in range(N_MIN):
        q_t = q_slice[t]
        if q_t <= 0:                       # 该分钟不交易，价格照常走
            F = F * (1.0 + sigma_m * z[:, t])
            continue
        rho = q_t / v[:, t]                # 该分钟的参与率（你的量/市场量）
        if frontrun:                       # 抢跑实验：对手提前买，永久冲击先砸到你头上
            F = F * (1.0 + K_PERM * rho)
        p_exec = F * (1.0 + HALF_SPREAD + K_TEMP * rho)   # 你的成交价
        cost += q_t * p_exec
        F = F * (1.0 + sigma_m * z[:, t] + K_PERM * rho)  # 你的永久冲击留给市场（也留给后来的自己）
    return (cost / (q_total * p0) - 1.0) * 1e4             # IS，单位 bp


def make_schedule(kind, q_total, horizon=None, u_pred=None, n_min=N_MIN):
    """构造切片计划（每股怎么拆到240分钟）：
    eager : 全部在开盘第1分钟买入
    twap  : 等时间切片，前 horizon 分钟每分钟买 q_total/horizon
    vwap  : 按预测U型量的比例切片（"跟量走"），只用预测形状，不用当日真实量（避免上帝视角）"""
    q = np.zeros(n_min)
    if kind == 'eager':
        q[0] = q_total
    elif kind == 'twap':
        q[:horizon] = q_total / horizon
    elif kind == 'vwap':
        q[:horizon] = q_total * u_pred[:horizon] / u_pred[:horizon].sum()
    else:
        raise ValueError(kind)
    return q


def is_stats(is_bp):
    """IS分布的摘要统计"""
    return {'均值': float(np.mean(is_bp)), '中位数': float(np.median(is_bp)),
            '标准差': float(np.std(is_bp, ddof=1)),
            'P5': float(np.percentile(is_bp, 5)), 'P95': float(np.percentile(is_bp, 95))}


# ----------------------------------------------------------------------
# 4. 主流程：校准 → 模拟 → 四组实验 → 出图 → 落盘
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    rows = []          # is_results.csv 的行缓冲

    def add_rows(exp, algo, horizon, size_pct, is_bp):
        for i, x in enumerate(is_bp):
            rows.append({'实验': exp, '算法': algo, '执行时长_分钟': horizon,
                         '订单占日成交量_pct': round(size_pct, 3), '模拟编号': i, 'IS_bp': round(float(x), 4)})

    # ---- 4.1 真实数据校准 ----
    p0, sigma_d, v_bar, cv_vol, turnover = load_real_calibration(CSV_PATH)
    sigma_m = sigma_d / np.sqrt(N_MIN)                 # 分钟波动率（简化：日内均匀，未模拟日内U型波动率）
    u = ushape()
    pool = build_day_pool(rng, u, sigma_d, v_bar, cv_vol, N_SIMS)

    print('=' * 72)
    print('【真实数据校准】(平安银行 000001，近250个交易日，价格前复权)')
    print(f'  决策价 P0 = {p0:.2f} 元 | 日波动率 σ_d = {sigma_d*1e4:.0f} bp '
          f'| 分钟波动 σ_m = {sigma_m*1e4:.2f} bp')
    print(f'  日均成交量 = {v_bar/1e4:.0f} 万股 | 变异系数 cv = {cv_vol:.2f} '
          f'| 日均成交额 ≈ {turnover/1e8:.1f} 亿元')

    # ---- 4.2 实验一：基准对比（1亿元订单，三种算法各500次模拟） ----
    q0 = ORDER_NOTIONAL / p0                            # 1亿元能买多少股
    pct_adv0 = q0 / v_bar                               # 占日均成交量的比例
    print(f'\n【基准订单】{ORDER_NOTIONAL/1e8:.0f} 亿元 = {q0/1e4:.0f} 万股 '
          f'= 日均成交量的 {pct_adv0*100:.1f}% （对个股来说这是必须认真对待的大单）')

    sched_eager = make_schedule('eager', q0)
    sched_twap = make_schedule('twap', q0, horizon=N_MIN)
    sched_vwap = make_schedule('vwap', q0, horizon=N_MIN, u_pred=u)
    is_eager = run_schedule(pool, sched_eager, p0, sigma_m)
    is_twap240 = run_schedule(pool, sched_twap, p0, sigma_m)
    is_vwap240 = run_schedule(pool, sched_vwap, p0, sigma_m)
    add_rows('基准对比(1亿元订单)', '一次性打入', 1, pct_adv0 * 100, is_eager)
    add_rows('基准对比(1亿元订单)', 'TWAP-240', N_MIN, pct_adv0 * 100, is_twap240)
    add_rows('基准对比(1亿元订单)', 'VWAP-240', N_MIN, pct_adv0 * 100, is_vwap240)

    tbl = pd.DataFrame({'一次性打入(开盘全买)': is_stats(is_eager),
                        'TWAP-240(等时间)': is_stats(is_twap240),
                        'VWAP-240(跟量走)': is_stats(is_vwap240)}).T
    print('\n【实验一：三算法IS分布】(bp，500次模拟)')
    print(tbl.round(1).to_string())

    # ---- 4.3 实验二：执行时长扫描（快=付冲击，慢=担风险，U型在哪） ----
    twap_scan, vwap_scan = {}, {}
    for T in HORIZONS:
        twap_scan[T] = run_schedule(pool, make_schedule('twap', q0, horizon=T), p0, sigma_m)
        vwap_scan[T] = run_schedule(pool, make_schedule('vwap', q0, horizon=T, u_pred=u), p0, sigma_m)
        add_rows('时长扫描(1亿元订单)', 'TWAP', T, pct_adv0 * 100, twap_scan[T])
        add_rows('时长扫描(1亿元订单)', 'VWAP', T, pct_adv0 * 100, vwap_scan[T])
    twap_mean = np.array([np.mean(twap_scan[T]) for T in HORIZONS])
    vwap_mean = np.array([np.mean(vwap_scan[T]) for T in HORIZONS])
    twap_risk = twap_mean + RISK_AVERSION * np.array([np.std(twap_scan[T], ddof=1) for T in HORIZONS])
    vwap_risk = vwap_mean + RISK_AVERSION * np.array([np.std(vwap_scan[T], ddof=1) for T in HORIZONS])
    t_best_twap = HORIZONS[int(np.argmin(twap_risk))]
    t_best_vwap = HORIZONS[int(np.argmin(vwap_risk))]

    print(f'\n【实验二：执行时长扫描】(1亿元订单，IS均值bp | 风险调整ISbp)')
    print('  T(分钟) |    TWAP           |    VWAP')
    for i, T in enumerate(HORIZONS):
        print(f'  {T:6d}  | {twap_mean[i]:7.1f} ({twap_risk[i]:7.1f}) | {vwap_mean[i]:7.1f} ({vwap_risk[i]:7.1f})')
    print(f'  一次性打入(≈1分钟): IS均值 {np.mean(is_eager):.1f} bp')
    print(f'  风险调整IS最优点：TWAP≈{t_best_twap}分钟, VWAP≈{t_best_vwap}分钟')

    # ---- 4.4 实验三：订单规模扫描（小单无所谓、大单必须切） ----
    scale_res = {a: [] for a in ('eager', 'twap', 'vwap')}
    for pct in SIZE_GRID:
        qs = pct * v_bar                                # 订单 = pct × 日均成交量（股）
        scale_res['eager'].append(np.mean(run_schedule(pool, make_schedule('eager', qs), p0, sigma_m)))
        scale_res['twap'].append(np.mean(run_schedule(pool, make_schedule('twap', qs, horizon=N_MIN), p0, sigma_m)))
        scale_res['vwap'].append(np.mean(run_schedule(pool, make_schedule('vwap', qs, horizon=N_MIN, u_pred=u), p0, sigma_m)))
        add_rows('规模扫描', '一次性打入', 1, pct * 100,
                 run_schedule(pool, make_schedule('eager', qs), p0, sigma_m))
        add_rows('规模扫描', 'TWAP-240', N_MIN, pct * 100,
                 run_schedule(pool, make_schedule('twap', qs, horizon=N_MIN), p0, sigma_m))
        add_rows('规模扫描', 'VWAP-240', N_MIN, pct * 100,
                 run_schedule(pool, make_schedule('vwap', qs, horizon=N_MIN, u_pred=u), p0, sigma_m))
    print('\n【实验三：订单规模扫描】(IS均值bp)')
    print('  规模%ADV | 一次性打入 |  TWAP-240 |  VWAP-240')
    for i, pct in enumerate(SIZE_GRID):
        print(f'  {pct*100:6.0f}%   | {scale_res["eager"][i]:9.1f} | {scale_res["twap"][i]:9.1f} '
              f'| {scale_res["vwap"][i]:9.1f}')

    # ---- 4.5 实验四：TWAP被抢跑（规律性买单的代价） ----
    is_front = run_schedule(pool, sched_twap, p0, sigma_m, frontrun=True)
    add_rows('抢跑实验(1亿元TWAP)', 'TWAP-240无抢跑', N_MIN, pct_adv0 * 100, is_twap240)
    add_rows('抢跑实验(1亿元TWAP)', 'TWAP-240被抢跑', N_MIN, pct_adv0 * 100, is_front)
    extra_bp = float(np.mean(is_front) - np.mean(is_twap240))
    extra_money = extra_bp / 1e4 * ORDER_NOTIONAL       # 多付的钱（元）
    print(f'\n【实验四：TWAP被抢跑】对手提前一分钟按同样切片买入')
    print(f'  正常TWAP IS均值 = {np.mean(is_twap240):.1f} bp → 被抢跑 = {np.mean(is_front):.1f} bp')
    print(f'  多付 {extra_bp:.1f} bp ≈ {extra_money/1e4:.0f} 万元（近似等于抢跑者无风险套走的钱）')

    # ---- 4.6 出图 ----
    # 图01：日内U型成交量 + 三算法切片对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    v0 = pool['v'][0]
    ax1.bar(np.arange(N_MIN), v0 / 1e4, color='lightsteelblue', width=1.0, label='模拟真实分钟成交量(第0天)')
    ax1.plot(np.arange(N_MIN), pool['v_day'][0] * u / N_MIN / 1e4, color='crimson', lw=2,
             label='预测U型（VWAP切片用它）')
    ax1.axhline(pool['v_day'][0] / N_MIN / 1e4, color='gray', ls='--', lw=1.2, label='等时间假设（TWAP切片）')
    ax1.set_xlabel('分钟（0=9:30开盘，239=15:00收盘）')
    ax1.set_ylabel('分钟成交量（万股）')
    ax1.set_title('日内成交量U型：开收高、中间低\n（合成数据，规模按平安银行真实日成交量校准）')
    ax1.legend(fontsize=8)
    ax2.step(np.arange(N_MIN), np.cumsum(sched_eager) / q0, where='post', color='crimson', lw=2,
             label='一次性打入：开盘一步到位')
    ax2.plot(np.arange(N_MIN), np.cumsum(sched_twap) / q0, color='steelblue', lw=2,
             label='TWAP：跟时间走（每分钟1/240）')
    ax2.plot(np.arange(N_MIN), np.cumsum(sched_vwap) / q0, color='seagreen', lw=2,
             label='VWAP：跟量走（量大多买、量少少买）')
    ax2.set_xlabel('分钟')
    ax2.set_ylabel('累计已执行比例')
    ax2.set_title('三种算法的执行切片：同样1亿元，路径完全不同')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle('图01 日内U型成交量与三算法切片对比', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_日内U型成交量与三算法切片对比.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图02：三算法IS分布箱线图
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [is_eager, is_twap240, is_vwap240]
    labels = ['一次性打入\n(开盘全买)', 'TWAP-240\n(等时间切片)', 'VWAP-240\n(按U型量切片)']
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True, widths=0.5,
                    meanprops={'marker': 'D', 'markerfacecolor': 'gold', 'markeredgecolor': 'black',
                               'markersize': 6})
    for patch, c in zip(bp['boxes'], ['#ffd6d6', '#d6e6ff', '#d6f0dd']):
        patch.set_facecolor(c)
    for i, d in enumerate(data):
        ax.text(i + 1.28, np.mean(d), f'均值\n{np.mean(d):.0f}bp', fontsize=9, va='center')
    ax.axhline(0, color='gray', lw=1, ls=':')
    ax.set_ylabel('执行缺口 IS（bp）')
    ax.set_title(f'图02 三算法IS分布（500次模拟，1亿元订单≈{pct_adv0*100:.0f}%日成交量）\n'
                 '一次性打入：代价集中在冲击成本（随当日开盘量浮动）；切片算法：均值低约200bp，'
                 '但被市场随机波动拉宽——慢=用确定的冲击换市场风险')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_三算法IS分布箱线图.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图03：IS均值 vs 执行时长（U型与最优点）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
    xs = [1] + HORIZONS
    ax1.plot(xs, [np.mean(is_eager)] + list(twap_mean), 'o-', color='steelblue', label='TWAP（等时间）')
    ax1.plot(xs, [np.mean(is_eager)] + list(vwap_mean), 's-', color='seagreen', label='VWAP（跟量）')
    ax1.scatter([1], [np.mean(is_eager)], marker='^', s=110, color='crimson', zorder=5)
    ax1.annotate(f'一次性打入\n{np.mean(is_eager):.0f}bp', xy=(1, np.mean(is_eager)),
                 xytext=(1.5, np.mean(is_eager) * 0.92), fontsize=9, color='crimson')
    ax1.set_xscale('log')
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_xlabel('执行时长 T（分钟，对数轴；1≈一次性打入）')
    ax1.set_ylabel('IS均值（bp）')
    ax1.set_title('IS均值 vs 执行时长\n快=参与率高付冲击；慢=拖进午盘薄流动性，冲击反而回升')
    ax1.legend()
    ax1.grid(alpha=0.3, which='both')
    ax2.plot(HORIZONS, twap_risk, 'o-', color='steelblue', label='TWAP：均值+0.5×标准差')
    ax2.plot(HORIZONS, vwap_risk, 's-', color='seagreen', label='VWAP：均值+0.5×标准差')
    for arr, col, mk, name, dxy in ((twap_risk, 'steelblue', '*', 'TWAP', (0.40, 1.16)),
                                    (vwap_risk, 'seagreen', '*', 'VWAP', (2.6, 1.005))):
        i = int(np.argmin(arr))
        ax2.scatter([HORIZONS[i]], [arr[i]], marker=mk, s=220, color=col, edgecolors='black', zorder=5)
        ax2.annotate(f'{name}最优 T={HORIZONS[i]}分钟\n{arr[i]:.0f}bp', xy=(HORIZONS[i], arr[i]),
                     xytext=(HORIZONS[i] * dxy[0], arr[i] * dxy[1]), fontsize=9)
    ax2.set_xscale('log')
    ax2.set_xticks(HORIZONS)
    ax2.set_xticklabels([str(x) for x in HORIZONS])
    ax2.set_xlabel('执行时长 T（分钟，对数轴）')
    ax2.set_ylabel('风险调整IS（bp）')
    ax2.set_title('风险调整IS：市场风险（σ√T）把曲线压成U型\n纯均值曲线在零漂移下没有"上升支"——风险藏在分布里')
    ax2.legend()
    ax2.grid(alpha=0.3, which='both')
    fig.suptitle('图03 执行时长扫描：1亿元订单的"快慢权衡"', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_IS均值vs执行时长.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图04：IS vs 订单规模
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pct_pts = [p * 100 for p in SIZE_GRID]
    ax.plot(pct_pts, scale_res['eager'], '^-', color='crimson', label='一次性打入（开盘全买）')
    ax.plot(pct_pts, scale_res['twap'], 'o-', color='steelblue', label='TWAP-240（等时间切片）')
    ax.plot(pct_pts, scale_res['vwap'], 's-', color='seagreen', label='VWAP-240（跟量切片）')
    ax.axvline(pct_adv0 * 100, color='gray', ls='--', lw=1.2)
    ax.text(pct_adv0 * 100 * 1.1, scale_res['vwap'][0] * 2.2,
            f'本课基准订单\n1亿元≈{pct_adv0*100:.0f}%', fontsize=9, color='gray')
    ratio = scale_res['eager'][-1] / scale_res['vwap'][-1]
    ax.annotate(f'50%ADV时：一次性打入 {scale_res["eager"][-1]:.0f}bp\n'
                f'≈ VWAP的 {ratio:.0f} 倍', xy=(50, scale_res['eager'][-1]),
                xytext=(13, scale_res['eager'][-1] * 0.55), fontsize=9,
                arrowprops={'arrowstyle': '->', 'color': 'black'})
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(pct_pts)
    ax.set_xticklabels([f'{p:g}%' for p in pct_pts])
    ax.set_xlabel('订单规模（占日均成交量的%，对数轴）')
    ax.set_ylabel('IS均值（bp，对数轴）')
    ax.set_title('图04 冲击敏感性：小单无所谓、大单必须切\n（线性冲击模型下切片收益∝参与率下降倍数；超大单真实冲击是凹形，见README）')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_ISvs订单规模.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图05：被抢跑实验
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bp = ax.boxplot([is_twap240, is_front], tick_labels=['TWAP-240\n（无抢跑）', 'TWAP-240\n（每片被提前抢买）'],
                    showmeans=True, patch_artist=True, widths=0.45,
                    meanprops={'marker': 'D', 'markerfacecolor': 'gold', 'markeredgecolor': 'black',
                               'markersize': 6})
    for patch, c in zip(bp['boxes'], ['#d6e6ff', '#ffe0b3']):
        patch.set_facecolor(c)
    ax.axhline(0, color='gray', lw=1, ls=':')
    m0, m1 = np.mean(is_twap240), np.mean(is_front)
    ax.annotate('', xy=(2.18, m1), xytext=(2.18, m0),
                arrowprops={'arrowstyle': '<->', 'color': 'crimson', 'lw': 1.6})
    ax.text(2.24, (m0 + m1) / 2, f'多付 {extra_bp:.1f} bp\n≈ {extra_money/1e4:.0f} 万元',
            fontsize=10, color='crimson', va='center')
    ax.set_ylabel('执行缺口 IS（bp）')
    ax.set_title('图05 TWAP的可预测性：对手看到你按时间规律买，提前一分钟照方抓药\n'
                 '抢跑者的买单先推高价格（永久冲击），你每一片都买在更贵的位置')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '05_被抢跑实验.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 4.7 落盘 CSV（utf-8-sig，Excel可直接打开） ----
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'is_results.csv'),
                              index=False, encoding='utf-8-sig')

    # ---- 4.8 关键数字汇总（README引用的都来自这里的真实输出） ----
    print('\n' + '=' * 72)
    print('【关键数字汇总】')
    print(f"  校准: P0={p0:.2f}元 σ_d={sigma_d*1e4:.0f}bp V_bar={v_bar/1e4:.0f}万股 "
          f"cv={cv_vol:.2f} 日均成交额={turnover/1e8:.1f}亿元")
    print(f"  基准订单: 1亿元={q0/1e4:.0f}万股≈{pct_adv0*100:.1f}%日均成交量")
    print(f"  三算法IS均值(bp): 一次性打入={np.mean(is_eager):.1f} TWAP-240={np.mean(is_twap240):.1f} "
          f"VWAP-240={np.mean(is_vwap240):.1f}")
    print(f"  三算法IS标准差(bp): 一次性打入={np.std(is_eager, ddof=1):.1f} "
          f"TWAP-240={np.std(is_twap240, ddof=1):.1f} VWAP-240={np.std(is_vwap240, ddof=1):.1f}")
    print(f"  IS均值最小处: TWAP T={HORIZONS[int(np.argmin(twap_mean))]}分钟({np.min(twap_mean):.1f}bp) "
          f"VWAP T={HORIZONS[int(np.argmin(vwap_mean))]}分钟({np.min(vwap_mean):.1f}bp)")
    print(f"  风险调整IS最优: TWAP T={t_best_twap}分钟({np.min(twap_risk):.1f}bp) "
          f"VWAP T={t_best_vwap}分钟({np.min(vwap_risk):.1f}bp)")
    print(f"  规模扫描(1%→50%ADV, IS均值bp): 一次性打入 {scale_res['eager'][0]:.1f}→{scale_res['eager'][-1]:.1f} "
          f"| VWAP-240 {scale_res['vwap'][0]:.1f}→{scale_res['vwap'][-1]:.1f}")
    print(f"  抢跑: TWAP-240 IS {m0:.1f}bp → {m1:.1f}bp, 多付{extra_bp:.1f}bp≈{extra_money/1e4:.0f}万元")
    print(f"\n输出文件已写入: {OUT_DIR}")
    print('  01_日内U型成交量与三算法切片对比.png ... 05_被抢跑实验.png, is_results.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
