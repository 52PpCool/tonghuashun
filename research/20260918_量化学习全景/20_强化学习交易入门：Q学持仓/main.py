# -*- coding: utf-8 -*-
"""
20_强化学习交易入门：Q学持仓
==============================
用约 30 行核心代码手写 Q-learning（纯 numpy，不用任何强化学习库），
让一个"表格型智能体"自己学会一件事：每天收盘时决定——明天 空仓 还是 满仓。

数据：上证指数日K（2010 起，真实数据缓存 data/cache/index_000001_上证指数.csv）。
划分：训练段 2010-2019（Q表只在这里学习），样本外 2020-01-01 至今（只评估、不学习）。

本课最重要的结论不是"RL 能赚钱"，而是让你亲眼看到裸 Q-learning 在金融上的
三大真实困难：市场非平稳、状态覆盖不足、结果对随机种子（探索路径）敏感。

运行：/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py
耗时：约 30 秒；随机种子 42，结果可复现。
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面环境绘图
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 全局配置 =================
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
DATA_CANDIDATES = [
    os.path.join(BASE_DIR, '..', 'data', 'cache', 'index_000001_上证指数.csv'),
    '/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/data/cache/index_000001_上证指数.csv',
]
TRAIN_END = '2020-01-01'      # 该日之前为训练段，之后为样本外

# ---- 环境（状态/奖励）设计 ----
MOM_DEAD = 0.02               # 动量死区：|20日动量|<=2% 记为"震荡"档（避免在0附近反复横跳）
N_STATE = 9                   # 动量3档 × 波动3档 = 9 个状态
COST = 3e-4                   # 单边佣金万3；一次换仓执行一边（一买一卖往返共万6）
A_NAMES = ['空仓', '满仓']     # 动作0=空仓，动作1=满仓

# ---- Q-learning 超参数 ----
GAMMA = 0.95                  # 折扣因子：有效视野约 1/(1-γ)=20 个交易日 ≈ 一个月
ALPHA = 0.1                   # 学习率：每次用多大幅度修正 Q 值
EPS_START, EPS_MIN, EPS_DECAY = 0.5, 0.05, 0.95   # ε-greedy：探索率从0.5指数衰减到0.05
N_EPISODES = 80               # 训练轮数（每轮完整遍历一遍训练段）

MOM_NAMES = ['负动量', '震荡', '正动量']
VOL_NAMES = ['低波动', '中波动', '高波动']


# ================= 1. 数据加载（真实优先，合成兜底，绝不失败） =================
def load_close():
    """读上证指数收盘价缓存；全部缺失时用 rng(42) 造合成数据并明确标注。"""
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=['date']).set_index('date')
            print(f'[数据] 使用真实缓存：{os.path.normpath(p)}（{len(df)} 行）')
            return df[['close']].copy(), False
    rng = np.random.default_rng(42)   # 兜底：带牛熊周期的合成GBM序列
    n = 4000
    drift = np.where(np.sin(np.arange(n) / 150.0) > 0, 0.0007, -0.0003)
    ret = rng.normal(drift, 0.012)
    idx = pd.bdate_range('2010-01-04', periods=n)
    df = pd.DataFrame({'close': 3000.0 * np.exp(np.cumsum(ret))}, index=idx)
    print('[数据] 警告：未找到真实缓存，改用合成数据（rng(42)，带牛熊周期的GBM）')
    return df, True


def build_states(close):
    """
    状态离散化（本课最关键的建模决策）：
      特征1：20日动量 close/close[-20]-1，按符号分3档（<-2% 负 / ±2% 内 震荡 / >2% 正）
      特征2：20日波动率（日收益标准差），按训练段三分位分3档（低/中/高）
      状态 = 动量档*3 + 波动档 → 0..8 共 9 个状态
    注意：波动率分位阈值只用训练段估计，避免偷看样本外（前视偏差）。
    """
    df = pd.DataFrame(index=close.index)
    df['ret'] = close['close'].pct_change()        # 当日收益
    df['mom'] = close['close'].pct_change(20)      # 20日动量（过去一个月涨跌幅）
    df['vol'] = df['ret'].rolling(20).std()        # 20日波动率
    df = df.dropna().copy()

    is_train = df.index < TRAIN_END
    v_lo, v_hi = df.loc[is_train, 'vol'].quantile([1 / 3, 2 / 3])   # 只用训练段定阈值
    df['mom_bin'] = np.select([df['mom'] < -MOM_DEAD, df['mom'] > MOM_DEAD], [0, 2], default=1)
    df['vol_bin'] = np.select([df['vol'] <= v_lo, df['vol'] <= v_hi], [0, 1], default=2)
    df['state'] = df['mom_bin'] * 3 + df['vol_bin']

    df['ret_next'] = df['ret'].shift(-1).fillna(0.0)   # 次日收益：今天收盘决策、明天兑现
    print(f'[状态] 波动率三分位阈值（仅用训练段估计）：低/中界={v_lo:.5f}，中/高界={v_hi:.5f}')
    print(f'[状态] 离散化后样本 {len(df)} 天，训练段 {int(is_train.sum())} 天，样本外 {int((~is_train).sum())} 天')
    return df


# ================= 2. 环境（MDP 五要素的落地） =================
class MarketEnv:
    """
    最小交易环境：智能体每天收盘观察状态，选择明天的目标仓位。

    MDP 五要素在这里的映射：
      状态 S：state[i]（动量档×波动档 → 0..8）——对市场的压缩描述
      动作 A：0=空仓，1=满仓（明天持多大仓位）
      转移 P：市场说了算。明天落到哪个状态智能体控制不了，这就是环境的随机性
      奖励 R：次日收益×仓位 − 换仓成本。奖励=你真正想要的目标（赚钱）的直接代理
      折扣 γ：未来奖励打多少折。γ=0.95 → 有效视野约 20 个交易日（约一个月），
              γ 越小越短视（近似单日收益），γ 越大越看重长期（但样本内更难学）
    """

    def __init__(self, state, ret_next):
        self.state = np.asarray(state, dtype=int)
        self.ret_next = np.asarray(ret_next, dtype=float)

    def step(self, i, action, prev_pos):
        """第 i 天收盘执行 action（明日仓位），返回即时奖励（次日才真正兑现盈亏）。"""
        cost = COST * abs(action - prev_pos)          # 换仓成本：按换手比例收单边万3
        r = action * self.ret_next[i] - cost          # 奖励 = 次日收益×仓位 − 成本
        return r


# ================= 3. Q-learning 核心（约30行，纯 numpy） =================
def q_learning_train(env, i0, i1, seed, n_episodes=N_EPISODES, record_curve=False):
    """
    表格型 Q-learning：Q表 9×2，ε-greedy 探索，每轮完整遍历训练段 [i0, i1]。
    返回 (Q表, 探索奖励曲线, 贪心评估曲线)。
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATE, 2))                    # Q表：9个状态×2个动作，全部初始化为0
    curve_explore, curve_greedy = [], []
    eps = EPS_START
    for ep in range(n_episodes):                  # 外层：多轮遍历同一段历史
        ep_reward, prev_pos = 0.0, 0              # 每轮从空仓开始（期初无持仓）
        for i in range(i0, i1 + 1):               # 内层：按时间顺序走一遍训练段
            s = env.state[i]
            # ---- ε-greedy：以 ε 概率随机探索，否则贪心利用当前 Q 表 ----
            if rng.random() < eps:
                a = int(rng.integers(2))
            else:
                a = int(np.argmax(Q[s]))          # 平手时 argmax 取索引0=空仓（偏保守）
            r = env.step(i, a, prev_pos)
            # ---- Q-learning 更新（全部算法就这2行）----
            # Q[s,a] ← Q[s,a] + α·[ r + γ·max_a' Q[s',a'] − Q[s,a] ]
            if i == i1:                           # 回合结束：无后续状态，目标只有 r
                target = r
            else:
                target = r + GAMMA * Q[env.state[i + 1]].max()
            Q[s, a] += ALPHA * (target - Q[s, a])
            prev_pos = a
            ep_reward += r
        eps = max(EPS_MIN, eps * EPS_DECAY)       # 探索率衰减：前期多探索、后期多利用
        if record_curve:
            curve_explore.append(ep_reward)
            curve_greedy.append(greedy_eval(env, Q, i0, i1))      # 无探索的贪心评估
    return Q, curve_explore, curve_greedy


def greedy_positions(env, Q, i0, i1):
    """贪心策略（不带探索）在 [i0, i1] 每天给出的仓位。"""
    return np.argmax(Q[env.state[i0:i1 + 1]], axis=1).astype(int)


def greedy_eval(env, Q, i0, i1):
    """贪心策略在 [i0, i1] 的（未折扣）累计奖励，用于画训练过程曲线。"""
    pos = greedy_positions(env, Q, i0, i1)
    turn = np.abs(np.diff(pos, prepend=0))        # 期初空仓，首次建仓也计一次换手
    return float((pos * env.ret_next[i0:i1 + 1] - COST * turn).sum())


# ================= 4. 回测与绩效指标 =================
def strategy_returns(pos, ret_next_seg):
    """由仓位序列生成日收益序列：仓位在 T 日收盘确定，赚取 T+1 日收益。"""
    turn = np.abs(np.diff(pos, prepend=0))
    return pos * ret_next_seg - COST * turn


def perf_stats(daily_ret):
    """年化收益/波动、夏普（统一 rf=2% 口径：(年化收益−0.02)/年化波动；现金腿记0收益未计息）、最大回撤。nav 基于 nav(T)=nav(T-1)*(1+r)。"""
    r = np.asarray(daily_ret, dtype=float)
    n = len(r)
    nav = np.cumprod(1.0 + r)
    total = nav[-1] - 1.0
    ann = (1.0 + total) ** (252.0 / max(n, 1)) - 1.0
    vol = r.std(ddof=1) * np.sqrt(252.0) if n > 1 else np.nan
    sharpe = (ann - 0.02) / vol if vol > 0 else np.nan   # 统一口径 rf=2%：年化收益扣 2% 后再除以年化波动；vol为0或NaN时夏普记为NaN
    dd = nav / np.maximum.accumulate(nav) - 1.0
    return {'年化收益%': ann * 100, '年化波动%': vol * 100, '夏普比率': sharpe,
            '最大回撤%': dd.min() * 100, '累计收益%': total * 100, '样本天数': n}


def fmt_stats(st):
    return (f"年化 {st['年化收益%']:+7.2f}%  波动 {st['年化波动%']:5.2f}%  "
            f"夏普 {st['夏普比率']:+5.2f}  最大回撤 {st['最大回撤%']:6.2f}%")


# ================= 5. 实验主流程 =================
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    close, is_synth = load_close()
    df = build_states(close)
    n = len(df)
    env = MarketEnv(df['state'].to_numpy(), df['ret_next'].to_numpy())

    is_train = df.index < TRAIN_END
    n_tr = int(is_train.sum())
    SEG = {'训练段(2010-2019)': (0, n_tr - 1), '样本外(2020起)': (n_tr, n - 2)}
    dates = df.index

    # ---------- 5.1 基准训练（种子42） ----------
    print(f'\n[训练] Q-learning：{N_EPISODES}轮 × 训练段，种子={SEED}，'
          f'α={ALPHA}, γ={GAMMA}, ε:{EPS_START}→{EPS_MIN}')
    Q, curve_explore, curve_greedy = q_learning_train(env, *SEG['训练段(2010-2019)'],
                                                      seed=SEED, record_curve=True)

    # ---------- 5.2 三种策略 × 两段数据 的绩效 ----------
    # 状态规则版：手写等价规则——"正动量且非高波动 → 满仓，否则空仓"
    rule_pos_all = ((df['mom_bin'] == 2) & (df['vol_bin'] <= 1)).astype(int).to_numpy()
    strategies = {
        'Q学习策略(种子42)': {},       # 稍后填入贪心仓位
        '状态规则(动量+波动)': {},
        '买入持有': {},
    }
    for seg_name, (i0, i1) in SEG.items():
        strategies['Q学习策略(种子42)'][seg_name] = greedy_positions(env, Q, i0, i1)
        strategies['状态规则(动量+波动)'][seg_name] = rule_pos_all[i0:i1 + 1]
        strategies['买入持有'][seg_name] = np.ones(i1 - i0 + 1, dtype=int)

    metrics_rows = []
    nav_curves = {}
    print()
    for seg_name, (i0, i1) in SEG.items():
        ret_seg = env.ret_next[i0:i1 + 1]
        print(f'==== {seg_name}（决策期 {dates[i0].date()} ~ {dates[i1].date()}，'
              f'收益兑现至 {dates[i1 + 1].date()}，共 {i1 - i0 + 1} 个交易日）====')
        for s_name in strategies:
            pos = strategies[s_name][seg_name]
            r = strategy_returns(pos, ret_seg)
            st = perf_stats(r)
            metrics_rows.append({'数据段': seg_name, '策略': s_name, '种子': str(SEED) if 'Q学习' in s_name else '',
                                 '备注': '', **st})
            nav_curves[(seg_name, s_name)] = (dates[i0 + 1:i1 + 2], np.cumprod(1 + r))
            print(f'  {s_name:<16s} {fmt_stats(st)}')

    # ---------- 5.3 Q表与状态覆盖 ----------
    cnt_tr = np.bincount(env.state[0:n_tr], minlength=N_STATE)
    cnt_te = np.bincount(env.state[n_tr:], minlength=N_STATE)
    # 各状态的"次日平均收益"（诊断用：对照Q表，看它学到的是真信号还是探索路径的偶然产物）
    rn_arr, st_arr = df['ret_next'].to_numpy(), df['state'].to_numpy()
    mean_tr = np.array([rn_arr[:n_tr][st_arr[:n_tr] == s].mean() for s in range(N_STATE)]) * 1e4
    mean_te = np.array([rn_arr[n_tr:n - 1][st_arr[n_tr:n - 1] == s].mean() for s in range(N_STATE)]) * 1e4
    q_rows = []
    for s in range(N_STATE):
        m, v = divmod(s, 3)
        q_rows.append({
            '状态编号': s, '动量档': MOM_NAMES[m], '波动档': VOL_NAMES[v],
            'Q_空仓': Q[s, 0], 'Q_满仓': Q[s, 1], 'Q差(满-空)': Q[s, 1] - Q[s, 0],
            '最终动作': A_NAMES[int(np.argmax(Q[s]))],
            '训练段天数': int(cnt_tr[s]), '训练段占比%': 100 * cnt_tr[s] / n_tr,
            '样本外天数': int(cnt_te[s]), '样本外占比%': 100 * cnt_te[s] / max(n - n_tr, 1),
            '训练段次日均收益bp': mean_tr[s], '样本外次日均收益bp': mean_te[s],
        })
    q_table = pd.DataFrame(q_rows)
    q_table.to_csv(os.path.join(OUT_DIR, 'q_table.csv'), index=False, encoding='utf-8-sig')

    print('\n==== Q表学到的策略（训练段学完，直接搬到样本外）====')
    for _, row in q_table.iterrows():
        seen = '训练时见过' if row['训练段天数'] > 0 else '训练时从未见过!'
        print(f"  状态{row['状态编号']}（{row['动量档']}+{row['波动档']}）→ {row['最终动作']}"
              f"  Q空={row['Q_空仓']:+.4f} Q满={row['Q_满仓']:+.4f}"
              f"  次日均收益 训练{row['训练段次日均收益bp']:+6.1f}bp→样本外{row['样本外次日均收益bp']:+6.1f}bp"
              f"  训练占比{row['训练段占比%']:5.1f}%→样本外{row['样本外占比%']:5.1f}%  [{seen}]")
    print('  注：Q值包含γ折扣的未来收益（bootstrap），故与"该状态次日平均收益"符号可以不同——这是本课要点之一。')

    # ---------- 5.4 多种子敏感性实验：基准42 + 额外9个种子 ----------
    print('\n[多种子实验] 固定超参数，仅换随机种子（初始化与探索路径不同），重新训练并评估样本外……')
    seed_rows, seed_sharpes, seed_flat = [], [], []
    extra_seeds = list(range(1, 10))
    all_seeds = [SEED] + extra_seeds
    seed_stats = {}

    def eval_seed_positions(pos_tr, pos_te):
        """对一个仓位序列算绩效；并检测"全空仓"（躺平解：收益恒0，夏普无定义，统计按0计）。"""
        st_tr = perf_stats(strategy_returns(pos_tr, env.ret_next[0:n_tr]))
        st_te = perf_stats(strategy_returns(pos_te, env.ret_next[n_tr:n - 1]))
        return st_tr, st_te, pos_tr.sum() == 0, pos_te.sum() == 0

    seed_stats[SEED] = eval_seed_positions(strategies['Q学习策略(种子42)']['训练段(2010-2019)'],
                                           strategies['Q学习策略(种子42)']['样本外(2020起)'])
    for sd in extra_seeds:
        Q_s, _, _ = q_learning_train(env, *SEG['训练段(2010-2019)'], seed=sd, record_curve=False)
        pos_tr = greedy_positions(env, Q_s, 0, n_tr - 1)
        pos_te = greedy_positions(env, Q_s, n_tr, n - 2)
        seed_stats[sd] = eval_seed_positions(pos_tr, pos_te)
        st_tr, st_te, flat_tr, flat_te = seed_stats[sd]
        seed_rows.append({'数据段': '训练段(2010-2019)', '策略': 'Q学习策略', '种子': str(sd),
                          '备注': '全空仓(躺平解)' if flat_tr else '', **st_tr})
        seed_rows.append({'数据段': '样本外(2020起)', '策略': 'Q学习策略', '种子': str(sd),
                          '备注': '全空仓(躺平解,夏普无定义)' if flat_te else '', **st_te})
    for sd in all_seeds:
        st_tr, st_te, flat_tr, flat_te = seed_stats[sd]
        sh_te = 0.0 if (flat_te and np.isnan(st_te['夏普比率'])) else st_te['夏普比率']
        sh_tr_s = '  n/a ' if (flat_tr and np.isnan(st_tr['夏普比率'])) else f"{st_tr['夏普比率']:+.2f}"
        sh_te_s = '+0.00' if (flat_te and np.isnan(st_te['夏普比率'])) else f"{st_te['夏普比率']:+.2f}"
        seed_sharpes.append(sh_te)
        seed_flat.append(flat_te)
        print(f'  种子 {sd:>2d}：训练段夏普 {sh_tr_s}{"（全空仓）" if flat_tr else "        "}'
              f'  →  样本外夏普 {sh_te_s}{"（全空仓：从未建仓，夏普无定义，按0计）" if flat_te else ""}')
    sh_arr = np.array(seed_sharpes)
    n_flat = int(np.sum(seed_flat))
    bh_te = [m for m in metrics_rows if m['数据段'] == '样本外(2020起)' and m['策略'] == '买入持有'][0]['夏普比率']
    rule_te = [m for m in metrics_rows if m['数据段'] == '样本外(2020起)' and m['策略'] == '状态规则(动量+波动)'][0]['夏普比率']
    print(f'  → 10个种子中 {n_flat} 个学成"全空仓"：躺平也是Q-learning的一个合法解（零初始化+argmax平手偏向空仓）')
    print(f'  → 样本外夏普（全空仓按0计）：min={sh_arr.min():+.2f}  中位={np.median(sh_arr):+.2f}  '
          f'max={sh_arr.max():+.2f}（标准差 {sh_arr.std(ddof=1):.2f}）')
    if n_flat < len(sh_arr):
        nf = sh_arr[~np.array(seed_flat)]
        print(f'  → 仅看真正建仓的 {len(nf)} 个种子：min={nf.min():+.2f}  中位={np.median(nf):+.2f}  max={nf.max():+.2f}')
    print(f'  → 对照：买入持有夏普 {bh_te:+.2f}，状态规则夏普 {rule_te:+.2f}'
          f'——同一个算法，换个种子结论可以从"跑赢"变成"躺平"再到"跑输"')

    # ---------- 5.5 保存 metrics.csv ----------
    pd.DataFrame(metrics_rows + seed_rows).to_csv(
        os.path.join(OUT_DIR, 'metrics.csv'), index=False, encoding='utf-8-sig')

    # ================= 绘图 =================
    # ---- 图01：训练过程奖励曲线 ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, N_EPISODES + 1), curve_explore, color='#999999', lw=1, label='每轮探索总奖励（含ε随机动作）')
    ax.plot(range(1, N_EPISODES + 1), curve_greedy, color='#C44E52', lw=2, label='每轮贪心评估总奖励（无探索）')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('训练轮数（episode）')
    ax.set_ylabel('训练段累计奖励（对数收益−成本）')
    ax.set_title('图01 Q-learning 训练过程：剧烈震荡而非平滑收敛（金融奖励信噪比极低，恰为真实情况）')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, '01_训练过程奖励曲线.png'), dpi=150); plt.close(fig)

    # ---- 图02：Q表热力图（学到什么策略） ----
    action_grid = np.array([[np.argmax(Q[m * 3 + v]) for v in range(3)] for m in range(3)])
    diff_grid = np.array([[Q[m * 3 + v, 1] - Q[m * 3 + v, 0] for v in range(3)] for m in range(3)])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    axes[0].imshow(action_grid, cmap=ListedColormap(['#4C72B0', '#C44E52']), vmin=0, vmax=1, aspect='auto')
    for m in range(3):
        for v in range(3):
            s = m * 3 + v
            axes[0].text(v, m, f'{A_NAMES[int(action_grid[m, v])]}\nQ空={Q[s,0]:+.4f}\nQ满={Q[s,1]:+.4f}',
                         ha='center', va='center', fontsize=9, color='white')
    axes[0].set_xticks(range(3), VOL_NAMES); axes[0].set_yticks(range(3), MOM_NAMES)
    axes[0].set_xlabel('20日波动档位'); axes[0].set_ylabel('20日动量档位')
    axes[0].set_title('每个状态的最终动作选择（蓝=空仓，红=满仓）')
    axes[1].imshow(diff_grid, cmap='RdBu_r', vmin=-np.abs(diff_grid).max(), vmax=np.abs(diff_grid).max(), aspect='auto')
    for m in range(3):
        for v in range(3):
            axes[1].text(v, m, f'{diff_grid[m, v]:+.4f}', ha='center', va='center', fontsize=10)
    axes[1].set_xticks(range(3), VOL_NAMES); axes[1].set_yticks(range(3), MOM_NAMES)
    axes[1].set_xlabel('20日波动档位'); axes[1].set_ylabel('20日动量档位')
    axes[1].set_title('Q值差 Q满仓−Q空仓（正=偏满仓，负=偏空仓）')
    fig.suptitle('图02 Q表热力图：智能体到底学到了什么（训练段 2010-2019）', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(OUT_DIR, '02_Q表热力图.png'), dpi=150); plt.close(fig)

    # ---- 图03：训练段/样本外净值对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    colors = {'Q学习策略(种子42)': '#C44E52', '状态规则(动量+波动)': '#55A868', '买入持有': '#7F7F7F'}
    for ax, seg_name in zip(axes, SEG.keys()):
        for s_name, c in colors.items():
            d, navv = nav_curves[(seg_name, s_name)]
            ax.plot(d, navv, color=c, lw=1.6, label=s_name)
        ax.axhline(1.0, color='k', lw=0.5)
        ax.set_title(f'{seg_name}（Q表在左图数据上训练）' if '训练' in seg_name else '样本外测试（2020起，Q表不再更新）')
        ax.set_ylabel('净值'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('图03 Q策略 vs 状态规则 vs 买入持有：训练段与样本外净值', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(OUT_DIR, '03_样本外净值对比.png'), dpi=150); plt.close(fig)

    # ---- 图04：多种子样本外夏普分布 ----
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bar_colors = ['#C44E52' if sd == SEED else ('#B0B0B0' if fl else '#8DA0CB')
                  for sd, fl in zip(all_seeds, seed_flat)]
    ax.bar([str(sd) for sd in all_seeds], sh_arr, color=bar_colors)
    ax.axhline(bh_te, color='#7F7F7F', ls='--', lw=1.5, label=f'买入持有夏普={bh_te:+.2f}')
    ax.axhline(rule_te, color='#55A868', ls=':', lw=1.5, label=f'状态规则夏普={rule_te:+.2f}')
    ax.axhline(0, color='k', lw=0.5)
    ax.text(0.02, 0.96, f'10个种子：min={sh_arr.min():+.2f}  中位={np.median(sh_arr):+.2f}  '
                        f'max={sh_arr.max():+.2f}  标准差={sh_arr.std(ddof=1):.2f}\n'
                        f'其中 {n_flat} 个种子学成"全空仓"（躺平解，夏普按0计）',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='#FFF3CD', ec='#CCCC00'))
    handles = [Patch(fc='#C44E52', label='基准种子42'),
               Patch(fc='#8DA0CB', label='其他种子（正常建仓）'),
               Patch(fc='#B0B0B0', label='其他种子（全空仓，夏普按0计）')]
    handles += ax.get_legend_handles_labels()[0]
    ax.set_xlabel('随机种子'); ax.set_ylabel('样本外(2020起)夏普比率')
    ax.set_title('图04 多种子实验：换初始化/探索路径，结论从"跑赢"到"躺平"再到"跑输"——RL策略的种子敏感性')
    ax.legend(handles=handles, fontsize=8.5, loc='upper right'); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, '04_多种子样本外夏普分布.png'), dpi=150); plt.close(fig)

    # ---- 图05：状态覆盖对比（训练 vs 样本外） ----
    labels = [f'{MOM_NAMES[m][:2]}·{VOL_NAMES[v][:2]}' for m in range(3) for v in range(3)]
    x = np.arange(N_STATE)
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.bar(x - 0.2, 100 * cnt_tr / n_tr, width=0.4, color='#4C72B0', label='训练段(2010-2019)占比')
    ax.bar(x + 0.2, 100 * cnt_te / max(n - n_tr, 1), width=0.4, color='#DD8452', label='样本外(2020起)占比')
    ax.set_xticks(x, labels, rotation=45, ha='right')
    ax.set_xlabel('状态（动量档·波动档）'); ax.set_ylabel('出现频率 (%)')
    ax.set_title('图05 状态覆盖对比：样本外的状态分布相对训练段发生了漂移（非平稳的直接证据）')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, '05_状态覆盖对比.png'), dpi=150); plt.close(fig)

    # ================= 收尾汇总 =================
    data_tag = '合成数据（缓存缺失兜底）' if is_synth else '真实数据（data/cache 缓存）'
    print(f'''
==================== 结果汇总（{data_tag}）====================
[训练段 2010-2019]
  Q学习策略(种子42)：{fmt_stats(perf_stats(strategy_returns(strategies['Q学习策略(种子42)']['训练段(2010-2019)'], env.ret_next[0:n_tr])))}
  状态规则        ：{fmt_stats(perf_stats(strategy_returns(strategies['状态规则(动量+波动)']['训练段(2010-2019)'], env.ret_next[0:n_tr])))}
  买入持有        ：{fmt_stats(perf_stats(strategy_returns(strategies['买入持有']['训练段(2010-2019)'], env.ret_next[0:n_tr])))}
[样本外 2020起]
  Q学习策略(种子42)：{fmt_stats(perf_stats(strategy_returns(strategies['Q学习策略(种子42)']['样本外(2020起)'], env.ret_next[n_tr:n - 1])))}
  状态规则        ：{fmt_stats(perf_stats(strategy_returns(strategies['状态规则(动量+波动)']['样本外(2020起)'], env.ret_next[n_tr:n - 1])))}
  买入持有        ：{fmt_stats(perf_stats(strategy_returns(strategies['买入持有']['样本外(2020起)'], env.ret_next[n_tr:n - 1])))}
[多种子敏感性] 10个种子中 {n_flat} 个学成全空仓（夏普按0计）；样本外夏普 min={sh_arr.min():+.2f} / 中位={np.median(sh_arr):+.2f} / max={sh_arr.max():+.2f}
[输出目录] {OUT_DIR}
[总耗时] {time.time() - t0:.1f} 秒
''')
    return 0


if __name__ == '__main__':
    sys_exit_code = main()
    raise SystemExit(sys_exit_code)
