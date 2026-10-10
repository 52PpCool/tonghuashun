# -*- coding: utf-8 -*-
"""
36 · 数据窥探检验：White's Reality Check
=========================================

一句话：我从 26 组双均线参数里挑了样本内最好的那个，它的超额收益"显著"吗？
朴素 t 检验会给它打一个偏乐观的分数——因为那个分数没算"它是 26 选 1 的赢家"。
White (2000) 的 Reality Check 用 stationary bootstrap 把"挑最好"这一步也搬进
零假设分布，得到诚实的 p 值：最好成绩减去"最好的运气"还剩多少。

课程设计（对应 README）：
  1) 背景与数据：26 组双均线参数在上证 2010-2026 日线上各跑一遍（含换手成本），
     记录每组"相对买入持有"的日超额收益序列；
  2) 朴素检验的谬误：对样本内最优参数单独做 t 检验，并展示 26 选 1 让它失真；
  3) 本课灵魂：stationary bootstrap（Politis-Romano，平均块长 l=5）+ max 统计量
     => Reality Check p 值；
  4) 校准实验：26 条同换手率的纯随机策略过同一套 RC 流程，应当"不显著"；
  5) 输出 4 张图 + rc_results.csv。

单文件自包含，随机种子 42，总运行约 10 秒。
数据：../data/cache/index_000001_上证指数.csv（真实日K，2010 起）。
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无显示器环境出图
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

# ---------------- 全局配置 ----------------
SEED = 42                  # 全局随机种子（可复现）
B = 1000                   # bootstrap 重采样次数（教学取值，兼顾耗时与 p 值精度 1/(B+1)≈0.001）
BLOCK_LEN = 5              # stationary bootstrap 平均块长 l（Politis-Romano）
COST_SIDE = 3e-4           # 单边换手成本：万3（双边=一买一卖各收一次，完整回合 0.06%）
ANN = 252                  # 年化因子（与课16口径一致）
FAST_LIST = [5, 10, 20, 40, 60]           # 快线候选
SLOW_LIST = [20, 60, 80, 120, 180, 250]   # 慢线候选（快<慢才有效）

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "cache" / "index_000001_上证指数.csv"
OUT_DIR = ROOT / "outputs"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 1. 数据与策略超额收益 =================
def load_close() -> pd.Series:
    """读取上证指数日K（2010 起），返回收盘价序列。"""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], encoding="utf-8-sig").set_index("date")
    return df["close"].astype(float)


def build_excess(close: pd.Series):
    """
    对全部有效参数组合（快<慢，共 26 组）计算"相对买入持有"的日超额收益序列。

    定义（第 t 日）：
        超额_t = (pos_t - 1) * ret_t - COST_SIDE * |pos_t - pos_{t-1}|
      - pos_t ∈ {0,1}：由 t-1 日收盘的均线状态决定 t 日持仓（shift(1)，防未来函数）；
      - pos_t = 1（满仓）：与买入持有同涨跌，超额 ≈ -换手成本；
      - pos_t = 0（空仓）：超额 = -ret_t - 成本（躲过下跌为正、踏空上涨为负）。
    成本按"每次仓位变化 1 个单位 = 单边一次"收取，双边（一买一卖）共 0.06%。

    返回：
      E   : T×26 日超额收益矩阵（公共有效窗口：截去最长慢线 250 日的预热期，
            保证 26 列同窗口、无缺失，RC 需要矩形数据面板）；
      POS : 同窗口的 26 列持仓矩阵（供校准实验匹配换手率）。
    """
    ret = close.pct_change()
    e_cols, pos_cols = {}, {}
    for f in FAST_LIST:
        for s in SLOW_LIST:
            if f >= s:          # 无效组合：快线不慢于慢线
                continue
            ma_f = close.rolling(f).mean()
            ma_s = close.rolling(s).mean()
            pos = (ma_f > ma_s).astype(float).shift(1)   # 今日持仓 = 昨日信号
            e = (pos - 1.0) * ret - COST_SIDE * pos.diff().abs()
            e_cols[f"{f}/{s}"] = e
            pos_cols[f"{f}/{s}"] = pos
    E = pd.DataFrame(e_cols).iloc[max(SLOW_LIST) + 1:]   # 对齐公共有效窗口
    POS = pd.DataFrame(pos_cols).loc[E.index]
    return E, POS


# ================= 2. 朴素检验（对照组：它为什么不可信） =================
def naive_tests(E: pd.DataFrame) -> pd.DataFrame:
    """
    对每个策略单独做 t 检验，H0: 均值超额 = 0（这就是"朴素检验"）。
    它对"最优者"失真的原因：最优者是 26 选 1 的赢家，其 t 统计量天然被
    选择过程抬高（详见 README"朴素t检验为何失效"）。
    """
    rows = []
    for name, e in E.items():
        r = stats.ttest_1samp(e.to_numpy(), 0.0)
        rows.append({"参数": name,
                     "日均超额_bp": e.mean() * 1e4,
                     "年化超额_%": e.mean() * ANN * 100,
                     "朴素t": float(r.statistic),
                     "朴素p": float(r.pvalue)})
    return pd.DataFrame(rows).set_index("参数")


# ================= 3. 本课灵魂：stationary bootstrap 与 Reality Check =================
def stationary_bootstrap_idx(rng: np.random.Generator, T: int, l: int) -> np.ndarray:
    """
    Politis & Romano (1994) stationary bootstrap 的向量化实现。

    直觉：把历史切成一串"随机长度的块"拼回 T 天——
      - 每块长度 ~ Geometric(p=1/l)（平均 l 天；l=5 约保留一周内的短期依赖）；
      - 每块起点在 T 天中均匀随机（环形拼接，允许环绕到开头）。
    返回长度为 T 的重采样索引数组（vec：一次生成全部块，无逐日 Python 循环）。
    """
    p = 1.0 / l
    lens = rng.geometric(p, size=int(T / l * 1.5) + 8)
    while lens.sum() < T:                       # 极小概率块总数不足，补一批
        lens = np.append(lens, rng.geometric(p, size=64))
    starts = rng.integers(0, T, size=lens.size)  # 每块起点均匀随机
    # 把各块拼接成一条索引带：块内偏移 = 全局位置 - 该块在输出中的起始位置
    block_start_out = np.repeat(np.cumsum(lens) - lens, lens)
    within = np.arange(block_start_out.size) - block_start_out
    idx = (np.repeat(starts, lens) + within) % T  # 环形取模
    return idx[:T]


def reality_check(E: pd.DataFrame, seed: int = SEED, B: int = B, l: int = BLOCK_LEN) -> dict:
    """
    White (2000) Reality Check。

    检验统计量（观测值）：
        obs = max_j mean(e_j)         —— 26 个策略里样本内均值超额最大的那个
    零假设分布（"最好的运气能有多好"）：
        把每列减去自身样本均值（强设 H0: 真实均值 = 0），做 B 次 stationary
        bootstrap，每次取 max_j（bootstrap 均值）=> boot_max 的经验分布。
    p 值：
        p_RC = (1 + #{boot_max >= obs}) / (B + 1)
        （分子分母各加 1 是 Davison-Hinkley 惯例，避免 p=0 并保持精确覆盖。）

    关键实现要点（等价性假设）：重采样对象是"策略超额收益序列"本身，而不是
    在重采样价格上重跑策略——持仓是历史的确定函数，重采样日收益时同步重排
    各策略的超额序列即可，这正是 White (2000) 的标准实现；局限见 README。
    """
    X = E.to_numpy()                     # T×J
    T, J = X.shape
    mu = X.mean(axis=0)                  # 各策略样本内均值超额
    Xc = X - mu                          # H0 中心化：每列减自身样本均值
    rng = np.random.default_rng(seed)
    boot_means = np.empty((B, J))
    for b in range(B):                   # ---- 重采样循环（README"代码怎么读"锚点）----
        idx = stationary_bootstrap_idx(rng, T, l)
        boot_means[b] = Xc[idx].mean(axis=0)
    boot_max = boot_means.max(axis=1)    # ---- max 统计量：每次"26选1"的最好成绩 ----
    j_best = int(mu.argmax())
    obs = float(mu[j_best])
    p_rc = (1 + int((boot_max >= obs).sum())) / (B + 1)   # ---- p 值计算 ----
    # 各策略均值在"运气冠军"分布中的百分位（最优策略处即 RC p 值的同口径）
    rank_pct = 100.0 * np.array([(boot_max >= m).mean() for m in mu])
    return {"mu": mu, "boot_means": boot_means, "boot_max": boot_max,
            "obs": obs, "j_best": j_best, "p_rc": p_rc,
            "rank_pct": rank_pct, "T": T, "J": J}


# ================= 4. 校准实验：26 条纯随机策略 =================
def random_excess(ret_win: np.ndarray, n_switch: np.ndarray,
                  rng: np.random.Generator) -> pd.DataFrame:
    """
    生成 J 条"纯随机策略"的日超额收益矩阵（列与真实策略一一对应）。

    与真实策略唯一的共同点：换手率相同（第 j 条随机策略的换仓次数 =
    第 j 个双均线策略的换仓次数），起始仓位随机、换仓日均匀随机、
    持仓仍是 {0,1}。因此它们对买入持有的期望超额 ≈ -(踏空机会成本) - 成本，
    全部不含有任何真实信息——它们是 RC 流程的"阴性对照"。
    """
    T = ret_win.size
    cols = {}
    for j, nsw in enumerate(n_switch):
        nsw = int(nsw)
        level0 = int(rng.integers(0, 2))                 # 起始仓位随机 0/1
        pos = np.full(T, float(level0))
        if nsw > 0:
            days = np.sort(rng.choice(np.arange(1, T), size=nsw, replace=False))
            signs = (1 - 2 * level0) * (-1.0) ** np.arange(nsw)   # 交替翻转 0<->1
            delta = np.zeros(T)
            delta[days] = signs
            pos = level0 + np.cumsum(delta)
            trade = np.zeros(T)
            trade[days] = 1.0                            # 每个换仓日：单边一次
        else:
            trade = np.zeros(T)
        e = (pos - 1.0) * ret_win - COST_SIDE * trade
        cols[f"随机{j + 1:02d}"] = e
    return pd.DataFrame(cols)


# ================= 5. 画图 =================
def plot_heatmap(res_df: pd.DataFrame, best_label: str) -> None:
    """图01：26 组参数的年化超额收益热力图（行=快线、列=慢线，无效组合灰色）。"""
    M = np.full((len(FAST_LIST), len(SLOW_LIST)), np.nan)
    for (f, s), v in {tuple(map(int, k.split("/"))): v
                      for k, v in res_df["年化超额_%"].items()}.items():
        M[FAST_LIST.index(f), SLOW_LIST.index(s)] = v
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdYlGn", aspect="auto")
    im.cmap.set_bad("lightgray")
    fig.colorbar(im, ax=ax, label="年化超额收益（%，相对买入持有，含成本）")
    for i in range(len(FAST_LIST)):
        for j in range(len(SLOW_LIST)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "×", ha="center", va="center", color="gray")
            else:
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=10, color="black")
    f_b, s_b = map(int, best_label.split("/"))
    ax.add_patch(Rectangle((SLOW_LIST.index(s_b) - 0.5, FAST_LIST.index(f_b) - 0.5), 1, 1,
                           fill=False, ec="red", lw=2.6))
    ax.set_xticks(range(len(SLOW_LIST)), SLOW_LIST)
    ax.set_yticks(range(len(FAST_LIST)), FAST_LIST)
    ax.set_xlabel("慢线（日）")
    ax.set_ylabel("快线（日）")
    ax.set_title(f"双均线参数网格的年化超额收益（上证 2010–2026，单边万3成本）\n"
                 f"有效组合 {len(res_df)} 个（快<慢，灰格无效）；红框 = 样本内最优 {best_label}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_25组参数超额收益热力图.png", dpi=150)
    plt.close(fig)


def plot_p_compare(res_df: pd.DataFrame, best_label: str) -> None:
    """图02：每个策略的朴素 p 值 vs RC 结论——'矮子里的将军'一览。"""
    d = res_df.sort_values("年化超额_%", ascending=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x, d["朴素p"], width=0.62, color="#9ecae1", label="朴素 t 检验 p 值（单独检验，被26选1美化）")
    ax.plot(x, 100.0 - d["RC下秩_%"], "o", color="#d95f02", ms=6,
            label="RC 超越比例 = 100% − RC下秩（均值超过「运气冠军」零分布的比例）")
    ax.axhline(0.05, color="red", ls="--", lw=1.2, label="0.05 显著性线")
    bi = list(d.index).index(best_label)
    ax.annotate(f"样本内最优 {best_label}\n朴素p={d.loc[best_label, '朴素p']:.3f}  "
                f"RC超越={100 - d.loc[best_label, 'RC下秩_%']:.1f}%",
                xy=(bi, d.loc[best_label, "朴素p"]),
                xytext=(bi + 1.0, 1.05), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_yscale("log")
    ax.set_ylim(0.02, 9.0)
    ax.set_xticks(x, d.index, rotation=60, fontsize=8)
    ax.set_ylabel("p 值 / 超越比例（对数轴）")
    ax.set_title("朴素 p 值 vs Reality Check（按年化超额从高到低）\n"
                 "橙点越低 = 该策略均值在「26选1最好运气」分布中排位越靠后 = 越像纯运气")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_朴素p值vs RC p值对比.png", dpi=150)
    plt.close(fig)


def plot_boot_dist(rc: dict, best_label: str) -> None:
    """
    图03：最优策略均值超额 vs RC 零分布。
    同时画出"错误做法"的零分布（只对最优策略自己做 bootstrap）作对照：
    它漏掉了"26 选 1"这一步，明显偏窄偏左。
    """
    ann = ANN * 100.0
    wrong = rc["boot_means"][:, rc["j_best"]] * ann   # 错误：只 boot 最优策略自己
    right = rc["boot_max"] * ann                      # 正确：每次 bootstrap 取 26 个的 max
    obs = rc["obs"] * ann
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.hist(wrong, bins=40, alpha=0.55, color="#9ecae1",
            label="错误零分布：只对最优策略自己 bootstrap（漏掉26选1）")
    ax.hist(right, bins=40, alpha=0.55, color="#fdb462",
            label="正确零分布：每次 bootstrap 取 max_j（Reality Check）")
    ax.axvline(obs, color="red", lw=2.2, label=f"观测：最优 {best_label} 年化超额 {obs:.2f}%")
    ax.set_xlabel("年化超额收益（%）")
    ax.set_ylabel("bootstrap 频数")
    ax.set_title(f"「最好的运气能有多好」：RC 零分布 vs 观测最优\n"
                 f"RC p 值 = {rc['p_rc']:.3f}（B={B}，平均块长 l={BLOCK_LEN}）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_bootstrap分布图.png", dpi=150)
    plt.close(fig)


def plot_calibration(rand_stats: pd.DataFrame, rand_rc: dict,
                     real_stats: pd.DataFrame, real_rc: dict, best_label: str) -> None:
    """图04：校准实验——同一套流程喂给纯随机策略，RC 应当'不显著'。"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.0))
    # 左：26 条随机策略的朴素 p 值直方图（若全是噪声，p 应近似均匀分布）
    p = rand_stats["朴素p"].to_numpy()
    ax1.hist(p, bins=np.arange(0, 1.05, 0.1), color="#b3cde3", edgecolor="white")
    ax1.axhline(len(p) * 0.1, color="gray", ls=":", lw=1.4, label="均匀分布期望（每桶2.6个）")
    n_sig = int((p < 0.05).sum())
    ax1.set_title(f"26 条纯随机策略的朴素 p 值分布\n"
                  f"p<0.05 的有 {n_sig} 个（全噪声下期望约 {len(p) * 0.05:.1f} 个）")
    ax1.set_xlabel("朴素 p 值")
    ax1.set_ylabel("策略个数")
    ax1.legend(fontsize=9)
    # 中：RC 零分布的校准——真实参数族 vs 随机策略族的「运气冠军」分布
    ann = ANN * 100.0
    ax2.hist(real_rc["boot_max"] * ann, bins=40, alpha=0.55, color="#fdb462",
             label=f"真实 26 组参数的 RC 零分布（均值 {real_rc['boot_max'].mean() * ann:.2f}%）")
    ax2.hist(rand_rc["boot_max"] * ann, bins=40, alpha=0.55, color="#b3cde3",
             label=f"随机 26 条策略的 RC 零分布（均值 {rand_rc['boot_max'].mean() * ann:.2f}%）")
    ax2.axvline(real_rc["obs"] * ann, color="red", lw=2,
                label=f"真实最优观测 {real_rc['obs'] * ann:.2f}%")
    ax2.set_xlabel("年化超额（%）")
    ax2.set_ylabel("bootstrap 频数")
    ax2.set_title("RC 零分布的校准：真实参数族 vs 随机族\n"
                  "（两条分布都在 0 右侧：纯噪声下「26选1」平均就是正的）")
    ax2.legend(fontsize=8)
    # 右：真实最优 vs 随机最优 的朴素 p / RC p 对比
    labels = [f"真实最优\n{best_label}\n朴素p", f"真实最优\n{best_label}\nRC p",
              f"随机最优\n{rand_stats.index[int(rand_rc['mu'].argmax())]}\n朴素p",
              f"随机最优\n{rand_stats.index[int(rand_rc['mu'].argmax())]}\nRC p"]
    vals = [real_stats.loc[best_label, "朴素p"], real_rc["p_rc"],
            float(rand_stats["朴素p"].iloc[int(rand_rc["mu"].argmax())]), rand_rc["p_rc"]]
    colors = ["#9ecae1", "#d95f02", "#9ecae1", "#d95f02"]
    bars = ax3.bar(range(4), vals, color=colors)
    ax3.axhline(0.05, color="red", ls="--", lw=1.2, label="0.05 显著性线")
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width() / 2, v * 1.15 + 1e-4, f"{v:.3f}",
                 ha="center", fontsize=10)
    ax3.set_yscale("log")
    ax3.set_ylim(1e-4, 1.3)
    ax3.set_xticks(range(4), labels, fontsize=9)
    ax3.set_ylabel("p 值（对数轴）")
    ax3.set_title("同一套检验，喂真策略 vs 喂随机策略\n"
                  "朴素 p 都不显著、RC p 都不显著：机器工作正常")
    ax3.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_校准实验.png", dpi=150)
    plt.close(fig)


# ================= 主流程 =================
def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)  # 例行播种（本课全部随机数走 default_rng(SEED)，此处仅兜底）

    print("=" * 72)
    print("36 · 数据窥探检验：White's Reality Check（种子 42，B=%d，块长 l=%d）" % (B, BLOCK_LEN))
    print("=" * 72)

    # ---- 1. 数据与 26 组策略超额收益 ----
    close = load_close()
    E, POS = build_excess(close)
    T, J = E.shape
    years = (E.index[-1] - E.index[0]).days / 365.25
    print(f"\n[1] 数据：{close.index[0].date()} ~ {close.index[-1].date()}，共 {len(close)} 个交易日")
    print(f"    公共有效窗口：{E.index[0].date()} ~ {E.index[-1].date()}，T={T} 日，{J} 组参数")
    bh_ann = (close.iloc[-1] / close.iloc[0]) ** (ANN / len(close)) * 100 - 100
    print(f"    成本：单边万3（双边一回合 0.06%）；买入持有年化 {bh_ann:.2f}%")
    C = E.corr().to_numpy()
    iu = np.triu_indices_from(C, k=1)
    print(f"    策略间两两相关（超额收益）：均值 {C[iu].mean():.3f}，范围 [{C[iu].min():.3f}, {C[iu].max():.3f}]"
          f"（高相关 => '26选1'的有效独立个数远小于26，见 README 与 Bonferroni 的对比）")

    # ---- 2. 朴素检验 ----
    res = naive_tests(E)

    # ---- 3. Reality Check ----
    rc = reality_check(E)
    best_label = E.columns[rc["j_best"]]
    res["RC下秩_%"] = rc["rank_pct"]
    res["换手_次每年"] = POS.diff().abs().sum() / years
    res["样本内最优"] = res.index == best_label
    res = res.sort_values("年化超额_%", ascending=False)

    print(f"\n[2] 样本内最优参数：{best_label}（年化超额 {res.loc[best_label, '年化超额_%']:.2f}%，"
          f"日均 {res.loc[best_label, '日均超额_bp']:.2f} bp，换手 {res.loc[best_label, '换手_次每年']:.1f} 次/年）")
    print(f"    朴素 t 检验：t = {res.loc[best_label, '朴素t']:.3f}，p = {res.loc[best_label, '朴素p']:.4f}"
          f"   <-- 单独检验'最优者'，读数本身已被 26 选 1 扭曲")
    print(f"    Reality Check：p = {rc['p_rc']:.4f}   <-- 把'26选1'算进零假设之后")
    ann = ANN * 100.0
    print(f"    「运气冠军」零分布（H0 下 26 选 1 的最好成绩）：均值 {rc['boot_max'].mean() * ann:.2f}%/年，"
          f"中位数 {np.median(rc['boot_max']) * ann:.2f}%/年")
    print(f"    观测最优排在零分布的第 {(boot_pct := 100 * (rc['boot_max'] < rc['obs']).mean()):.1f} 百分位"
          f"（低于 {(100 - boot_pct):.1f}% 的纯运气冠军）")
    n_sig = int((res["朴素p"] < 0.05).sum())
    p_any = 1 - 0.95 ** J
    print(f"    朴素 p<0.05 的参数组：{n_sig}/{J} 个（若 26 组全是噪声，期望约 {J * 0.05:.1f} 个假阳性）")
    print(f"    朴素检验的谬误（解析）：即使 26 组全是废物，'至少挖出一个 p<0.05'的概率 = "
          f"1-0.95^{J} = {p_any:.1%}   <-- 这就是数据窥探偏差")
    # 口径对照：日均超额（RC 检验的对象）vs CAGR 超额——两者符号可以相反（波动拖累）
    ret_win_s = close.pct_change().loc[E.index]
    bh_cagr = (1.0 + ret_win_s).prod() ** (ANN / T) - 1
    strat_daily = E[best_label] + ret_win_s          # 策略净日收益 = 超额 + 基准
    strat_cagr = (1.0 + strat_daily).prod() ** (ANN / T) - 1
    drag = (ret_win_s.var() - strat_daily.var()) / 2 * ANN * 100
    print(f"    口径对照（公共窗口）：CAGR 口径 {best_label} {strat_cagr * 100:+.2f}% vs 买入持有 "
          f"{bh_cagr * 100:+.2f}%（差 {(strat_cagr - bh_cagr) * 100:+.2f} pp/年）；"
          f"换成'日均超额'口径差距扩大到 {res.loc[best_label, '年化超额_%']:+.2f}%/年——其中约 {drag:.2f} pp "
          f"是波动拖累的口径差（B&H 波动更大，其算术均值被 σ²/2 抬得更多）。RC 检验的是日均超额口径")

    # ---- 4. 校准实验：同换手率的 26 条纯随机策略 ----
    ret_win = close.pct_change().loc[E.index].to_numpy()
    n_switch = POS.diff().abs().sum().to_numpy()          # 每个真实策略的换仓次数
    rng_rand = np.random.default_rng(SEED)                # 随机持仓的发生器（种子 42）
    E_rand = random_excess(ret_win, n_switch, rng_rand)
    rand_stats = naive_tests(E_rand)
    rand_rc = reality_check(E_rand)
    rand_best = E_rand.columns[int(rand_rc["mu"].argmax())]
    rand_sig = int((rand_stats["朴素p"] < 0.05).sum())
    print(f"\n[3] 校准实验（26 条同换手率纯随机策略）：")
    print(f"    随机最优：{rand_best}，年化超额 {rand_rc['mu'].max() * ANN * 100:.2f}%，"
          f"朴素 p = {float(rand_stats['朴素p'].iloc[int(rand_rc['mu'].argmax())]):.4f}，RC p = {rand_rc['p_rc']:.4f}")
    print(f"    随机族「运气冠军」零分布：均值 {rand_rc['boot_max'].mean() * ann:.2f}%/年，"
          f"中位数 {np.median(rand_rc['boot_max']) * ann:.2f}%/年（与真实参数族对照 => 校准）")
    print(f"    随机策略朴素 p<0.05 的个数：{rand_sig}/{J}（全噪声下期望约 {J * 0.05:.1f} 个）")

    # ---- 5. 落盘：CSV ----
    res.to_csv(OUT_DIR / "rc_results.csv", encoding="utf-8-sig")
    print(f"\n[4] 已写出 {OUT_DIR / 'rc_results.csv'}（{len(res)} 行）")
    print(res[["日均超额_bp", "年化超额_%", "换手_次每年", "朴素t", "朴素p", "RC下秩_%"]]
          .round(3).to_string())

    # ---- 6. 落盘：四张图 ----
    plot_heatmap(res, best_label)
    plot_p_compare(res, best_label)
    plot_boot_dist(rc, best_label)
    plot_calibration(rand_stats, rand_rc, res, rc, best_label)
    print(f"\n[5] 已写出 4 张图到 {OUT_DIR}")

    # ---- 7. 结论（如实） ----
    verdict = ("显著（不是运气）" if rc["p_rc"] < 0.05
               else "无法与运气区分（样本内最优被正式确认可能是'矮子里的将军'）")
    print("\n" + "=" * 72)
    print(f"结论：最优参数 {best_label} 朴素 p = {res.loc[best_label, '朴素p']:.4f}，"
          f"RC p = {rc['p_rc']:.4f} => 在 5% 水平上：{verdict}")
    print("注：26 组参数的均值超额全部为负——2011-2026 的上证上，没有一组双均线"
          "\n    在'日均超额'口径下跑赢买入持有；RC 零分布（纯运气冠军均值 "
          f"{rc['boot_max'].mean() * ann:.2f}%/年）\n    全部高于现实最优，这是比'不显著'更冷的一句话。")
    print(f"总耗时 {time.time() - t0:.1f} 秒")
    print("=" * 72)


if __name__ == "__main__":
    main()
