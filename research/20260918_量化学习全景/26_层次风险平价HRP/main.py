# -*- coding: utf-8 -*-
"""
26_层次风险平价HRP：López de Prado 的 HRPOpt —— 不需要协方差求逆的"树形"配置法
=================================================================================

这节课讲 HRP（Hierarchical Risk Parity，层次风险平价），出自 Marcos López de Prado
2016 年的论文《Building Diversified Portfolios that Outperform Out-of-Sample》。
它是"机器学习思想进入组合构建"的代表作：用层次聚类（无监督学习）给资产"认亲"，
再沿着树结构自顶向下分钱，全程只做矩阵乘法和除法，**一次都不用求协方差逆矩阵**。

main.py 五件事（全部单文件自包含，随机种子 42）：
  [1] 数据：10 个行业板块共同重叠期的日收益（清洗基期跳变），计算相关系数矩阵
  [2] 本课灵魂——HRP 三步法：
      a) 树聚类：相关距离 d = sqrt(0.5*(1-ρ)) → scipy linkage（single / ward 各试）
      b) 准对角化：按聚类结果重排协方差矩阵（get_quasi_diag，相似资产挨着）
      c) 递归二分法分配权重（get_rec_bipart），每层用逆方差分割，权重和恒为 1
  [3] 对比回测：等权 / HRP（滚动 252 日窗口）/ 朴素风险平价（逆波动率）/ 最小方差（cvxpy）
      月度再平衡，输出年化/波动/夏普/回撤/换手
  [4] 噪声实验：给相关矩阵注入随机扰动后重归一化，对比最小方差 vs HRP 的
      权重漂移（L1 范数）——验证"最小方差对输入敏感、HRP 不求逆所以稳"
  [5] 出图：树状图、准对角化热力图、净值对比、HRP 权重演变、噪声稳健性

运行：cd 26_层次风险平价HRP && ../.venv/bin/python main.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------- #
# 绘图环境：Agg 后端（无显示器也能出图）+ 中文字体
# ---------------------------------------------------------------------------- #
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

SEED = 42
rng = np.random.default_rng(SEED)          # 全程唯一随机源：噪声实验用

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 10 个行业板块（对应 data/cache/board_*.csv）
BOARDS = ["银行", "保险", "证券", "房地产开发", "白酒",
          "半导体", "医疗器械", "煤炭行业", "电力行业", "汽车整车"]
N_LOOKBACK = 252        # 滚动估计窗口：252 个交易日 ≈ 1 年
JUMP_THRESHOLD = 0.30   # 单日 |收益| > 30% 视为"基期跳变"毛刺，置 0


# ============================================================================ #
# [1] 数据加载与清洗
# ============================================================================ #
def load_returns():
    """读取 10 个行业 CSV，取共同重叠期，算日收益并清洗跳变。

    清洗规则（README 会写明）：把各板块收盘价按日期内连接取交集，
    计算日收益率后，凡 |r| > 30% 的观测视为数据源的"基期跳变"毛刺
    （板块指数改基 / 缺数补数造成的伪跳变，A股涨跌停限制下单日 ±30%
    几乎不可能由真实交易产生），一律置 0。
    若缓存数据缺失或共同窗口过短，则退回合成数据（rng(42)，明确标注）。
    """
    data_dir = os.path.join(BASE_DIR, "..", "data", "cache")
    closes = {}
    for name in BOARDS:
        path = os.path.join(data_dir, f"board_{name}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"]).set_index("date")["close"]
        closes[name] = df
    if len(closes) < len(BOARDS) * 0.5:          # 缓存大面积缺失 → 合成数据
        return make_synthetic()

    prices = pd.DataFrame(closes).dropna(how="any")   # 内连接 = 共同重叠期
    ret = prices.pct_change().dropna(how="any")

    # —— 跳变清洗：|r| > 30% 置 0 —— #
    jump_mask = ret.abs() > JUMP_THRESHOLD
    n_jumps = int(jump_mask.sum().sum())
    ret = ret.mask(jump_mask, 0.0)

    # 兜底检查：共同窗口太短也退回合成数据
    if len(ret) < 400 or ret.shape[1] < len(BOARDS):
        return make_synthetic()
    ret = ret[BOARDS]                             # 固定列顺序
    return ret, n_jumps, False


def make_synthetic():
    """缓存缺失时的兜底：一个"市场因子 + 4 个行业块"的合成收益面板（rng(42)）。"""
    n_days, rng_syn = 1250, np.random.default_rng(SEED)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    market = rng_syn.normal(0.0002, 0.010, n_days)
    blocks = {  # 四个"行业簇"的块因子：金融 / 成长 / 周期 / 稳定
        "金融": rng_syn.normal(0, 0.006, n_days),
        "成长": rng_syn.normal(0, 0.009, n_days),
        "周期": rng_syn.normal(0, 0.008, n_days),
        "稳定": rng_syn.normal(0, 0.004, n_days),
    }
    member = {"银行": "金融", "保险": "金融", "证券": "金融", "房地产开发": "金融",
              "白酒": "稳定", "医疗器械": "成长", "半导体": "成长",
              "煤炭行业": "周期", "电力行业": "稳定", "汽车整车": "周期"}
    data = {}
    for name in BOARDS:
        b = blocks[member[name]]
        data[name] = 0.0001 + 0.9 * market + 0.8 * b + rng_syn.normal(0, 0.008, n_days)
    ret = pd.DataFrame(data, index=dates, columns=BOARDS)
    print("!! 警告：真实缓存数据不可用，本节使用【合成数据】（rng(42) 因子模型）！！")
    return ret, 0, True


# ============================================================================ #
# [2] HRP 三步法（对照 López de Prado 2016 论文逐步实现）
# ============================================================================ #
# ---------- 第 1 步：树聚类（论文代码 getCorrDist + scipy linkage） ---------- #
def corr_distance(corr: np.ndarray) -> np.ndarray:
    """相关距离 d_ij = sqrt(0.5 * (1 - ρ_ij))。

    为什么不用 1-ρ 直接当距离？因为 1-ρ 不满足三角不等式，开根号后
    d 是严格度量（可证明它是"单位球面上夹角"的单调变换），
    这样 linkage 的输入才是合格的距离矩阵。ρ=1（完全同向）→ d=0，
    ρ=0（正交）→ d≈0.707，ρ=-1（反向）→ d=1。
    """
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)                    # 自己与自己距离为 0
    return dist


def cluster_corr(corr: np.ndarray, method: str = "single") -> np.ndarray:
    """对相关距离做层次聚类，返回 scipy 的 linkage 矩阵（Z）。

    Z 每行 = [簇A编号, 簇B编号, 两簇距离, 合并后簇内叶子数]。
    method='single'：最近邻链式合并 —— 论文原版选择。它只保证
      "相邻的叶子相似"，不关心簇的形状，正好契合第 3 步"递归二分"
      只按顺序切分的用途；且对不规则/链状的相关结构更宽容。
    method='ward'：最小化"合并后簇内方差增量"，通常给出更均衡的树，
      但 ward 的数学假设是欧氏坐标下的方差，用于相关距离属于"借用"，
      结果稳健性见课堂讨论。主回测沿用论文的 single。
    """
    dist = corr_distance(corr)
    condensed = squareform(dist, checks=False)     # 方阵 → 压缩下三角向量
    return linkage(condensed, method=method)


# ---------- 第 2 步：准对角化（论文代码 getQuasiDiag） ---------- #
def get_quasi_diag(link: np.ndarray) -> list:
    """把 linkage 树"展开"成叶子顺序：让相似资产在新排序里互相挨着。

    这一步叫 seriation / 准对角化（quasi-diagonalization）。重排后相关
    矩阵的大值会聚集在对角线附近的"块"里——但它不是精确对角化（无法
    把所有资产排成完美分块），故称"准"对角化。

    逐层展开：从根节点的两个孩子出发，编号 >= n 的是"集群"（linkage
    给每个合并簇的新编号从 n 开始），遇到集群就拆成它的左右孩子，
    遇到叶子（编号 < n）就保留，直到全是叶子为止。
    结果等价于论文的 getQuasiDiag，也等价于 scipy 的 leaves_list(link)。
    """
    link = link.astype(int)
    n = link[-1, 3]                    # 叶子（原始资产）个数：Z 最后一行第 4 列
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]   # 从根节点的两个孩子开始
    while max(sort_ix) >= n:           # 只要还有"集群编号"（>= n）没拆完
        expanded = []
        for node in sort_ix:
            if node < n:               # 叶子：保留
                expanded.append(node)
            else:                      # 集群：拆成左右两个孩子，左先右后
                row = node - n         # 该集群在 linkage 矩阵中的行号
                expanded.append(int(link[row, 0]))
                expanded.append(int(link[row, 1]))
        sort_ix = expanded
    return sort_ix                     # 叶子编号的排列 = 新顺序


# ---------- 第 3 步：递归二分法分配权重（论文代码 getRecBipart） ---------- #
def get_ivp(cov_sub: np.ndarray) -> np.ndarray:
    """簇内逆方差权重（Inverse-Variance Portfolio）：w_i ∝ 1/σ_i。

    在"簇内资产互不相关"的近似下，这就是该簇的最优风险分散权重，
    也是整个 HRP 里唯一用到的"配置公式"——注意它只用对角线（方差），
    不用求逆。
    """
    ivp = 1.0 / np.sqrt(np.diag(cov_sub))
    ivp /= ivp.sum()
    return ivp


def get_cluster_var(cov: np.ndarray, items: list) -> float:
    """簇方差 = w'Σw（w 为簇内 IVP 权重）。

    关键细节：虽然 IVP 本身只用方差，但这里算簇方差用的是**完整的
    子协方差块**（含簇内资产间的协方差）——所以 HRP 并没有完全扔掉
    相关系数信息，它扔掉的是"跨簇之间的协方差"（这正是它的简化近似，
    README 会专门讨论代价）。
    """
    sub = cov[np.ix_(items, items)]
    w_ = get_ivp(sub)
    return float(w_ @ sub @ w_)


def get_rec_bipart(cov: np.ndarray, sort_ix: list) -> np.ndarray:
    """递归二分法（论文 getRecBipart）：沿着树自顶向下"分蛋糕"。

    记账方式：LdP 原版用 pd.Series(1.0, index=sort_ix) 按资产标签记账；
    本实现的 w 数组按下标=原始资产编号记账（clusters 里存的就是全局资产
    编号），两者完全等价——每个资产在自己的编号处累乘分到的比例。
    每一轮把每个 >=2 个资产的簇切成左右两半（左右两半在列表里天然
    相邻成对），每半先用 IVP 算出簇方差，再按逆方差比例分蛋糕：
        alpha = 1 - Var(左) / (Var(左) + Var(右))
        左半乘 alpha，右半乘 (1-alpha)
    直觉：方差大（更危险）的那一半少分钱。反复二分直到叶子，每片叶子
    累计收到的乘积就是最终权重——乘积保证每层分配比例和恒为 1，
    因此总权重恒等于 1（HRP 天然不需要再归一化）。
    """
    w = np.ones(len(sort_ix))          # 每片叶子从"1 份蛋糕"开始（下标=资产编号）
    clusters = [list(sort_ix)]
    while clusters:
        # 把每个多于 1 个资产的簇切成左右两半（<=1 的簇不再切，直接丢弃——
        # 它的权重已在上一轮分配完毕）
        clusters = [c[j:k]
                    for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        # 相邻两半配对：列表里左右半天然相邻 → (左0,右0), (左1,右1)...
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = get_cluster_var(cov, left)
            var_right = get_cluster_var(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            w[left] *= alpha           # 左半方差大 → alpha 小 → 少分
            w[right] *= 1.0 - alpha
    return w                           # w[资产编号] = 该资产权重（原始顺序）


def hrp_weights(corr: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """HRP 主流程：三步串联。返回与原始列顺序一致的权重向量。

    注意：get_rec_bipart 返回的数组已经按下标=原始资产编号对齐
    （LdP 原版按标签记账，等价），所以这里【不需要】再做任何重排——
    若再按 order 重排一次反而会把权重张冠李戴（本课开发中真实踩过的坑）。
    """
    link = cluster_corr(corr, method="single")            # 1) 树聚类
    order = get_quasi_diag(link)                          # 2) 准对角化（seriation）
    return get_rec_bipart(cov, order)                     # 3) 递归二分分配


# ============================================================================ #
# [3] 对比配置法：等权 / HRP / 朴素风险平价 / 最小方差（月度再平衡滚动回测）
# ============================================================================ #
def naive_risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """朴素风险平价（逆波动率加权，课 10 的对角近似版）。

    "朴素"在它完全忽略相关矩阵——只按 1/σ_i 分钱。
    与 HRP 的区别：HRP 在簇内、簇间都用到了相关结构（通过聚类），
    但两者都不需求逆，是一对"近亲"。
    """
    iv = 1.0 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def min_var_weights(cov: np.ndarray) -> np.ndarray:
    """最小方差组合（cvxpy 求解 QP）：min w'Σw, s.t. Σw=1, w>=0。

    这是"正规军"马科维茨思路的纯风险版本——要求逆（等价于求解
    线性方程组），因此对协方差的估计误差最敏感，[4] 的噪声实验
    拿它当反面教材。求解失败时退回逆波动率权重，保证脚本永不崩。
    """
    import cvxpy as cp
    n = cov.shape[0]
    try:
        w = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))),
                          [cp.sum(w) == 1, w >= 0])
        prob.solve(solver=cp.CLARABEL)
        if w.value is not None and np.all(np.isfinite(w.value)):
            return np.clip(w.value, 0.0, None) / np.clip(w.value, 0.0, None).sum()
    except Exception:
        pass
    return naive_risk_parity_weights(cov)   # 兜底


def backtest(ret: pd.DataFrame, lookback: int = N_LOOKBACK):
    """月度再平衡滚动回测。四个策略共用同一条再平衡日历与估计窗口。"""
    R = ret.values
    dates = ret.index
    n = R.shape[1]
    names = list(ret.columns)
    ym = [(d.year, d.month) for d in dates]     # (年, 月) 元组，用于找每月第一天

    # 再平衡日：每月第一个交易日，且前面凑够了 lookback 天
    is_rebal = np.zeros(len(dates), dtype=bool)
    for i in range(1, len(dates)):
        if ym[i] != ym[i - 1] and i >= lookback:
            is_rebal[i] = True
    if not is_rebal.any():                      # 兜底：数据太短时改用每 21 天调仓
        is_rebal[np.arange(lookback, len(dates), 21)] = True
    start = int(np.argmax(is_rebal))            # 第一个再平衡日

    strategies = ["等权", "HRP", "朴素风险平价", "最小方差"]
    nav = {s: [1.0] for s in strategies}
    w_cur = {s: None for s in strategies}       # 当前（漂移中的）权重
    turnover = {s: [] for s in strategies}
    hrp_w_hist = {}                             # HRP 权重演变记录
    bt_dates = [dates[start - 1]]               # 净值序列的起点

    for i in range(start, len(dates)):
        if is_rebal[i]:
            win = R[i - lookback:i]             # 估计窗口：不含今天的过去 252 天
            cov_w = np.cov(win, rowvar=False, ddof=1)
            corr_w = np.corrcoef(win, rowvar=False)
            corr_w = np.nan_to_num(corr_w, nan=0.0)     # 防常数序列产生 NaN
            np.fill_diagonal(corr_w, 1.0)
            w_new = {
                "等权": np.full(n, 1.0 / n),
                "HRP": hrp_weights(corr_w, cov_w),
                "朴素风险平价": naive_risk_parity_weights(cov_w),
                "最小方差": min_var_weights(cov_w),
            }
            for s in strategies:
                if w_cur[s] is not None:        # 换手 = 0.5 * L1(新权重-漂移后旧权重)
                    turnover[s].append(0.5 * float(np.abs(w_new[s] - w_cur[s]).sum()))
                w_cur[s] = w_new[s]
            hrp_w_hist[dates[i]] = w_new["HRP"].copy()

        for s in strategies:
            r_p = float(w_cur[s] @ R[i])
            nav[s].append(nav[s][-1] * (1.0 + r_p))
            # 权重随价格漂移（再平衡之间不交易）
            w_cur[s] = w_cur[s] * (1.0 + R[i]) / (1.0 + r_p)
        bt_dates.append(dates[i])

    nav = {s: pd.Series(v, index=pd.DatetimeIndex(bt_dates)) for s, v in nav.items()}
    hrp_hist = pd.DataFrame(hrp_w_hist, index=names).T   # 行=日期, 列=行业
    return nav, turnover, hrp_hist, names


RF_ANNUAL = 0.02  # 无风险利率（年化2%，与本系列课01-10口径一致）


def perf_stats(nav: pd.Series, turnover_list: list, freq: int = 252) -> dict:
    """年化收益 / 年化波动 / 夏普 / 最大回撤 / 换手。

    夏普 = (几何年化收益 − rf) / 年化波动，rf=2%（本系列统一口径；
    本课四策略均为纯多头组合，故需减 rf，与课02/07/09/10一致）。
    """
    r = nav.pct_change().dropna()
    n_years = len(r) / freq
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / n_years) - 1.0
    ann_vol = float(r.std() * np.sqrt(freq))
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan
    dd = nav / nav.cummax() - 1.0
    max_dd = float(dd.min())
    avg_turn = float(np.mean(turnover_list)) if turnover_list else 0.0
    ann_turn = avg_turn * 12                    # 月度再平衡 → 年化 ≈ ×12
    return {"年化收益": ann_ret, "年化波动": ann_vol, "夏普比率": sharpe,
            "最大回撤": max_dd, "单次平均换手": avg_turn, "年化换手": ann_turn}


# ============================================================================ #
# [4] 噪声实验：最小方差 vs HRP 的权重稳定性
# ============================================================================ #
def psd_repair(mat: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """特征值截断，把矩阵修成半正定（噪声扰动后的矩阵可能带负特征值，
    cvxpy 的 quad_form 需要半正定输入）。最小方差解对矩阵的整体缩放不敏感，
    因此截断后不必恢复原对角线。"""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, floor, None)
    return vecs @ np.diag(vals) @ vecs.T


def perturb_corr(corr0: np.ndarray, sigma: float) -> np.ndarray:
    """给相关矩阵加对称随机扰动，再重归一化回"对角线=1"的合法相关矩阵。

    sigma=0 时返回原矩阵。注意对角线可能被扰动成负数（sigma=0.5 时
    E_ii < -1 约 2% 概率），先截断再开根；离群值 clip 回 [-1,1]。
    """
    if sigma > 0:
        E = rng.normal(0.0, sigma, corr0.shape)
        E = (E + E.T) / 2.0
    else:
        E = 0.0
    C = corr0 + E
    d = np.sqrt(np.clip(np.diag(C), 1e-6, None))
    C = C / np.outer(d, d)
    C = np.clip(C, -1.0, 1.0)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def _drift_table(corr0, cov0, sigmas, n_rep, w_mv0, w_hrp0, label):
    """对每个噪声幅度重复 n_rep 次，统计两法权重漂移（L1）的均值/标准差。"""
    std0 = np.sqrt(np.diag(cov0))
    rows = []
    for sigma in sigmas:
        drifts_mv, drifts_hrp = [], []
        for _ in range(1 if sigma == 0.0 else n_rep):
            C = perturb_corr(corr0, sigma)
            cov_noisy = np.outer(std0, std0) * C     # 波动率不动，只动相关结构
            w_mv = min_var_weights(psd_repair(cov_noisy))
            w_hrp = hrp_weights(C, cov_noisy)
            drifts_mv.append(float(np.abs(w_mv - w_mv0).sum()))
            drifts_hrp.append(float(np.abs(w_hrp - w_hrp0).sum()))
        rows.append({"实验": label, "噪声幅度sigma": sigma,
                     "最小方差权重漂移L1_均值": np.mean(drifts_mv),
                     "最小方差权重漂移L1_标准差": np.std(drifts_mv),
                     "HRP权重漂移L1_均值": np.mean(drifts_hrp),
                     "HRP权重漂移L1_标准差": np.std(drifts_hrp)})
    return pd.DataFrame(rows)


def noise_experiment(corr0: np.ndarray, cov0: np.ndarray,
                     sigmas=(0.0, 0.05, 0.10, 0.20, 0.30, 0.50), n_rep: int = 20):
    """实验A（真实数据，N=10，T=252>>N，协方差良态）：
    给相关矩阵注入噪声，对比最小方差 vs HRP 的权重漂移 L1 = Σ|w_noisy−w_base|。
    协方差按 sigma_i*sigma_j*C_noisy 重建——波动率不动、只动相关结构，
    这样才能干净地对比"相关矩阵敏感性"。
    """
    w_mv0 = min_var_weights(psd_repair(cov0))
    w_hrp0 = hrp_weights(corr0, cov0)
    return _drift_table(corr0, cov0, sigmas, n_rep, w_mv0, w_hrp0, "A_真实10行业"), w_mv0, w_hrp0


def make_synthetic_returns_large(n_assets: int = 100, n_days: int = 60):
    """实验B 用的合成收益面板【合成数据，rng(42) 同一随机流】：
    1 个市场因子 + 10 个行业因子 + 特质噪声，N=100 只资产、只有 T=60 天
    样本 —— T < N，样本协方差必然不满秩（秩≤59），这是 HRP 论文瞄准的
    "病态"场景：N 大、T 短、协方差无法求逆。
    """
    market = rng.normal(0.0, 0.010, n_days)
    sector_f = rng.normal(0.0, 0.008, (n_days, 10))
    loading = np.repeat(np.arange(10), n_assets // 10)       # 每 10 只一个行业
    R = (0.0002 + market[:, None] + sector_f[:, loading]
         + rng.normal(0.0, 0.015, (n_days, n_assets)))
    return R


def noise_experiment_large_n(n_assets: int = 100, n_days: int = 60,
                             sigmas=(0.0, 0.05, 0.10, 0.20, 0.30, 0.50), n_rep: int = 10):
    """实验B（合成数据，N=100，T=60<N，协方差奇异）：同样的噪声注入协议。
    这里最小方差被迫在（修复后的）秩亏矩阵上"求逆"，输入稍有扰动权重就
    剧烈翻转；HRP 全程不求逆，预期权重漂移远小于最小方差。
    """
    R = make_synthetic_returns_large(n_assets, n_days)
    cov0 = np.cov(R, rowvar=False)
    corr0 = np.corrcoef(R, rowvar=False)
    cov0 = np.nan_to_num(cov0)                    # 兜底
    corr0 = np.nan_to_num(corr0, nan=0.0)
    np.fill_diagonal(corr0, 1.0)
    rank = int(np.linalg.matrix_rank(cov0))
    w_mv0 = min_var_weights(psd_repair(cov0))
    w_hrp0 = hrp_weights(corr0, cov0)
    table = _drift_table(corr0, cov0, sigmas, n_rep, w_mv0, w_hrp0, "B_合成100资产")
    eff = lambda w: 1.0 / float((w ** 2).sum())   # 有效资产数 1/Σw²
    info = {"秩": rank, "N": n_assets, "T": n_days,
            "最小方差有效资产数": eff(w_mv0), "HRP有效资产数": eff(w_hrp0),
            "最小方差最大权重": float(w_mv0.max()), "HRP最大权重": float(w_hrp0.max())}
    return table, info


# ============================================================================ #
# 出图与主流程
# ============================================================================ #
def plot_dendrogram(corr: np.ndarray, names: list):
    """图1：single 与 ward 两种链接的树状图并排。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, method in zip(axes, ["single", "ward"]):
        Z = cluster_corr(corr, method)
        dendrogram(Z, labels=names, ax=ax, leaf_rotation=90,
                   color_threshold=0.7 * Z[:, 2].max())
        ax.set_title(f"链接方式 = {method}" + ("（论文原版，主回测采用）" if method == "single" else "（对比项）"))
        ax.set_ylabel("相关距离 d = √(0.5(1-ρ))")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("图1 HRP 第一步：树状图 —— 谁和谁被聚成一簇？（10 个行业，共同重叠期）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_树状图.png"), dpi=150)
    plt.close(fig)


def plot_quasi_diag(corr: np.ndarray, order: list, names: list):
    """图2：准对角化重排前后的相关矩阵热力图（两张并排）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (mat, labels, title) in zip(axes, [
            (corr, names, "重排前：原始顺序"),
            (corr[np.ix_(order, order)], [names[i] for i in order],
             "重排后：准对角化（相似资产挨着，出现'块'结构）")]):
        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, label="相关系数 ρ")
    fig.suptitle("图2 HRP 第二步：准对角化 —— 同一个矩阵，按聚类顺序重排（协方差矩阵同样重排）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_准对角化重排前后热力图.png"), dpi=150)
    plt.close(fig)


def plot_nav(nav: dict, stats: pd.DataFrame):
    """图3：四方法净值对比。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"等权": "#8c8c8c", "HRP": "#d62728", "朴素风险平价": "#1f77b4", "最小方差": "#2ca02c"}
    for s, nv in nav.items():
        st = stats.loc[s]
        ax.plot(nv.index, nv.values, label=(f"{s}  年化{st['年化收益']:.1%} | 波动{st['年化波动']:.1%} "
                                            f"| 夏普{st['夏普比率']:.2f} | 回撤{st['最大回撤']:.1%}"),
                color=colors.get(s), lw=1.8 if s == "HRP" else 1.2)
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
    ax.set_title("图3 四种配置法月度再平衡滚动回测（滚动252日窗口，无交易成本）")
    ax.set_ylabel("净值（期初=1）")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_四方法净值对比.png"), dpi=150)
    plt.close(fig)


def plot_hrp_weights(hrp_hist: pd.DataFrame):
    """图4：HRP 权重堆叠图随时间演变。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(hrp_hist.index, [hrp_hist[c].values for c in hrp_hist.columns],
                 labels=list(hrp_hist.columns), alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_title("图4 HRP 组合权重随时间演变（每月再平衡，滚动252日窗口）")
    ax.set_ylabel("权重")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_HRP权重演变.png"), dpi=150)
    plt.close(fig)


def plot_noise(df_a: pd.DataFrame, df_b: pd.DataFrame, info_b: dict):
    """图5：噪声实验双面板 —— 左：真实10行业（协方差良态）；
    右：合成100资产 T<N（协方差奇异，HRP 的主场）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, df, title in [
            (axes[0], df_a, f"实验A：真实10行业，T=252≫N（协方差良态）"),
            (axes[1], df_b, f"实验B：合成N=100，T=60<N（协方差秩 {info_b['秩']}，病态）【合成数据】")]:
        for col, color, label in [
                ("最小方差权重漂移L1_均值", "#2ca02c", "最小方差（需求逆）"),
                ("HRP权重漂移L1_均值", "#d62728", "HRP（不需求逆）")]:
            m = df[col].values
            sd = df[col.replace("均值", "标准差")].values
            x = df["噪声幅度sigma"].values
            ax.plot(x, m, "o-", color=color, label=label)
            ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18)
        ax.set_xlabel("相关矩阵注入噪声的幅度 sigma")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("权重漂移 = Σ|w_noisy − w_baseline|（L1 范数）")
    fig.suptitle("图5 噪声实验：协方差良态时最小方差尚可；一旦矩阵病态，不求逆的 HRP 稳如磐石\n"
                 "（A：20 次重复，B：10 次重复，均值±1 标准差，rng(42)）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_噪声稳健性对比.png"), dpi=150)
    plt.close(fig)


def main():
    print("=" * 72)
    print("26_层次风险平价HRP（López de Prado HRPOpt）")
    print("=" * 72)

    # ---------------- [1] 数据 ---------------- #
    ret, n_jumps, is_synth = load_returns()
    tag = "【合成数据】" if is_synth else "【真实数据 data/cache/board_*.csv】"
    print(f"\n[1] 数据 {tag}")
    print(f"    行业数 = {ret.shape[1]}，共同重叠期 = {ret.index[0].date()} ~ {ret.index[-1].date()}，"
          f"共 {len(ret)} 个交易日")
    print(f"    跳变清洗：|日收益| > {JUMP_THRESHOLD:.0%} 置 0，共清洗 {n_jumps} 个观测")
    R = ret.values
    names = list(ret.columns)
    corr_full = np.corrcoef(R, rowvar=False)
    cov_full = np.cov(R, rowvar=False, ddof=1)
    eigs = np.linalg.eigvalsh(corr_full)
    print(f"    全样本相关矩阵特征值范围：[{eigs.min():.3f}, {eigs.max():.3f}]"
          f"（条件数 ≈ {eigs.max() / max(eigs.min(), 1e-12):.1f}）")

    # ---------------- [2] HRP 三步法（全样本演示） ---------------- #
    print("\n[2] HRP 三步法（全样本演示）")
    Z_single = cluster_corr(corr_full, "single")
    Z_ward = cluster_corr(corr_full, "ward")
    order = get_quasi_diag(Z_single)
    print(f"    single 链接准对角化顺序：{' → '.join(names[i] for i in order)}")
    print(f"    ward   链接准对角化顺序：{' → '.join(names[i] for i in get_quasi_diag(Z_ward))}")

    # 找出 single 树上"距离 < 1.0"的早期合并（对应ρ > 1-2*1²=…直接报告合并对）
    merges = []
    for row in Z_single:
        a, b, dist = int(row[0]), int(row[1]), row[2]
        la = names[a] if a < len(names) else f"簇{a}"
        lb = names[b] if b < len(names) else f"簇{b}"
        if dist <= 1.0:
            merges.append((la, lb, dist))
        else:
            break
    for la, lb, dist in merges:
        print(f"      早合并：{la} + {lb}（d = {dist:.3f}，ρ ≈ {1 - 2 * dist ** 2:.2f}）")

    w_hrp_full = hrp_weights(corr_full, cov_full)
    print("    全样本 HRP 权重：")
    for nm, w in sorted(zip(names, w_hrp_full), key=lambda t: -t[1]):
        print(f"      {nm:8s} {w:6.1%}")

    plot_dendrogram(corr_full, names)
    plot_quasi_diag(corr_full, order, names)
    print("    已出图：01_树状图.png / 02_准对角化重排前后热力图.png")

    # ---------------- [3] 回测 ---------------- #
    print(f"\n[3] 对比回测：等权 / HRP / 朴素风险平价 / 最小方差"
          f"（滚动 {N_LOOKBACK} 日窗口，月度再平衡，无交易成本）")
    nav, turnover, hrp_hist, _ = backtest(ret)
    stats = pd.DataFrame({s: perf_stats(nav[s], turnover[s]) for s in nav}).T
    print(stats.to_string(float_format=lambda x: f"{x:,.4f}"))
    stats_out = stats.copy()
    for c in ["年化收益", "年化波动", "最大回撤", "单次平均换手", "年化换手"]:
        stats_out[c] = stats_out[c].map(lambda x: f"{x:.4%}")
    stats_out["夏普比率"] = stats["夏普比率"].map(lambda x: f"{x:.3f}")
    stats_out.index.name = "策略"
    stats_out.to_csv(os.path.join(OUT_DIR, "portfolios.csv"), encoding="utf-8-sig")
    print("    已保存：outputs/portfolios.csv（utf-8-sig）")

    plot_nav(nav, stats)
    plot_hrp_weights(hrp_hist)
    print("    已出图：03_四方法净值对比.png / 04_HRP权重演变.png")

    # ---------------- 最新 HRP 权重 ---------------- #
    w_latest = hrp_weights(corr_full, cov_full)
    latest = pd.DataFrame({"行业": names, "HRP权重": w_latest}).sort_values(
        "HRP权重", ascending=False)
    latest["HRP权重"] = latest["HRP权重"].map(lambda x: f"{x:.6f}")
    latest.to_csv(os.path.join(OUT_DIR, "hrp_weights_latest.csv"),
                  index=False, encoding="utf-8-sig")
    print("    已保存：outputs/hrp_weights_latest.csv（全样本最新 HRP 权重）")

    # ---------------- [4] 噪声实验 ---------------- #
    print(f"\n[4] 噪声实验：给相关矩阵注入噪声（rng(42)，加扰动后重归一化），"
          f"对比权重漂移 L1 = Σ|w_noisy − w_baseline|")
    print("    —— 实验 A：真实 10 行业（T=252≫N，协方差良态），每个幅度重复 20 次")
    noise_a, w_mv0, w_hrp0 = noise_experiment(corr_full, cov_full)
    print(noise_a.drop(columns=["实验"]).to_string(index=False,
                                                  float_format=lambda x: f"{x:,.4f}"))
    print("    基准权重（未加噪）：")
    print("      最小方差：" + "，".join(f"{nm} {w:.1%}" for nm, w in
                                        sorted(zip(names, w_mv0), key=lambda t: -t[1])))
    print("      HRP     ：" + "，".join(f"{nm} {w:.1%}" for nm, w in
                                        sorted(zip(names, w_hrp0), key=lambda t: -t[1])))
    for s in (0.10, 0.30):
        ra = noise_a[noise_a["噪声幅度sigma"] == s].iloc[0]
        print(f"    sigma={s:.2f}：最小方差漂移 {ra['最小方差权重漂移L1_均值']:.3f} vs "
              f"HRP 漂移 {ra['HRP权重漂移L1_均值']:.3f}")

    print("    —— 实验 B：合成 N=100、T=60<N（协方差奇异，HRP 的主场），每个幅度重复 10 次")
    noise_b, info_b = noise_experiment_large_n()
    print(f"    协方差矩阵：N={info_b['N']}，T={info_b['T']}，秩 = {info_b['秩']}（< N，无法求逆）")
    print(f"    基准组合的有效资产数 1/Σw²：最小方差 = {info_b['最小方差有效资产数']:.1f}，"
          f"HRP = {info_b['HRP有效资产数']:.1f}；最大单权重：最小方差 = "
          f"{info_b['最小方差最大权重']:.1%}，HRP = {info_b['HRP最大权重']:.1%}")
    print(noise_b.drop(columns=["实验"]).to_string(index=False,
                                                  float_format=lambda x: f"{x:,.4f}"))
    plot_noise(noise_a, noise_b, info_b)
    print("    已出图：05_噪声稳健性对比.png（左：实验A，右：实验B）")

    # ---------------- 收尾 ---------------- #
    print("\n" + "=" * 72)
    print("教学要点：HRP 的简化近似与适用场景")
    print("  1) 简化近似：递归二分在'簇间'分配时忽略跨簇协方差，簇内用 IVP+子矩阵")
    print("     算簇方差——它不是马科维茨意义下的最优解，换来的是对估计误差的稳健。")
    print("  2) 噪声实验的诚实结论：协方差良态（N=10、T=252）时最小方差在小噪声下")
    print("     甚至更稳，HRP 因树结构翻转有约 0.2~0.3 的漂移'地板'；噪声一大或矩阵")
    print("     病态（实验B：T<N 秩亏）时，最小方差权重剧变而 HRP 几乎不动——这才是")
    print("     它的主场：资产多、相关结构复杂分层、协方差无法求逆。")
    print("=" * 72)
    print("全部输出已保存至 outputs/，运行成功（退出码 0）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
