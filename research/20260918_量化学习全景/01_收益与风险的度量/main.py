# -*- coding: utf-8 -*-
"""
01_收益与风险的度量
====================
主题：用真实A股数据回答两个最朴素的问题——"赚了多少"和"冒了多大风险"。

数据（前复权日K，来自工作区 data/cache/）：
    标的：平安银行（000001.SZ，2015年起）
    基准：沪深300指数（000300.SH），按共同交易日对齐

课程内容：
    1. 简单收益率 vs 对数收益率：时序可加性差异（图1）
    2. 年化收益（244交易日，A股惯例）、年化波动率、下行波动率
    3. 夏普比率（rf=2%）、索提诺比率、卡玛比率
    4. 最大回撤：净值/历史高点/回撤三线图，最大回撤、回撤区间、最长修复天数（图2）
    5. 历史法 VaR / ES（95%、99%），标在收益率分布直方图上（图3）
    6. 滚动60日年化波动率（观察波动率聚集，图4）
    7. 全部指标汇总对比表 -> outputs/metrics.csv

运行方式：
    cd 01_收益与风险的度量
    /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py

依赖：numpy / pandas / matplotlib（.venv 中已预装）。总运行时间 < 2 分钟。
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 无显示环境，纯出图
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 配置 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # 本课程文件夹
OUT_DIR = os.path.join(BASE_DIR, "outputs")                    # 输出目录
WORK_DIR = os.path.dirname(BASE_DIR)                           # 量化学习全景根目录
CACHE_DIR = os.path.join(WORK_DIR, "data", "cache")

STOCK_CSV = os.path.join(CACHE_DIR, "stock_000001_平安银行.csv")
INDEX_CSV = os.path.join(CACHE_DIR, "index_000300_沪深300.csv")

TRADING_DAYS = 244        # A股惯例：一年约244个交易日
RISK_FREE = 0.02          # 无风险利率 rf = 2%（年化）
ROLL_WIN = 60             # 滚动波动率窗口（交易日）
MAR = 0.0                 # 下行风险的最低可接受收益 MAR = 0（只把亏损日算下行风险）

# 中文绘图设置（必须 Agg 后端 + 中文字体）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 随机数种子：本脚本仅在"缓存缺失时的合成数据兜底"里用到随机数，
# 用 default_rng(42) 固定种子保证任何情况下结果可复现。
rng = np.random.default_rng(42)
IS_SYNTHETIC = False  # 若缓存缺失被迫用合成数据，置为 True，图与表中都会标注


# ---------------------------------------------------------------- 数据 ----
def load_close(path: str, name: str) -> pd.Series:
    """读取缓存CSV，返回收盘价序列（索引为日期）。

    若缓存文件缺失或无有效数据，则用 default_rng(42) 生成几何布朗运动
    合成数据兜底（全局规则：脚本绝不允许因缺数据而失败），
    并把全局开关 IS_SYNTHETIC 置 True，在所有图和表中明确标注"合成数据"。
    """
    global IS_SYNTHETIC
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
        s = df.set_index("date")["close"].dropna()
        s.name = name
        if len(s) > 60:  # 有效数据才使用
            return s
    # ---------- 合成数据兜底（几何布朗运动，参数仅用于教学演示） ----------
    print(f"[警告] {name} 缓存缺失/无效，改用合成数据（np.random.default_rng(42)）！")
    IS_SYNTHETIC = True
    n = 2440
    dates = pd.bdate_range("2015-01-05", periods=n)
    mu, sigma = 0.05 / TRADING_DAYS, 0.25 / np.sqrt(TRADING_DAYS)
    log_ret = mu - 0.5 * sigma**2 + sigma * rng.standard_normal(n)
    price = 10.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(price, index=dates, name=name)


# ------------------------------------------------------------ 收益与净值 ----
def compute_returns(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """由收盘价计算日简单收益率和对数收益率（首日无收益，置NaN后剔除）。

    简单收益率 r_t = P_t/P_{t-1} - 1  （截面可加：可直接算组合权重加权收益）
    对数收益率 l_t = ln(P_t/P_{t-1}) （时序可加：跨期直接相加）
    """
    ret = close / close.shift(1) - 1.0
    log_ret = np.log(close / close.shift(1))
    return ret.iloc[1:], log_ret.iloc[1:]


def nav_compound(simple_ret: pd.Series) -> pd.Series:
    """复利净值：1元起投，每日 (1+r) 连乘（正确做法）。"""
    return (1.0 + simple_ret).cumprod()


def nav_from_log(log_ret: pd.Series) -> pd.Series:
    """由对数收益率复利得到的净值：exp(累加对数收益)。

    数学上与 nav_compound 完全等价：exp(sum(ln(1+r))) = prod(1+r)，
    这正是"对数收益率时序可加"的含义——画出来两线重合。
    """
    return np.exp(log_ret.cumsum())


def nav_naive_sum(simple_ret: pd.Series) -> pd.Series:
    """错误示范：把日简单收益率按时序直接累加 1 + sum(r_t)。

    忽略复利与"波动拖累"（涨跌相乘的交叉项大多为负），
    对波动资产通常会明显高估真实净值，
    用来直观展示"简单收益率不能跨期相加"。
    """
    return 1.0 + simple_ret.cumsum()


# ------------------------------------------------------------ 核心指标 ----
def annual_return(nav_last: float, n_days: int) -> float:
    """年化收益（几何口径，244交易日惯例）：净值^(244/天数) - 1。

    等价于 exp(日均对数收益 × 244) - 1，即"按复利折算的一年收益"。
    """
    return nav_last ** (TRADING_DAYS / n_days) - 1.0


def annual_volatility(simple_ret: pd.Series) -> float:
    """年化波动率 = 日收益样本标准差(ddof=1) × sqrt(244)。

    为什么乘 sqrt(244)？独立同分布假设下，T 日收益方差 = T × 日方差，
    故波动率放大 sqrt(T) 倍。波动率是"可平方加总"的量，不能直接乘244。
    """
    return float(simple_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))


def downside_volatility(simple_ret: pd.Series, mar: float = MAR) -> float:
    """下行波动率 = sqrt( mean( min(r - MAR, 0)^2 ) ) × sqrt(244)。

    只把低于 MAR（本课取0）的收益计入风险，上涨不算风险。
    注意：上涨日以0计入均值（不是直接剔除），否则会高估下行集中度。
    """
    downside = np.minimum(simple_ret - mar, 0.0)
    return float(np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(ann_ret: float, ann_vol: float, rf: float = RISK_FREE) -> float:
    """夏普比率 = (年化收益 - rf) / 年化波动率。

    减 rf 的原因：把钱放在国债/货基也能白拿约2%收益，
    风险的"代价"必须用超出无风险收益的部分（超额收益）来衡量。
    """
    return (ann_ret - rf) / ann_vol


def sortino_ratio(ann_ret: float, d_vol: float, rf: float = RISK_FREE) -> float:
    """索提诺比率 = (年化收益 - rf) / 下行波动率：只惩罚下行波动。"""
    return (ann_ret - rf) / d_vol


def calmar_ratio(ann_ret: float, mdd: float) -> float:
    """卡玛比率 = 年化收益 / |最大回撤|：用"历史最痛的一跌"衡量风险。"""
    return ann_ret / abs(mdd)


# ------------------------------------------------------------ 回撤分析 ----
def drawdown_series(nav: pd.Series) -> tuple[pd.Series, pd.Series]:
    """返回(历史高点, 回撤序列)。回撤 = 净值/历史最高 - 1，<=0。"""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return peak, dd


def drawdown_stats(nav: pd.Series) -> dict:
    """计算最大回撤、回撤区间、修复日期、最长回撤修复天数。"""
    _, dd = drawdown_series(nav)

    # --- 最大回撤及其发生区间（峰 -> 谷） ---
    mdd = float(dd.min())
    trough_date = dd.idxmin()                       # 谷底日
    peak_date = nav.loc[:trough_date].idxmax()      # 谷底前最后一个历史高点日
    after = nav.loc[trough_date:]
    rec = after[after >= nav.loc[peak_date]]
    recovery_date = rec.index[0] if len(rec) else None  # 净值重回前高的日期

    # --- 最长回撤修复期：相邻两次"创新高日"之间的最大间隔（交易日） ---
    # 净值创新高 <=> 回撤为0。两次新高之间的每一天都处于"水下"。
    pos = pd.Series(np.arange(len(nav)), index=nav.index)
    high_pos = pos[dd > -1e-12].to_numpy()          # 创新高日的位置序号
    gaps = np.diff(high_pos) if len(high_pos) > 1 else np.array([0])
    ongoing = int(pos.iloc[-1] - high_pos[-1])      # 样本末仍在水下的进行中期
    if len(gaps) and gaps.max() >= ongoing:
        k = int(np.argmax(gaps))
        longest_days = int(gaps[k])                 # 峰到修复日的交易日数
        longest_peak = nav.index[high_pos[k]]
        longest_rec = nav.index[high_pos[k + 1]]
        longest_ongoing = False
    else:
        longest_days = ongoing                      # 至今未修复，计到样本末
        longest_peak = nav.index[high_pos[-1]]
        longest_rec = None
        longest_ongoing = True

    return {
        "mdd": mdd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,             # None 表示至今未修复
        "longest_days": longest_days,               # 交易日
        "longest_peak": longest_peak,
        "longest_rec": longest_rec,
        "longest_ongoing": longest_ongoing,
    }


# ------------------------------------------------------------ VaR / ES ----
def var_es(simple_ret: pd.Series, level: float) -> tuple[float, float]:
    """历史法 VaR 与 ES（日频，返回正数表示损失幅度）。

    VaR(95%) = 日收益的5%分位数取绝对值：
        "正常情况下，95%的交易日单日亏损不会超过这个数。"
    ES(95%)（预期损失/条件VaR）= 跌破 VaR 分位线的所有交易日的平均亏损：
        "一旦真的跌进最差的5%，平均要亏多少。" ES 永远 >= VaR。
    """
    q = float(simple_ret.quantile(1.0 - level))     # 负数
    tail = simple_ret[simple_ret <= q]
    es = float(tail.mean()) if len(tail) else q
    return -q, -es


# ------------------------------------------------------------ 汇总作图 ----
def fmt(x: float, pct: bool = True, digits: int = 2) -> str:
    """按百分比或小数格式化数字，NaN 显示为 '-'。"""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"


def build_metrics_row(name: str, close: pd.Series, simple_ret: pd.Series,
                      log_ret: pd.Series) -> tuple[dict, dict]:
    """计算单个资产的全部指标，返回 (metrics表行, 额外统计供打印)。"""
    nav = nav_compound(simple_ret)
    n_days = len(simple_ret)
    ann_ret = annual_return(float(nav.iloc[-1]), n_days)
    ann_vol = annual_volatility(simple_ret)
    d_vol = downside_volatility(simple_ret)
    dd_info = drawdown_stats(nav)
    var95, es95 = var_es(simple_ret, 0.95)
    var99, es99 = var_es(simple_ret, 0.99)
    roll_vol = simple_ret.rolling(ROLL_WIN).std(ddof=1) * np.sqrt(TRADING_DAYS)

    row = {
        "数据区间": f"{close.index[0]:%Y-%m-%d} ~ {close.index[-1]:%Y-%m-%d}",
        "交易日数": n_days,
        "区间累计收益": fmt(float(nav.iloc[-1] - 1)),
        "年化收益（几何,244日）": fmt(ann_ret),
        "年化波动率": fmt(ann_vol),
        "下行波动率（MAR=0）": fmt(d_vol),
        "夏普比率（rf=2%）": fmt(sharpe_ratio(ann_ret, ann_vol), pct=False),
        "索提诺比率": fmt(sortino_ratio(ann_ret, d_vol), pct=False),
        "卡玛比率": fmt(calmar_ratio(ann_ret, dd_info["mdd"]), pct=False),
        "最大回撤": fmt(dd_info["mdd"]),
        "最大回撤区间（峰→谷）": f"{dd_info['peak_date']:%Y-%m-%d} → {dd_info['trough_date']:%Y-%m-%d}",
        "回撤修复日期": (f"{dd_info['recovery_date']:%Y-%m-%d}"
                         if dd_info["recovery_date"] is not None else "未修复（截至样本末）"),
        "最长回撤修复天数（交易日）": dd_info["longest_days"],
        "最长回撤区间（峰→修复）": (
            f"{dd_info['longest_peak']:%Y-%m-%d} → "
            + (f"{dd_info['longest_rec']:%Y-%m-%d}" if dd_info["longest_rec"] is not None
               else "未修复")),
        "日VaR95（历史法）": fmt(var95),
        "日VaR99（历史法）": fmt(var99),
        "日ES95（历史法）": fmt(es95),
        "日ES99（历史法）": fmt(es99),
        "滚动60日年化波动率（均值）": fmt(float(roll_vol.mean())),
        "滚动60日年化波动率（最高）": fmt(float(roll_vol.max())),
        "滚动60日年化波动率（最低）": fmt(float(roll_vol.min())),
        "期末净值（1元起投）": fmt(float(nav.iloc[-1]), pct=False),
        "错误示范净值（日收益直接累加）": fmt(float(nav_naive_sum(simple_ret).iloc[-1]), pct=False),
        "对数净值 exp(Σln)（应与期末净值一致）": fmt(float(nav_from_log(log_ret).iloc[-1]), pct=False),
    }
    extra = {"ann_ret": ann_ret, "ann_vol": ann_vol, "d_vol": d_vol,
             "var95": var95, "es95": es95, "var99": var99, "es99": es99,
             "dd": dd_info, "nav": nav, "roll_vol": roll_vol}
    return row, extra


def plot_fig1(nav: pd.Series, log_nav: pd.Series, naive: pd.Series) -> None:
    """图1：复利净值 vs 对数净值 vs "日收益直接累加"的错误净值。"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, nm in zip(axes, nav.columns):
        ax.plot(nav.index, nav[nm], lw=1.8, color="#1f77b4", label="复利净值 ∏(1+r)（正确）")
        ax.plot(log_nav.index, log_nav[nm], lw=1.2, ls="--", color="#ff7f0e",
                label="对数净值 exp(Σln(1+r))（与复利重合）")
        ax.plot(naive.index, naive[nm], lw=1.5, color="#d62728",
                label="日简单收益直接累加 1+Σr（错误）")
        ax.set_title(f"{nm}：{'（合成数据）' if IS_SYNTHETIC else ''}"
                     f"复利净值={nav[nm].iloc[-1]:.3f}，错误累加={naive[nm].iloc[-1]:.3f}")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_ylabel("净值（1元起投）")
    axes[1].set_xlabel("日期")
    fig.suptitle("图1 简单收益率 vs 对数收益率：时序可加性的直观验证", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT_DIR, "01_累计净值对比.png"), dpi=150)
    plt.close(fig)


def plot_fig2(nav: pd.Series, peak: pd.Series, dd: pd.Series,
              dd_infos: dict) -> None:
    """图2：净值/历史高点/回撤序列 三线图（上：净值与历史高点，下：回撤）。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex="col")
    for j, nm in enumerate(nav.columns):
        info = dd_infos[nm]
        axes[0, j].plot(nav.index, nav[nm], lw=1.2, color="#1f77b4", label="净值")
        axes[0, j].plot(peak.index, peak[nm], lw=1.0, color="#2ca02c", ls="--",
                        label="历史最高点（cummax）")
        axes[0, j].axvspan(info["peak_date"], info["trough_date"], color="red", alpha=0.12)
        axes[0, j].set_title(f"{nm} 净值与历史高点（红带=最大回撤区间）")
        axes[0, j].legend(fontsize=9)
        axes[0, j].grid(alpha=0.3)
        axes[0, j].set_ylabel("净值")

        axes[1, j].fill_between(dd.index, dd[nm], 0, color="#d62728", alpha=0.35, step=None)
        axes[1, j].plot(dd.index, dd[nm], lw=1.0, color="#d62728")
        axes[1, j].axvspan(info["peak_date"], info["trough_date"], color="red", alpha=0.12)
        axes[1, j].annotate(f"最大回撤 {info['mdd']:.1%}\n"
                            f"{info['peak_date']:%Y-%m-%d}→{info['trough_date']:%Y-%m-%d}",
                            xy=(info["trough_date"], info["mdd"]),
                            xytext=(15, -8), textcoords="offset points", fontsize=9,
                            color="#8b0000")
        axes[1, j].set_title(f"{nm} 回撤序列（净值/历史高点-1）")
        axes[1, j].grid(alpha=0.3)
        axes[1, j].set_ylabel("回撤")
    for ax in axes[1]:
        ax.set_xlabel("日期")
    fig.suptitle(f"图2 最大回撤分析：{'（合成数据）' if IS_SYNTHETIC else '平安银行 vs 沪深300'}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "02_回撤分析.png"), dpi=150)
    plt.close(fig)


def plot_fig3(rets: pd.DataFrame, tail_stats: dict) -> None:
    """图3：日收益率分布直方图，标注历史法 VaR 与 ES 位置。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, nm in zip(axes, rets.columns):
        r_pct = rets[nm] * 100.0
        ax.hist(r_pct, bins=100, color="#9ecae1", edgecolor="none", alpha=0.9,
                density=True, label="日收益分布")
        st = tail_stats[nm]
        ax.axvline(-st["var95"] * 100, color="#ff7f0e", lw=1.6, ls="--",
                   label=f"VaR95={st['var95']:.2%}")
        ax.axvline(-st["var99"] * 100, color="#d62728", lw=1.6, ls="-",
                   label=f"VaR99={st['var99']:.2%}")
        ax.axvline(-st["es95"] * 100, color="#ff7f0e", lw=1.4, ls=":",
                   label=f"ES95={st['es95']:.2%}")
        ax.axvline(-st["es99"] * 100, color="#d62728", lw=1.4, ls=":",
                   label=f"ES99={st['es99']:.2%}")
        # x轴范围：左端略超出ES99以展示尾部，右端到99.9%分位（避免极端值压扁主体）
        x_left = min(-st["es99"] * 100 * 1.15, float(r_pct.quantile(0.001)))
        x_right = float(r_pct.quantile(0.999))
        ax.set_xlim(x_left, x_right)
        ax.set_title(f"{nm} 日收益率分布与 VaR/ES"
                     f"{'（合成数据）' if IS_SYNTHETIC else ''}")
        ax.set_xlabel("日收益率（%）")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("概率密度")
    fig.suptitle("图3 历史法 VaR 与 ES：竖线以左是最差的5%/1%交易日（虚线VaR，点线ES）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "03_收益率分布与VaR.png"), dpi=150)
    plt.close(fig)


def plot_fig4(roll_vols: dict, full_vols: dict) -> None:
    """图4：滚动60日年化波动率曲线（观察波动率聚集）。"""
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"平安银行": "#1f77b4", "沪深300": "#d62728"}
    for nm, rv in roll_vols.items():
        ax.plot(rv.index, rv, lw=1.2, color=colors.get(nm), label=f"{nm} 滚动{ROLL_WIN}日年化波动率")
        ax.axhline(full_vols[nm], color=colors.get(nm), lw=0.9, ls="--", alpha=0.6,
                   label=f"{nm} 全样本年化波动率 {full_vols[nm]:.1%}")
    ax.set_title(f"图4 滚动{ROLL_WIN}日年化波动率：高波动与低波动成串出现（波动率聚集）"
                 f"{'（合成数据）' if IS_SYNTHETIC else ''}")
    ax.set_xlabel("日期")
    ax.set_ylabel("年化波动率")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_滚动波动率.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程 ----
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("01_收益与风险的度量：平安银行 vs 沪深300（前复权，2015年起）")
    print("=" * 72)

    # 1. 读数与对齐：取两序列共同交易日的收盘价（内连接），保证同一起点
    stock = load_close(STOCK_CSV, "平安银行")
    index = load_close(INDEX_CSV, "沪深300")
    close = pd.concat([stock, index], axis=1, join="inner").dropna()
    print(f"对齐后共同交易日：{close.index[0]:%Y-%m-%d} ~ {close.index[-1]:%Y-%m-%d}，"
          f"共 {len(close)} 个交易日")
    if IS_SYNTHETIC:
        print("[注意] 本次运行使用合成数据！所有图与指标表均已注明。")

    # 2. 两种收益率
    rets = close / close.shift(1) - 1.0            # 日简单收益率
    log_rets = np.log(close / close.shift(1))      # 日对数收益率
    rets = rets.iloc[1:]
    log_rets = log_rets.iloc[1:]

    # 3. 逐资产计算指标
    rows, extras = {}, {}
    for nm in close.columns:
        rows[nm], extras[nm] = build_metrics_row(nm, close[nm], rets[nm], log_rets[nm])

    # 4. 指标对比表 -> outputs/metrics.csv（utf-8-sig，Excel可直接打开）
    metrics = pd.DataFrame(rows, index=list(rows["平安银行"].keys()))
    metrics.index.name = "指标"
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    metrics.to_csv(csv_path, encoding="utf-8-sig")

    # 5. 四张图
    nav = pd.DataFrame({nm: extras[nm]["nav"] for nm in close.columns})
    log_nav = pd.DataFrame({nm: nav_from_log(log_rets[nm]) for nm in close.columns})
    naive = pd.DataFrame({nm: nav_naive_sum(rets[nm]) for nm in close.columns})
    peak = nav.cummax()
    dd = nav / peak - 1.0
    plot_fig1(nav, log_nav, naive)
    plot_fig2(nav, peak, dd, {nm: extras[nm]["dd"] for nm in close.columns})
    plot_fig3(rets, extras)
    plot_fig4({nm: extras[nm]["roll_vol"] for nm in close.columns},
              {nm: extras[nm]["ann_vol"] for nm in close.columns})

    # 6. 控制台输出全部关键数字（供README引用，全部来自真实运行）
    print("\n" + "=" * 72)
    print("指标对比表（完整内容已写入 outputs/metrics.csv）")
    print("=" * 72)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 44,
                           "display.width", 160):
        print(metrics.to_string())
    print(f"\n输出文件：")
    for f in ["01_累计净值对比.png", "02_回撤分析.png", "03_收益率分布与VaR.png",
              "04_滚动波动率.png", "metrics.csv"]:
        print(f"  outputs/{f}")
    print("\n运行完成。")


if __name__ == "__main__":
    main()
