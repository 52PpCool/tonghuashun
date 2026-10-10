# -*- coding: utf-8 -*-
"""
07 机器学习选股：用 LightGBM 学"未来5日谁涨得好"，把数据泄漏防控讲透
====================================================================

本课只讲透一件事：机器学习选股里，"标签怎么对齐"和"统计量用什么样本算"
比模型本身重要十倍。我们故意构造一个【泄漏版】流水线和一个【正确版】流水线，
让新手亲眼看到：一个"标签错位错一天"的小 bug，能把样本外 IC 吹高多少。

流程（两个版本共享同一套 walk-forward 骨架，只有两处不同）：
  1. 特征面板：每个交易日 x 每只股票一行，全部只用 t 日及以前的信息；
  2. 标签：
     - 正确版：t 日特征 -> 未来5日收益 close[t+5]/close[t]-1
       （收益在 t+1 ~ t+5 这5个交易日里发生，t 日收盘出信号后才能赚到的部分）；
     - 泄漏版：标签窗口往前错一天 close[t+4]/close[t-1]-1
       （窗口里混进了"t日当天的收益"——你算特征时它已经发生，且永远无法再交易捕捉）；
  3. 标准化：
     - 正确版：特征均值/方差只用训练集估计，再套到测试集上；
     - 泄漏版：用全样本（训练+测试）均值/方差——第二个经典错误
       （对树模型+秩相关IC影响很小，但对线性/神经网络模型是真坑，见README）；
  4. Walk-forward：训练 2018-2022，从 2023 起每半年滚动重训一次、预测下一个半年；
  5. 样本外评价：每日截面 Spearman IC、Top5 等权组合 vs 30股等权基准。

随机种子固定 42；LightGBM 用极小参数保证速度；总运行时间 < 2 分钟。
仅供教学演示，不构成任何实盘建议。
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无界面环境出图
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 全局配置
SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "cache")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = "2018-01-01"      # 数据起点（按课程要求，2018年起）
FIRST_OOS = "2023-01-01"       # 样本外起点：训练 2018-2022，2023 起滚动预测
TOP_K = 5                      # 每日截面预测 Top5 等权
HOLD_DAYS = 5                  # 标签/持有期：5个交易日
MIN_CS_N = 8                   # 截面有效股票数下限（低于则该日截面不用）
MIN_IC_N = 10                  # 计算当日IC所需的最少(预测,标签)对数
RF_ANNUAL = 0.02               # 无风险利率（夏普比率用 rf=2%，与本系列课01-04口径一致）

# LightGBM 极小参数（保证几十秒内跑完，这也是教学演示的合理规模）
LGB_PARAMS = dict(
    n_estimators=100,
    num_leaves=15,
    learning_rate=0.05,
    verbose=-1,
    random_state=SEED,
)

# 中文字体（按课程规范设置）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

IS_SYNTHETIC = False  # 若缓存数据缺失、被迫用合成数据，则置 True 并在图表中标注


# ---------------------------------------------------------------- 数据加载
def load_panel():
    """遍历 data/cache/stock_*.csv，拼成长表：一行 = 某股票某交易日。

    不同股票来源不同、有的列缺失（如 turnover），
    代码必须兼容：有则用、无则该列置 NaN，绝不因缺列报错。
    """
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("stock_") and f.endswith(".csv")]
    if not files:
        # 数据缺失兜底：造合成数据（明确标注），保证课程永远能跑通
        global IS_SYNTHETIC
        IS_SYNTHETIC = True
        print("[警告] 未找到 data/cache/stock_*.csv，改用合成数据演示（图表将标注）")
        return make_synthetic_panel()

    frames = []
    for fname in sorted(files):
        # 文件名形如 stock_000001_平安银行.csv
        parts = fname[:-4].split("_")
        code, name = parts[1], parts[2] if len(parts) > 2 else parts[1]
        df = pd.read_csv(os.path.join(DATA_DIR, fname), encoding="utf-8-sig")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= START_DATE].copy()
        df["code"], df["name"] = code, name
        # 兼容缺失列：有则用，无则置 NaN
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[["date", "code", "name", "open", "high", "low", "close", "volume", "turnover"]])
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
    panel = panel.dropna(subset=["close"])
    return panel


def make_synthetic_panel():
    """合成数据兜底：30只虚构股票、2018年起的几何随机游走（种子42）。"""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START_DATE, "2026-09-17")
    frames = []
    for i in range(30):
        n = len(dates)
        ret = rng.normal(0.0003, 0.02, n)
        close = 20 * np.exp(np.cumsum(ret))
        high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
        volume = rng.lognormal(15, 0.5, n)
        frames.append(pd.DataFrame({
            "date": dates, "code": f"S{i:03d}", "name": f"合成股{i}",
            "open": close * (1 + rng.normal(0, 0.004, n)),
            "high": high, "low": low, "close": close, "volume": volume,
            "turnover": rng.uniform(0.5, 5, n),
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 特征与标签
# 特征面板：全部由 t 日收盘及以前的数据算出 —— 也就是说，
# 每个特征都要到"t日收盘后"才最终确定，最早只能在 t 日收盘后 / t+1 开盘前用来下单。
#   mom5/mom20/mom60 : 过去5/20/60日涨幅（用到 close[t] 及更早）   -> t日收盘后确定
#   vol60            : 过去60日日收益率标准差                      -> t日收盘后确定
#   vol_ratio        : 当日成交量 / 过去20日均量（量比）           -> t日收盘后确定
#   dev_ma20         : 收盘价相对20日均线的偏离                    -> t日收盘后确定
#   amp5             : 过去5日平均振幅 (high-low)/昨收             -> t日收盘后确定
#   turn5            : 过去5日平均换手率（部分股票无此列 -> NaN）   -> t日收盘后确定
FEATURES = ["mom5", "mom20", "mom60", "vol60", "vol_ratio", "dev_ma20", "amp5", "turn5"]
FEATURE_NAMES_CN = {
    "mom5": "5日动量", "mom20": "20日动量", "mom60": "60日动量",
    "vol60": "60日波动率", "vol_ratio": "量比(20日)", "dev_ma20": "MA20偏离",
    "amp5": "5日平均振幅", "turn5": "5日平均换手率",
}


def build_wide(panel):
    """长表 -> 每个字段一张宽表（index=交易日, columns=股票代码），向量化算特征与标签。"""
    close_w = panel.pivot(index="date", columns="code", values="close").sort_index()
    high_w = panel.pivot(index="date", columns="code", values="high").sort_index()
    low_w = panel.pivot(index="date", columns="code", values="low").sort_index()
    vol_w = panel.pivot(index="date", columns="code", values="volume").sort_index()
    turn_w = panel.pivot(index="date", columns="code", values="turnover").sort_index()

    ret1_w = close_w / close_w.shift(1) - 1.0          # 日收益率

    # ---- 特征（只用 t 日及以前）----
    feat_w = {}
    feat_w["mom5"] = close_w / close_w.shift(5) - 1.0
    feat_w["mom20"] = close_w / close_w.shift(20) - 1.0
    feat_w["mom60"] = close_w / close_w.shift(60) - 1.0
    feat_w["vol60"] = ret1_w.rolling(60).std()
    feat_w["vol_ratio"] = vol_w / vol_w.rolling(20).mean()
    feat_w["dev_ma20"] = close_w / close_w.rolling(20).mean() - 1.0
    amp1_w = (high_w - low_w) / close_w.shift(1)       # 当日振幅（昨收已知）
    feat_w["amp5"] = amp1_w.rolling(5).mean()
    feat_w["turn5"] = turn_w.rolling(5).mean()          # 无 turnover 列的股票整列为 NaN
    feat_w = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in feat_w.items()}

    # ---- 标签：防泄漏的核心就在这两行的错位 ----
    # 正确标签：t日特征 -> 未来5日收益（收益发生在 t+1 ~ t+5 这5个交易日）
    fwd5_w = close_w.shift(-HOLD_DAYS) / close_w - 1.0
    # 泄漏标签：窗口整体往前错1天 -> close[t-1] 到 close[t+4]
    #   = t日当天收益 * t+1..t+4的收益。其中"t日当天收益"在算特征时已经发生，
    #     模型可以从动量类特征里"猜出"它的相当一部分 => 样本外IC被凭空吹高。
    fwdleak_w = close_w.shift(-(HOLD_DAYS - 1)) / close_w.shift(1) - 1.0

    return feat_w, ret1_w, fwd5_w, fwdleak_w


def cs_zscore(w, min_n=MIN_CS_N):
    """截面 z-score：每个交易日内部做标准化（只用当日截面信息，无时间泄漏）。"""
    mu = w.mean(axis=1)
    sd = w.std(axis=1).where(lambda s: s > 1e-12)      # 截面无离散度 -> 全NaN
    z = w.sub(mu, axis=0).div(sd, axis=0)
    too_few = w.notna().sum(axis=1) < min_n
    z[too_few] = np.nan
    return z


def to_long(w):
    """宽表 -> 长表（MultiIndex=(date, code)），供模型使用。"""
    return w.stack()


# ---------------------------------------------------------------- 标准化
def fit_standardizer(X, train_mask, leaky):
    """估计特征的均值/方差。

    正确版(leaky=False)：只用训练集行估计，再套到测试集 —— 这是纪律；
    泄漏版(leaky=True)：用全样本(训练+测试)估计 —— 标准化统计量里混入了未来。
    注意：对 LightGBM 这类树模型 + 秩相关IC，这个特定泄漏影响极小
    （树只看特征的大小次序，全局仿射变换不改变次序）；
    它对线性模型/神经网络是真正的性能杀手。此处保留它是为了展示完整防线。
    """
    if leaky:
        X_fit = X
    else:
        X_fit = X.loc[train_mask]
    mu = X_fit.mean(axis=0)
    sd = X_fit.std(axis=0).where(lambda s: s > 1e-12, 1.0)
    return mu, sd


# ---------------------------------------------------------------- Walk-forward
def half_year_buckets(dates):
    """从 2023-01-01 起把样本外切成半年一段（每段重训一次模型）。"""
    end = dates.max()
    buckets, y = [], 2023
    while pd.Timestamp(y, 1, 1) <= end:
        for m0, d0, m1, d1, tag in [(1, 1, 6, 30, "H1"), (7, 1, 12, 31, "H2")]:
            s = pd.Timestamp(y, m0, d0)
            e = pd.Timestamp(y, m1, d1)
            if s > end:
                break
            buckets.append((max(s, pd.Timestamp(FIRST_OOS)), min(e, end), f"{y}{tag}"))
        y += 1
    return buckets


def run_walkforward(feat_long, y_z_long, leaky, version_name):
    """滚动训练-预测：训练 2018-2022，2023 起每半年重训一次（ expanding 窗口）。

    每个半年段：
      1. 用"段起点之前"的全部数据当训练集（正确版标准化也只用这段训练集统计量）；
      2. 训练 LightGBM，预测该半年段内每个交易日 x 每只股票的标签分值；
      3. 收集预测 -> 拼成宽表；累计 gain 特征重要性。
    """
    buckets = half_year_buckets(feat_long.index.get_level_values("date").unique())
    pred_frames, imp_sums = [], np.zeros(len(FEATURES))
    feat_valid = feat_long.notna().sum(axis=1) >= 4   # 至少4个特征非NaN才算有效样本行

    for b_start, b_end, b_name in buckets:
        dates_lvl = feat_long.index.get_level_values("date")
        train_mask = (dates_lvl >= pd.Timestamp(START_DATE)) & (dates_lvl < b_start) & feat_valid
        train_mask = train_mask & y_z_long.reindex(feat_long.index).notna()

        mu, sd = fit_standardizer(feat_long, train_mask, leaky)
        X_std = (feat_long - mu) / sd                  # NaN 保持 NaN，LightGBM 原生处理

        X_tr = X_std.loc[train_mask]
        y_tr = y_z_long.reindex(X_tr.index).dropna()
        X_tr = X_tr.loc[y_tr.index]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_tr, y_tr)
        imp_sums += model.booster_.feature_importance(importance_type="gain")

        oos_mask = (dates_lvl >= b_start) & (dates_lvl <= b_end) & feat_valid
        pred = pd.Series(model.predict(X_std.loc[oos_mask]), index=X_std.index[oos_mask])
        pred_frames.append(pred.rename(b_name))
        print(f"  [{version_name}] 训练至 {b_start.date()}  "
              f"训练样本 {len(y_tr):>6} 行 -> 预测 {b_name}")

    pred_long = pd.concat(pred_frames)
    pred_w = pred_long.unstack()                        # 宽表：行=日期, 列=股票
    imp_share = imp_sums / max(imp_sums.sum(), 1e-12)
    return pred_w, pd.Series(imp_share, index=FEATURES)


# ---------------------------------------------------------------- 评价
def daily_ic_series(pred_w, realized_w, min_n=MIN_IC_N):
    """每日截面 Spearman IC：当日预测 vs 当日对应的真实未来收益。"""
    ics = {}
    for dt in pred_w.index:
        if dt not in realized_w.index:
            continue
        p, r = pred_w.loc[dt], realized_w.loc[dt]
        mask = p.notna() & r.notna()
        if mask.sum() < min_n:
            continue
        ic = spearmanr(p[mask], r[mask])[0]
        if np.isfinite(ic):
            ics[dt] = ic
    return pd.Series(ics).sort_index()


def topk_portfolio(pred_w, ret1_w, topk=TOP_K, hold=HOLD_DAYS):
    """Top5 等权组合（5日持有、每日滚动、5个重叠子组合）。

    t 日收盘出预测 -> 选 Top5 -> 持有5个交易日（t+1..t+5 的日收益）；
    每天都有5个不同入场日的子组合在持仓，组合日收益 = 5个子组合的平均。
    等权基准 = 当日全部股票等权（也是每日再平衡）。
    返回 (组合日收益, 基准日收益)。
    """
    sel = (pred_w.rank(axis=1, ascending=False) <= topk) & pred_w.notna()
    sel = sel.astype(float)

    basket_rets = []
    for k in range(1, hold + 1):
        sel_lag = sel.shift(k)                         # k天前选出的子组合
        cnt = sel_lag.sum(axis=1)
        r = (sel_lag * ret1_w).sum(axis=1) / cnt
        basket_rets.append(r.where(cnt >= topk))       # 子组合不满5只 -> 视为无效
    port = pd.concat(basket_rets, axis=1).mean(axis=1, skipna=True)
    bench = ret1_w.mean(axis=1)
    return port, bench


def perf_stats(daily_ret):
    """年化收益 / 年化波动 / 最大回撤 / 夏普（无风险利率 rf=2%，与本系列课01-04一致）。"""
    r = daily_ret.dropna()
    if len(r) < 10:
        return dict(ann_ret=np.nan, ann_vol=np.nan, max_dd=np.nan, sharpe=np.nan, n_days=len(r))
    nv = (1 + r).cumprod()
    n = len(nv)
    ann_ret = nv.iloc[-1] ** (252.0 / n) - 1.0
    ann_vol = r.std() * np.sqrt(252)
    max_dd = (nv / nv.cummax() - 1.0).min()
    sharpe = (r.mean() * 252 - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, max_dd=max_dd, sharpe=sharpe, n_days=n)


# ---------------------------------------------------------------- 绘图
def plot_netvalue(port_c, port_l, bench):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for s, label, color, lw in [
        ((1 + port_c.dropna()).cumprod(), "正确版 Top5", "tab:blue", 1.8),
        ((1 + port_l.dropna()).cumprod(), "泄漏版 Top5（IC虚高的假象）", "tab:red", 1.4),
        ((1 + bench.dropna()).cumprod(), "30股等权基准", "tab:gray", 1.4),
    ]:
        ax.plot(s.index, s.values, label=label, color=color, lw=lw)
    st_c, st_b = perf_stats(port_c), perf_stats(bench)
    ax.set_title("样本外（2023起）Top5组合 vs 等权基准净值\n"
                 f"正确版年化 {st_c['ann_ret']*100:.1f}% / 最大回撤 {st_c['max_dd']*100:.1f}%  |  "
                 f"基准年化 {st_b['ann_ret']*100:.1f}% / 最大回撤 {st_b['max_dd']*100:.1f}%")
    ax.set_ylabel("净值（起点=1）")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "01_样本外Top5组合vs基准净值.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 01_样本外Top5组合vs基准净值.png")


def plot_importance(imp_share):
    order = imp_share.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh([FEATURE_NAMES_CN[f] for f in order.index], order.values * 100, color="tab:blue", alpha=0.8)
    for i, v in enumerate(order.values * 100):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_title("LightGBM 特征重要性（gain，8次滚动重训平均）")
    ax.set_xlabel("重要性占比 %")
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(os.path.join(OUT_DIR, "02_特征重要性.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 02_特征重要性.png")


def plot_ic_compare(ic_c, ic_l, bucket_names_c, bucket_ic_c, bucket_names_l, bucket_ic_l):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    axes[0].bar(["泄漏版", "正确版"], [ic_l.mean(), ic_c.mean()],
                color=["tab:red", "tab:blue"], alpha=0.85, width=0.5)
    for x, v in zip([0, 1], [ic_l.mean(), ic_c.mean()]):
        axes[0].text(x, v + 0.002, f"{v:.4f}", ha="center", fontsize=11)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title(f"样本外平均IC对比（泄漏版比正确版高 {ic_l.mean()-ic_c.mean():+.4f}）")
    axes[0].set_ylabel("平均 Spearman IC")
    axes[0].grid(alpha=0.3, axis="y")

    x = np.arange(len(bucket_names_c))
    axes[1].bar(x - 0.2, bucket_ic_l, width=0.4, label="泄漏版", color="tab:red", alpha=0.85)
    axes[1].bar(x + 0.2, bucket_ic_c, width=0.4, label="正确版", color="tab:blue", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bucket_names_c, rotation=45, fontsize=8)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("各半年段的样本外平均IC")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    fig.savefig(os.path.join(OUT_DIR, "03_泄漏版vs正确版IC对比.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 03_泄漏版vs正确版IC对比.png")


def plot_rolling_ic(ic_c, ic_l, window=60):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ic_c.index, ic_c.values, color="tab:blue", alpha=0.25, lw=0.7, label="正确版 日度IC")
    ax.plot(ic_l.index, ic_l.values, color="tab:red", alpha=0.2, lw=0.7, label="泄漏版 日度IC")
    ax.plot(ic_c.rolling(window).mean().index, ic_c.rolling(window).mean().values,
            color="tab:blue", lw=2, label=f"正确版 {window}日滚动IC")
    ax.plot(ic_l.rolling(window).mean().index, ic_l.rolling(window).mean().values,
            color="tab:red", lw=2, label=f"泄漏版 {window}日滚动IC")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"滚动样本外IC（{window}日均线）\n"
                 f"正确版均值 {ic_c.mean():.4f} | 泄漏版均值 {ic_l.mean():.4f}")
    ax.set_ylabel("Spearman IC")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "04_滚动样本外IC.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 04_滚动样本外IC.png")


# ---------------------------------------------------------------- 主流程
def main():
    t0 = time.time()
    print("=" * 70)
    print("07 机器学习选股：LightGBM + 防泄漏（随机种子 42）")
    print("=" * 70)

    # 1. 数据
    panel = load_panel()
    n_stocks = panel["code"].nunique()
    print(f"股票数 {n_stocks}，交易日 {panel['date'].nunique()}，"
          f"区间 {panel['date'].min().date()} ~ {panel['date'].max().date()}，"
          f"{'合成数据' if IS_SYNTHETIC else '真实数据'}")

    # 2. 特征与标签（宽表向量化）
    feat_w, ret1_w, fwd5_w, fwdleak_w = build_wide(panel)
    feat_long = pd.concat({k: to_long(v) for k, v in feat_w.items()}, axis=1)[FEATURES]

    # 标签做截面 z-score（同一日内可比，跨日标准化；截面信息无时间泄漏）
    y_correct_z = to_long(cs_zscore(fwd5_w))    # 正确版训练标签
    y_leak_z = to_long(cs_zscore(fwdleak_w))    # 泄漏版训练标签

    # 3. 两条流水线：同骨架、不同标签对齐/标准化口径
    print("\n[Walk-forward 训练] 训练2018-2022，2023起每半年重训：")
    pred_correct_w, imp_share = run_walkforward(feat_long, y_correct_z, leaky=False, version_name="正确版")
    pred_leak_w, _ = run_walkforward(feat_long, y_leak_z, leaky=True, version_name="泄漏版")

    # 4. 样本外 IC（正确版 vs 正确真实标签；泄漏版 vs 泄漏标签=新手口径）
    ic_c = daily_ic_series(pred_correct_w, fwd5_w)
    ic_l = daily_ic_series(pred_leak_w, fwdleak_w)
    ic_l_vs_true = daily_ic_series(pred_leak_w, fwd5_w)   # 泄漏版预测 vs 真实未来收益（诚实口径）

    # 各半年段平均IC（作图用）
    buckets = half_year_buckets(pred_correct_w.index)
    bn = [b[2] for b in buckets]
    bi_c, bi_l = [], []
    for s, e, _ in buckets:
        bi_c.append(ic_c[(ic_c.index >= s) & (ic_c.index <= e)].mean())
        bi_l.append(ic_l[(ic_l.index >= s) & (ic_l.index <= e)].mean())

    # 5. 组合回测（注意：两个版本用的是同一条真实日收益——泄漏改善不了真实收益）
    port_c, bench = topk_portfolio(pred_correct_w, ret1_w)
    port_l, _ = topk_portfolio(pred_leak_w, ret1_w)
    # 公平对比：基准与Top5组合取完全相同的样本外区间（组合实际开始交易的日期起算）
    oos_start = port_c.dropna().index.min()
    bench_oos = bench[bench.index >= oos_start]
    st_c, st_l, st_b = perf_stats(port_c), perf_stats(port_l), perf_stats(bench_oos)

    # 6. 控制台汇总（README 中的数字全部来自这里）
    print("\n" + "=" * 70)
    print("【样本外结果汇总】区间：%s ~ %s（共 %d 个交易日）"
          % (ic_c.index.min().date(), ic_c.index.max().date(), len(ic_c)))
    print(f"  正确版  平均IC={ic_c.mean():+.4f}  ICIR={ic_c.mean()/ic_c.std():+.3f}  "
          f"IC>0比例={100*(ic_c>0).mean():.1f}%")
    print(f"  泄漏版  平均IC={ic_l.mean():+.4f}  ICIR={ic_l.mean()/ic_l.std():+.3f}  "
          f"IC>0比例={100*(ic_l>0).mean():.1f}%   （按新手自身口径：vs 泄漏标签）")
    print(f"  泄漏版  平均IC={ic_l_vs_true.mean():+.4f}（诚实口径：预测 vs 真实未来5日收益）")
    print(f"  IC虚高幅度 = {ic_l.mean()-ic_c.mean():+.4f}（{100*(ic_l.mean()/ic_c.mean()-1):+.1f}%）")
    print(f"  正确版Top5：年化 {st_c['ann_ret']*100:+.2f}%  最大回撤 {st_c['max_dd']*100:.2f}%  "
          f"夏普 {st_c['sharpe']:.2f}")
    print(f"  泄漏版Top5：年化 {st_l['ann_ret']*100:+.2f}%  最大回撤 {st_l['max_dd']*100:.2f}%  "
          f"夏普 {st_l['sharpe']:.2f}")
    print(f"  等权基准  ：年化 {st_b['ann_ret']*100:+.2f}%  最大回撤 {st_b['max_dd']*100:.2f}%  "
          f"夏普 {st_b['sharpe']:.2f}")
    print(f"  正确版Top5相对基准超额年化：{(st_c['ann_ret']-st_b['ann_ret'])*100:+.2f} 个百分点")
    print("=" * 70)

    # 7. 落盘
    print("\n[输出文件]")
    plot_netvalue(port_c, port_l, bench_oos)
    plot_importance(imp_share)
    plot_ic_compare(ic_c, ic_l, bn, bi_c, bn, bi_l)
    plot_rolling_ic(ic_c, ic_l)

    metrics = pd.DataFrame([
        {"版本": "正确版", "样本外平均IC": round(ic_c.mean(), 4),
         "ICIR": round(ic_c.mean() / ic_c.std(), 3),
         "IC为正比例": round((ic_c > 0).mean(), 3),
         "Top5年化收益": round(st_c["ann_ret"], 4), "Top5最大回撤": round(st_c["max_dd"], 4),
         "Top5夏普": round(st_c["sharpe"], 2),
         "基准年化收益": round(st_b["ann_ret"], 4), "基准最大回撤": round(st_b["max_dd"], 4),
         "Top5超额年化": round(st_c["ann_ret"] - st_b["ann_ret"], 4)},
        {"版本": "泄漏版", "样本外平均IC": round(ic_l.mean(), 4),
         "ICIR": round(ic_l.mean() / ic_l.std(), 3),
         "IC为正比例": round((ic_l > 0).mean(), 3),
         "Top5年化收益": round(st_l["ann_ret"], 4), "Top5最大回撤": round(st_l["max_dd"], 4),
         "Top5夏普": round(st_l["sharpe"], 2),
         "基准年化收益": round(st_b["ann_ret"], 4), "基准最大回撤": round(st_b["max_dd"], 4),
         "Top5超额年化": round(st_l["ann_ret"] - st_b["ann_ret"], 4)},
    ]).set_index("版本")
    metrics.to_csv(os.path.join(OUT_DIR, "metrics.csv"), encoding="utf-8-sig")
    print("  已保存 metrics.csv")

    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "特征中文名": [FEATURE_NAMES_CN[f] for f in FEATURES],
        "gain占比": imp_share.round(4),
    }).sort_values("gain占比", ascending=False).reset_index(drop=True)
    imp_df.index += 1
    imp_df.index.name = "排名"
    imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), encoding="utf-8-sig")
    print("  已保存 feature_importance.csv")

    # 逐日预测+真实标签留档（方便课后自己验算IC）；只留样本外行，与文件名一致
    pred_detail = pd.DataFrame({
        "pred_correct": pred_correct_w.stack(),
        "pred_leaky": pred_leak_w.stack(),
        "fwd5_realized": fwd5_w.stack(),
        "fwdleak_realized": fwdleak_w.stack(),
    })
    pred_detail = pred_detail[pred_detail.index.get_level_values("date") >= pd.Timestamp(FIRST_OOS)]
    pred_detail.to_csv(os.path.join(OUT_DIR, "oos_predictions.csv"), encoding="utf-8-sig")
    print(f"  已保存 oos_predictions.csv（{len(pred_detail)} 行）")

    print(f"\n总耗时 {time.time()-t0:.1f} 秒。教学演示，不构成实盘建议。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
