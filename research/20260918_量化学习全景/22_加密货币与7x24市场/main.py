# -*- coding: utf-8 -*-
"""
第22课：加密货币与 7×24 市场
================================
走出 A 股，研究一个全年无休、没有任何涨跌停和休市的市场：
  一) 数据加载   ：BTC / ETH 日线（2018 至今）+ 最近730天 BTC 小时线（先缓存后下载，失败合成兜底）
  二) 波动率画像 ：年化波动 / 单日最大跌幅 / VaR99，与沪深300对照；BTC 逐年波动率（市场成熟化）
  三) 日历效应   ：星期 × 小时 平均收益 / 波动热力图（7×24 特色，统一北京时间）
  四) 避险叙事   ：BTC vs 沪深300 滚动90日相关系数，用数据检验"数字黄金"说法
  五) 跨市场对比 ：同一套 20/60 双均线分别跑 BTC 与沪深300（含各自合理成本），如实对比
  六) 稳健性提示 ：√365 vs √244 年化、加密数据源质量、无杠杆现货假设

运行（用本项目的 venv）：
  /home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py
首次运行在线下载并写缓存到本课 data/ 目录；之后离线也能跑（先读缓存）。
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无显示环境出图
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
# 沪深300缓存（只读，不改动工作区其他目录；注意：该缓存实为 510300 ETF 净值序列，只可用于算收益）
CSI300_CACHE = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache", "index_000300_沪深300.csv"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

START = "2018-01-01"      # 统一起点
ANN_CRYPTO = 365.0        # 7×24 全年无休 → 年化因子 √365
ANN_ASHARE = 244.0        # A股一年约 244 个交易日 → 年化因子 √244
COST_BTC = 0.0005         # BTC 现货 taker 手续费 万5/单边（买、卖各收一次）
COST_CSI = 0.0001         # 沪深300ETF 佣金约 万1/单边（免印花税）

METRICS = []  # 收集 (指标, 数值, 说明)，最后写 outputs/metrics.csv
SYNTH = []    # 记录哪些数据用了合成兜底


def add_metric(name, value, note=""):
    METRICS.append((name, value, note))


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ----------------------------------------------------------------------
# 一、数据加载：先读缓存 → 无缓存才在线下载并写缓存 → 下载失败用合成数据兜底
# ----------------------------------------------------------------------
def _fmt_yf_daily(df):
    """把 yfinance 返回统一成 open/high/low/close/volume + 去时区日期索引。"""
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance 返回空数据")
    if isinstance(df.columns, pd.MultiIndex):  # 新版 yfinance 单票也返回双层列
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_daily(symbol, cache_name, synth_fn):
    """日线通用加载器：缓存优先，下载兜底，最后合成数据兜底（绝不因下载失败而崩溃）。"""
    path = os.path.join(DATA_DIR, cache_name)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        print(f"    [缓存] {cache_name}：{len(df)} 行（{df.index[0].date()} ~ {df.index[-1].date()}）")
        return df, False
    try:
        import yfinance as yf
        df = _fmt_yf_daily(yf.download(symbol, start=START, interval="1d",
                                       auto_adjust=True, progress=False))
        df.to_csv(path, index_label="date")
        print(f"    [下载] {symbol} 日线 {len(df)} 行，已写缓存 → data/{cache_name}")
        return df, False
    except Exception as exc:
        print(f"    [警告] {symbol} 在线下载失败（{exc}），改用合成数据兜底（已标注）！")
        df = synth_fn()
        df.to_csv(path, index_label="date")
        SYNTH.append(cache_name)
        return df, True


def synth_btc_daily():
    rng = np.random.default_rng(SEED)
    days = pd.date_range(START, pd.Timestamp.today().normalize(), freq="D")
    ret = rng.normal(0.0006, 0.04, len(days))
    close = 13800.0 * np.exp(np.cumsum(ret))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 0.0}, index=days)
    df.index.name = "date"
    print(f"    [合成] BTC 日线合成数据 {len(df)} 行（default_rng(42)，请勿当真实数据使用）")
    return df


def synth_eth_daily():
    rng = np.random.default_rng(SEED)
    days = pd.date_range(START, pd.Timestamp.today().normalize(), freq="D")
    ret = rng.normal(0.0008, 0.045, len(days))
    close = 750.0 * np.exp(np.cumsum(ret))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 0.0}, index=days)
    df.index.name = "date"
    print(f"    [合成] ETH 日线合成数据 {len(df)} 行（default_rng(42)，请勿当真实数据使用）")
    return df


def load_btc_hourly():
    """最近730天 BTC 小时线。统一转成【北京时间】再存缓存，便于讲"周末流动性"故事。"""
    path = os.path.join(DATA_DIR, "btc_usd_hourly_recent.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        print(f"    [缓存] btc_usd_hourly_recent.csv：{len(df)} 行（索引=北京时间）")
        return df, False
    try:
        import yfinance as yf
        raw = yf.download("BTC-USD", period="730d", interval="1h",
                          auto_adjust=True, progress=False)
        if raw is None or len(raw) == 0:
            raise RuntimeError("小时线返回为空")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
        raw.columns = ["open", "high", "low", "close", "volume"]
        idx = pd.to_datetime(raw.index)
        if idx.tz is None:  # 雅虎小时线一般是 UTC；若已是时区感知则直接转换
            idx = idx.tz_localize("UTC")
        idx = idx.tz_convert("Asia/Shanghai").tz_localize(None)
        raw.index = idx
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()
        raw.to_csv(path, index_label="date")
        print(f"    [下载] BTC 小时线 {len(raw)} 行（已转北京时间），已写缓存")
        return raw, False
    except Exception as exc:
        print(f"    [警告] BTC 小时线下载失败（{exc}），改用合成数据兜底（已标注）！")
        rng = np.random.default_rng(SEED)
        hours = pd.date_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=730),
                              pd.Timestamp.today(), freq="h")
        sigma = np.where(hours.weekday >= 5, 0.0032, 0.0042)  # 周末波动略低的假象样本
        ret = rng.normal(2e-5, 1.0, len(hours)) * sigma
        close = 30000.0 * np.exp(np.cumsum(ret))
        df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                           "volume": 0.0}, index=hours)
        df.index.name = "date"
        df.to_csv(path, index_label="date")
        SYNTH.append("btc_usd_hourly_recent.csv")
        print(f"    [合成] BTC 小时线合成数据 {len(df)} 行（default_rng(42)）")
        return df, True


def load_csi300():
    """沪深300（工作区公共缓存，只读）。注意：该缓存实为 510300 ETF 净值，算收益没问题。"""
    if os.path.exists(CSI300_CACHE):
        df = pd.read_csv(CSI300_CACHE, encoding="utf-8-sig",
                         parse_dates=["date"], index_col="date")
        df = df.dropna(subset=["close"])  # 缓存尾部有一行 OHLC 全空的记录，剔除
        df = df[~df.index.duplicated(keep="last")].sort_index()
        print(f"    [缓存] 沪深300（实为510300 ETF净值）：{len(df)} 行（{df.index[0].date()} ~ {df.index[-1].date()}）")
        return df, False
    print("    [警告] 找不到工作区沪深300缓存，改用合成数据兜底（已标注）！")
    rng = np.random.default_rng(SEED)
    days = pd.bdate_range(START, pd.Timestamp.today().normalize())
    close = 4000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.013, len(days))))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 0.0}, index=days)
    df.index.name = "date"
    SYNTH.append("index_000300_沪深300(合成)")
    return df, True


# ----------------------------------------------------------------------
# 通用统计
# ----------------------------------------------------------------------
def perf_stats(returns, ann_factor):
    """年化收益(CAGR) / 年化波动 / 夏普(统一 rf=2%：(年化收益−0.02)÷年化波动；年化因子 √365/√244 不变) / 最大回撤 / 累计倍数。"""
    r = returns.dropna()
    nav = (1.0 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = nav.iloc[-1] ** (1.0 / years) - 1.0
    vol = r.std() * np.sqrt(ann_factor)
    sharpe = (cagr - 0.02) / vol if vol > 0 else np.nan  # 统一口径 rf=2%：年化收益扣 2%（等价日度扣 0.02/365）后再除以年化波动
    mdd = (nav / nav.cummax() - 1.0).min()
    return {"年化收益": cagr, "年化波动": vol, "夏普": sharpe,
            "最大回撤": mdd, "累计倍数": nav.iloc[-1], "年数": years}


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    [图] 已保存 outputs/{name}")


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    banner("第22课 · 加密货币与 7×24 市场（随机种子 42）")

    # ============ 一、数据加载与区间对齐 ============
    banner("一、数据加载（先缓存后下载，失败合成兜底）")
    btc, _ = load_daily("BTC-USD", "btc_usd_daily.csv", synth_btc_daily)
    eth, _ = load_daily("ETH-USD", "eth_usd_daily.csv", synth_eth_daily)
    csi, _ = load_csi300()
    hh, _ = load_btc_hourly()
    if SYNTH:
        print(f"    >>> 注意：以下数据为合成数据（已标注），结论仅供参考：{SYNTH}")

    # 截取到统一研究区间（三者共同的起止日期；沪深300只在工作日有数据）
    btc = btc[btc.index >= START]
    eth = eth[eth.index >= START]
    csi = csi[csi.index >= START]
    d_start = max(btc.index[0], eth.index[0], csi.index[0])
    d_end = min(btc.index[-1], eth.index[-1], csi.index[-1])
    btc_c, eth_c, csi_c = btc.loc[d_start:d_end], eth.loc[d_start:d_end], csi.loc[d_start:d_end]
    print(f"    共同研究区间：{d_start.date()} ~ {d_end.date()}"
          f"（BTC {len(btc_c)}个自然日 vs 沪深300 {len(csi_c)}个交易日）")
    add_metric("共同研究区间", f"{d_start.date()}~{d_end.date()}", "三资产共同起止日期")

    btc_r = btc_c["close"].pct_change().dropna()
    eth_r = eth_c["close"].pct_change().dropna()
    csi_r = csi_c["close"].pct_change().dropna()

    # ============ 二、波动率画像 ============
    banner("二、波动率画像：BTC vs 沪深300（对照）")
    btc_vol365 = btc_r.std() * np.sqrt(ANN_CRYPTO)   # 正确：7×24 → √365
    btc_vol244 = btc_r.std() * np.sqrt(ANN_ASHARE)   # 错误示范：误用 A 股的 √244
    csi_vol = csi_r.std() * np.sqrt(ANN_ASHARE)
    eth_vol = eth_r.std() * np.sqrt(ANN_CRYPTO)
    print(f"    BTC   年化波动(√365, 正确) = {pct(btc_vol365)}")
    print(f"    BTC   年化波动(√244, 错误) = {pct(btc_vol244)}  ← 低估约 {1 - btc_vol244 / btc_vol365:.0%}")
    print(f"    ETH   年化波动(√365)      = {pct(eth_vol)}")
    print(f"    沪深300 年化波动(√244)     = {pct(csi_vol)}")
    print(f"    → BTC 波动是沪深300的 {btc_vol365 / csi_vol:.1f} 倍")
    add_metric("BTC年化波动(√365)", f"{btc_vol365:.2%}", "7×24市场正确年化因子")
    add_metric("BTC年化波动(误用√244)", f"{btc_vol244:.2%}", "错误示范：低估约244/365开方")
    add_metric("ETH年化波动(√365)", f"{eth_vol:.2%}", "")
    add_metric("沪深300年化波动(√244)", f"{csi_vol:.2%}", "")
    add_metric("BTC/沪深300波动倍数", f"{btc_vol365 / csi_vol:.2f}x", "")

    btc_worst = btc_r.min()
    csi_worst = csi_r.min()
    print(f"    单日最大跌幅：BTC {pct(btc_worst, 2)}（{btc_r.idxmin().date()}）"
          f" vs 沪深300 {pct(csi_worst, 2)}（{csi_r.idxmin().date()}）")
    add_metric("BTC单日最大跌幅", f"{btc_worst:.2%}", str(btc_r.idxmin().date()))
    add_metric("沪深300单日最大跌幅", f"{csi_worst:.2%}", str(csi_r.idxmin().date()))

    btc_var99 = np.percentile(btc_r, 1)
    csi_var99 = np.percentile(csi_r, 1)
    print(f"    历史VaR99（日）：BTC {pct(btc_var99, 2)} vs 沪深300 {pct(csi_var99, 2)}"
          f"（含义：BTC 约每100天有1天跌超 {abs(btc_var99) * 100:.1f}%）")
    add_metric("BTC历史VaR99(日)", f"{btc_var99:.2%}", "日收益1%分位数")
    add_metric("沪深300历史VaR99(日)", f"{csi_var99:.2%}", "日收益1%分位数")

    # 图01：净值对比（三者共同交易日对齐，注意周末BTC被剔除——相关性研究必须如此）
    join_close = pd.concat([btc_c["close"], eth_c["close"], csi_c["close"]], axis=1).dropna()
    join_close.columns = ["BTC-USD", "ETH-USD", "沪深300"]
    nav = join_close / join_close.iloc[0]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(nav.index, nav["BTC-USD"], color="#f7931a", lw=1.4, label=f"BTC（期末 {nav['BTC-USD'].iloc[-1]:.2f}x）")
    ax.plot(nav.index, nav["ETH-USD"], color="#627eea", lw=1.1, alpha=0.85, label=f"ETH（期末 {nav['ETH-USD'].iloc[-1]:.2f}x）")
    ax.plot(nav.index, nav["沪深300"], color="#c0392b", lw=1.4, label=f"沪深300（期末 {nav['沪深300'].iloc[-1]:.2f}x）")
    ax.set_yscale("log")
    ax.set_title(f"BTC / ETH / 沪深300 净值对比（{d_start.date()}=1，对数轴，仅共同交易日）")
    ax.set_ylabel("净值（对数）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    save_fig(fig, "01_BTC与沪深300净值对比.png")

    # 图02：逐年波动率（市场成熟化：波动是否在下降）
    yr_btc = btc_r.groupby(btc_r.index.year).std() * np.sqrt(ANN_CRYPTO)
    yr_csi = csi_r.groupby(csi_r.index.year).std() * np.sqrt(ANN_ASHARE)
    years = sorted(set(yr_btc.index) | set(yr_csi.index))
    x = np.arange(len(years))
    yr_btc_pct = [yr_btc.get(y, np.nan) * 100 for y in years]
    yr_csi_pct = [yr_csi.get(y, np.nan) * 100 for y in years]
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - 0.2, yr_btc_pct, width=0.4, color="#f7931a", label="BTC（√365）")
    b2 = ax.bar(x + 0.2, yr_csi_pct, width=0.4, color="#c0392b", label="沪深300（√244）")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{h:.0f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(y) for y in years])
    ax.set_ylabel("年化波动率（%）")
    ax.set_title("BTC 逐年波动率 2018→2026：市场成熟化了吗？（2026为不完整年，单位%）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save_fig(fig, "02_BTC逐年波动率.png")
    for y in years:
        add_metric(f"BTC年化波动_{y}年", f"{yr_btc.get(y, float('nan')):.1%}",
                   "2026为不完整年" if y == 2026 else "")

    # ============ 三、日历效应（7×24 特色）：星期 × 小时 ============
    banner("三、日历效应：BTC 最近730天 星期×小时 热力图（北京时间）")
    hr_ret = hh["close"].pct_change().dropna()
    wd = hr_ret.index.weekday          # 0=周一 ... 6=周日（北京时间）
    hr = hr_ret.index.hour
    wd_name = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    mean_p = hr_ret.groupby([wd, hr]).mean().unstack().reindex(index=range(7), columns=range(24))
    std_p = hr_ret.groupby([wd, hr]).std().unstack().reindex(index=range(7), columns=range(24))
    cnt_p = hr_ret.groupby([wd, hr]).count().unstack().reindex(index=range(7), columns=range(24))

    # 显著性说话：168 个格子按 5% 水平约有 8 个纯靠运气"显著"
    sig_cnt = 0
    for i in range(7):
        for j in range(24):
            n, m, s = cnt_p.iloc[i, j], mean_p.iloc[i, j], std_p.iloc[i, j]
            if n >= 30 and s > 0 and abs(m) / (s / np.sqrt(n)) > 2:
                sig_cnt += 1
    n_per_cell = int(cnt_p.stack().median())
    print(f"    每格样本约 {n_per_cell} 条；168格中 |t|>2 的有 {sig_cnt} 个"
          f"（纯随机的期望约 {168 * 0.05:.0f} 个 → {'难以排除运气' if sig_cnt <= 12 else '存在一定结构'}）")
    add_metric("星期×小时显著格子数(|t|>2)", f"{sig_cnt}/168", "随机期望约8个")

    weekend = wd >= 5
    r_wkd, r_wke = hr_ret[~weekend], hr_ret[weekend]
    u_p = stats.mannwhitneyu(r_wke.abs(), r_wkd.abs(), alternative="two-sided").pvalue
    t_p = stats.ttest_ind(r_wke, r_wkd, equal_var=False).pvalue
    vol_ratio = r_wke.std() / r_wkd.std()
    vol_mean_ratio = r_wke.abs().mean() / r_wkd.abs().mean()
    vol_col = hh.loc[hr_ret.index, "volume"]
    v_wkd, v_wke = vol_col[~weekend].mean(), vol_col[weekend].mean()
    print(f"    周末 vs 工作日（小时级，北京时间）：")
    print(f"      波动(小时std)：周末 {r_wke.std():.4%} / 工作日 {r_wkd.std():.4%}，"
          f"比值 {vol_ratio:.2f}，Mann-Whitney p={u_p:.3g}")
    print(f"      平均绝对收益：周末/工作日 = {vol_mean_ratio:.2f}")
    print(f"      平均小时成交量：周末/工作日 = {v_wke / v_wkd:.2f}（流动性稀薄程度）")
    print(f"      平均收益差异 t 检验 p={t_p:.3g} → "
          f"{'周末收益系统性≠工作日' if t_p < 0.05 else '看不出周末收益系统性偏离0或不同于工作日'}")
    add_metric("周末/工作日小时波动比", f"{vol_ratio:.2f}", f"Mann-Whitney p={u_p:.3g}")
    add_metric("周末/工作日小时量比", f"{v_wke / v_wkd:.2f}", "成交量口径，流动性稀薄程度")
    add_metric("周末vs工作日收益t检验p", f"{t_p:.3g}", "")

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    bp = mean_p * 1e4  # 万分之一/小时
    vmax = np.nanmax(np.abs(bp.values))
    im0 = axes[0].imshow(bp.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(7):
        for j in range(24):
            v = bp.iloc[i, j]
            if np.isfinite(v):
                axes[0].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5)
    axes[0].set_title(f"平均收益（万分之一/小时）\n最近730天·北京时间·每格约{n_per_cell}样本")
    im1 = axes[1].imshow(std_p.values * 100, cmap="YlOrRd", aspect="auto")
    for i in range(7):
        for j in range(24):
            v = std_p.iloc[i, j]
            if np.isfinite(v):
                axes[1].text(j, i, f"{v * 100:.2f}", ha="center", va="center", fontsize=5.5)
    axes[1].set_title("波动（小时收益率标准差 %）")
    for ax_i, im in zip(axes, (im0, im1)):
        ax_i.set_xticks(range(0, 24, 2), [str(h) for h in range(0, 24, 2)], fontsize=8)
        ax_i.set_yticks(range(7), wd_name, fontsize=9)
        ax_i.set_xlabel("小时（北京时间）")
        fig.colorbar(im, ax=ax_i, shrink=0.85)
    save_fig(fig, "03_星期小时热力图.png")

    # ============ 四、避险叙事检验：滚动相关 ============
    banner("四、避险叙事检验：BTC vs 沪深300 滚动90日相关")
    jr = pd.concat([btc_r, csi_r], axis=1).dropna()
    jr.columns = ["BTC", "沪深300"]
    roll = jr["BTC"].rolling(90).corr(jr["沪深300"]).dropna()
    pre = roll[roll.index < "2022-01-01"].mean()
    post = roll[roll.index >= "2022-01-01"].mean()
    latest = roll.iloc[-1]
    print(f"    滚动90日相关：2022前均值 {pre:.2f} / 2022后均值 {post:.2f} / 当前 {latest:.2f}")
    print(f"    全样本相关 {jr['BTC'].corr(jr['沪深300']):.2f}；"
          f"BTC-ETH 相关 {btc_r.corr(eth_r):.2f}")
    add_metric("滚动90日相关_2022前均值", f"{pre:.2f}", "")
    add_metric("滚动90日相关_2022后均值", f"{post:.2f}", "")
    add_metric("滚动90日相关_当前", f"{latest:.2f}", str(roll.index[-1].date()))
    add_metric("全样本日收益相关_BTC-沪深300", f"{jr['BTC'].corr(jr['沪深300']):.2f}", "")
    add_metric("全样本日收益相关_BTC-ETH", f"{btc_r.corr(eth_r):.2f}", "")

    # 条件相关：A股最惨的5%交易日里，BTC 在干什么？
    thr = jr["沪深300"].quantile(0.05)
    bad_days = jr[jr["沪深300"] <= thr]
    print(f"    沪深300最差的5%交易日（n={len(bad_days)}，日均 {pct(bad_days['沪深300'].mean(), 2)}）："
          f"BTC 平均 {pct(bad_days['BTC'].mean(), 2)} → "
          f"{'并未逆势护盘' if bad_days['BTC'].mean() <= 0 else '略有正收益'}")
    add_metric("A股最差5%交易日BTC平均收益", f"{bad_days['BTC'].mean():.2%}", f"n={len(bad_days)}")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(roll.index, roll.values, color="#2c3e50", lw=1.1, label="滚动90日相关系数")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(pre, color="#27ae60", ls="--", lw=1, label=f"2022前均值 {pre:.2f}")
    ax.axhline(post, color="#c0392b", ls="--", lw=1, label=f"2022后均值 {post:.2f}")
    ax.axvspan(pd.Timestamp("2022-01-01"), roll.index[-1], color="grey", alpha=0.08)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(f"BTC 与 沪深300 滚动90日相关系数（共同交易日，当前 {latest:.2f}）——避险叙事的数据检验")
    ax.set_ylabel("相关系数")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    save_fig(fig, "04_滚动相关系数.png")

    # ============ 五、趋势策略跨市场对比：同一套 20/60 双均线 ============
    banner("五、双均线(20/60)跨市场对比（BTC万5/单边，沪深300ETF万1/单边）")

    def dual_ma(px, cost):
        """昨收信号今日执行（无前视）：MA20>MA60 持有，否则空仓；换仓收单边成本。"""
        ret = px.pct_change()
        pos = (px.rolling(20).mean() > px.rolling(60).mean()).astype(float).shift(1)
        pos = pos.fillna(0.0)
        trade = pos.diff().abs().fillna(0.0)   # 每次换仓 1 次单边
        return pos * ret - cost * trade, trade.sum()

    btc_strat, btc_trades = dual_ma(btc_c["close"], COST_BTC)
    csi_strat, csi_trades = dual_ma(csi_c["close"], COST_CSI)
    rows = []
    for mkt, bh, st, tr in (("BTC", btc_r, btc_strat, btc_trades),
                            ("沪深300", csi_r, csi_strat, csi_trades)):
        s_bh, s_st = perf_stats(bh, ANN_CRYPTO if mkt == "BTC" else ANN_ASHARE), \
                     perf_stats(st, ANN_CRYPTO if mkt == "BTC" else ANN_ASHARE)
        rows.append((mkt, "买入持有", s_bh, 0))
        rows.append((mkt, "双均线20/60", s_st, tr))
    header = f"{'市场':<8}{'策略':<12}{'年化收益':>9}{'年化波动':>9}{'夏普':>7}{'最大回撤':>9}{'单边交易':>8}"
    print("    " + header)
    print("    " + "-" * len(header))
    for mkt, sname, s, tr in rows:
        print(f"    {mkt:<8}{sname:<12}{pct(s['年化收益']):>9}{pct(s['年化波动']):>9}"
              f"{s['夏普']:>7.2f}{pct(s['最大回撤']):>9}{int(tr):>8}")
    print("    注：两个市场日历不同（BTC 365天/年 vs A股约244天/年），年化均按各自日历因子折算，"
          "时间跨度相同（共同区间）。")
    for mkt, sname, s, tr in rows:
        add_metric(f"{mkt}-{sname}-年化收益", f"{s['年化收益']:.1%}", "")
        add_metric(f"{mkt}-{sname}-年化波动", f"{s['年化波动']:.1%}", "")
        add_metric(f"{mkt}-{sname}-夏普", f"{s['夏普']:.2f}", "")
        add_metric(f"{mkt}-{sname}-最大回撤", f"{s['最大回撤']:.1%}", "")
        add_metric(f"{mkt}-{sname}-单边交易次数", f"{int(tr)}", "")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    panels = [(f"BTC（成本万5/单边，√365年化，{d_start.date()}~{d_end.date()}）", btc_r, btc_strat, "#f7931a", 365),
              (f"沪深300（成本万1/单边，√244年化，{d_start.date()}~{d_end.date()}）", csi_r, csi_strat, "#c0392b", 244)]
    for ax, (title, bh, st, color, fac) in zip(axes, panels):
        s_bh, s_st = perf_stats(bh, fac), perf_stats(st, fac)
        nav_bh = (1 + bh).cumprod()
        nav_st = (1 + st.dropna()).cumprod()
        ax.plot(nav_bh.index, nav_bh.values, color=color, lw=1.2,
                label=f"买入持有（年化 {pct(s_bh['年化收益'])}，夏普 {s_bh['夏普']:.2f}，回撤 {pct(s_bh['最大回撤'])}）")
        ax.plot(nav_st.index, nav_st.values, color="#2c3e50", lw=1.2,
                label=f"双均线20/60（年化 {pct(s_st['年化收益'])}，夏普 {s_st['夏普']:.2f}，回撤 {pct(s_st['最大回撤'])}）")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    save_fig(fig, "05_双均线跨市场对比.png")

    # ============ 六、稳健性提示 ============
    banner("六、稳健性提示（写进 README 的「坑」）")
    ratio = np.sqrt(ANN_CRYPTO / ANN_ASHARE)
    print(f"    1) 年化因子：BTC 若误用 √244，年化波动会从 {pct(btc_vol365)} 低估到 {pct(btc_vol244)}"
          f"（差 √(365/244)≈{ratio:.3f} 倍）。7×24 市场必须用 √365。")
    print(f"    2) 数据质量：雅虎的 BTC 价格是多家交易所聚合，不是单一可成交价；"
          f"瞬时「插针」（极端高低价）在交易所间差异极大，用 high/low 做止损回测要非常小心。")
    print(f"    3) 本课回测是无杠杆现货假设：现实里加密永续合约有资金费率（ funding rate ），"
          f"杠杆策略的资金费率损耗/收益可能改写结论。")
    print(f"    4) 沪深300缓存实为 510300 ETF 净值（工作区 data/README 已注明），算收益率没问题，"
          f"但不能当指数点位用。")
    add_metric("√365/√244年化因子比", f"{ratio:.3f}", "误用年化因子的低估幅度")

    # ============ 输出 metrics.csv ============
    pd.DataFrame(METRICS, columns=["指标", "数值", "说明"]).to_csv(
        os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
    print(f"\n    [表] 已保存 outputs/metrics.csv（{len(METRICS)} 条指标，utf-8-sig）")

    banner(f"运行完成：耗时 {time.time() - t0:.1f} 秒。全部图表见 outputs/，指标见 metrics.csv。")


if __name__ == "__main__":
    main()
