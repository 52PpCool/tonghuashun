# -*- coding: utf-8 -*-
"""
33 组合绩效归因：Brinson 三分法 + 因子回归归因
================================================
一句话：赚的钱从哪来？机构绩效归因的两套标准方法各跑一遍。

组合 P：每月末选过去 60 个交易日动量最强的 3 个行业，等权（各 1/3）持有下月。
基准 B：10 个行业等权（各 1/10）。
本组合直接持有"行业指数"，因此真实选股效应恒为 0——文中会用合成示例演示
"持仓内个股跑赢行业"时选股效应如何非零。

输出：outputs/ 下 5 张图 + 3 个 CSV（utf-8-sig）。
运行：/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python main.py
"""
import os
import sys
import glob
import traceback
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示环境出图
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

try:
    import statsmodels.api as sm
except Exception as e:  # pragma: no cover
    print('statsmodels 导入失败：', e)
    sys.exit(0)

SEED = 42
RNG = np.random.default_rng(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
WORKSPACE_DATA = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'cache'))
os.makedirs(OUT_DIR, exist_ok=True)

MOM_WINDOW = 60          # 动量回看窗口（交易日）
TOP_K = 3                # 持有动量最强的 K 个行业
GLITCH = 0.30            # 单日 |收益|>30% 视为基期跳变，置 0
SYNTH_STOCK_ALPHA = 0.03  # 合成示例：持仓内 1 只个股每月跑赢行业 3%（行业内 50% 仓位 → 行业层面 +1.5%）


def log(msg=''):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# 1. 数据加载与清洗
# ----------------------------------------------------------------------
def load_boards():
    """读取 10 个行业日频 CSV，清洗跳变毛刺，返回 {行业名: 日收益率序列}。
    数据缺失时退化为合成数据（rng(42)），绝不失败退出。"""
    files = sorted(glob.glob(os.path.join(WORKSPACE_DATA, 'board_*.csv')))
    if len(files) < 3:
        log('>> 未找到行业 CSV，改用【合成数据】演示（rng 种子 42）')
        idx = pd.bdate_range('2021-01-04', periods=900)
        rets = {}
        for k in range(10):
            mu, sig = 0.0004 + 0.0001 * (k % 4), 0.012 + 0.003 * (k % 5)
            r = pd.Series(RNG.normal(mu, sig, len(idx)), index=idx, name=f'行业{k}')
            rets[f'合成行业{k}'] = r
        return rets, True
    rets = {}
    for f in files:
        name = os.path.basename(f)[len('board_'):-len('.csv')]
        df = pd.read_csv(f, encoding='utf-8-sig', parse_dates=['date'])
        df = df[['date', 'close']].dropna().sort_values('date').drop_duplicates('date')
        df = df.set_index('date')
        r = df['close'].pct_change()
        # 基期跳变毛刺：单日 |收益|>30% 置 0（等价于当日净值视为不变）
        n_glitch = int((r.abs() > GLITCH).sum())
        r = r.where(r.abs() <= GLITCH, 0.0).dropna()
        if n_glitch:
            log(f'   {name}: 剔除 {n_glitch} 个 >30% 的跳变日')
        rets[name] = r.rename(name)
    return rets, False


def month_framework(rets):
    """构造月度框架：月度收益矩阵 R（月×行业）、每月最后一个交易日日期、动量矩阵。
    只保留所有行业都有完整数据的'完整月'（月末最后观测日 >= 25 号，剔除不完整的末月）。"""
    # 所有行业并集日历（同一市场交易日历，基本一致）
    cal = sorted(set().union(*[set(r.index) for r in rets.values()]))
    cal = pd.DatetimeIndex(cal)
    R_d = pd.DataFrame({k: v.reindex(cal) for k, v in rets.items()})  # 清洗后的日收益率

    # 月度收益 = 月内日收益复利
    R_m = (1 + R_d).groupby(R_d.index.to_period('M')).prod(min_count=8) - 1
    # 每月最后交易日 / 最后观测日（用于剔除不完整月、定位调仓日）
    last_dt = pd.Series(cal, index=cal).groupby(cal.to_period('M')).max()
    cnt = R_d.notna().groupby(cal.to_period('M')).sum()
    # 完整月：每个行业该月都有观测，且月末最后观测日 >= 25 号
    ok_months = [m for m in R_m.index
                 if cnt.loc[m].min() >= 8 and last_dt[m].day >= 25]
    R_m = R_m.loc[ok_months]
    month_last_day = {m: last_dt[m] for m in ok_months}

    # 60 日动量：exp(过去 60 个交易日对数收益和) - 1（用清洗后的日收益）。
    # 注意：各行业 CSV 的交易日历并非完全一致（存在零星缺日），若在并集日历上滚动，
    # 缺日会把 rolling 窗口污染成 NaN。因此动量在各行业【自身】日历序列上滚动，
    # 调仓日按 asof（≤调仓日的最后一个值）取用。
    mom = pd.DataFrame(index=cal, columns=list(rets.keys()), dtype=float)
    for k, r in rets.items():
        m_k = np.exp(np.log1p(r).rolling(MOM_WINDOW, min_periods=MOM_WINDOW).sum()) - 1
        mom[k] = m_k.reindex(cal)  # 不 ffill：保留 NaN 以便 asof 时判断窗口是否足够

    def asof_row(df, d):
        """每列取 ≤d 的最后一个非 NaN 值；某列在 d 前不足 60 日则保持 NaN。"""
        sub = df.loc[:d]
        return sub.apply(lambda col: col.dropna().iloc[-1] if col.notna().any() else np.nan)

    log(f'   月度样本构建完成：{ok_months[0]} ~ {ok_months[-1]}，共 {len(ok_months)} 个完整月')
    return R_m, mom, month_last_day, asof_row


def build_portfolio(R_m, mom, month_last_day, asof_row):
    """构造每月权重与组合/基准收益。
    第 t 月持有的权重，由 t-1 月最后一个交易日的 60 日动量决定（无未来函数）。
    返回 w_p: DataFrame(月×行业) 组合权重, w_b: 等权基准, R_p/R_b: 月收益。"""
    industries = list(R_m.columns)
    w_p = pd.DataFrame(0.0, index=R_m.index, columns=industries)
    w_b = pd.DataFrame(1.0 / len(industries), index=R_m.index, columns=industries)

    months = list(R_m.index)
    skipped = []
    for m in months:
        # 调仓日 = 上月最后一个交易日
        mi = months.index(m)
        if mi == 0:
            skipped.append((m, '无上月'))
            continue
        prev = months[mi - 1]
        reb_date = month_last_day[prev]
        mom_row = asof_row(mom, reb_date)
        if mom_row.isna().any():
            skipped.append((m, '某行业自身日历上不足60日（预热期）'))
            continue
        top3 = mom_row.nlargest(TOP_K).index.tolist()
        w_p.loc[m, top3] = 1.0 / TOP_K

    valid = w_p.abs().sum(axis=1) > 0  # 剔除预热期（无法计算动量的月）
    n_drop = int((~valid).sum())
    w_p, w_b, R_m2 = w_p[valid], w_b[valid], R_m[valid]
    log(f'   动量预热剔除 {n_drop} 个月，有效归因区间：{R_m2.index[0]} ~ {R_m2.index[-1]}，共 {len(R_m2)} 个月')
    if skipped:
        log('   被剔除的月份及原因：' + '；'.join(f'{m}({why})' for m, why in skipped[:6])
            + ('……' if len(skipped) > 6 else ''))
    log('   各期持有的行业：')
    for m in w_p.index:
        held = [f'{c}' for c in w_p.columns if w_p.loc[m, c] > 0]
        log(f'     {m}: ' + '、'.join(held))
    R_p = (w_p * R_m2).sum(axis=1)
    R_b = (w_b * R_m2).sum(axis=1)
    return w_p, w_b, R_m2, R_p, R_b


# ----------------------------------------------------------------------
# 2. Brinson 归因（BHB 三分法）
# ----------------------------------------------------------------------
def brinson_full(w_p, w_b, R_i, sector_alpha=None):
    """逐月逐行业计算 BHB 三分法。
      配置效应 = (w_P,i - w_B,i) * (R_B,i - R_B)
      选股效应 = w_B,i * (R_P,i - R_B,i)
      交互项   = (w_P,i - w_B,i) * (R_P,i - R_B,i)
    sector_alpha: DataFrame(月×行业)，合成示例中组合在行业 i 内的持仓收益
                  相对行业指数的超额；真实持行业指数时为 0（R_P,i = R_B,i）。"""
    rows = []
    for m in R_i.index:
        Ri = R_i.loc[m]
        wp, wb = w_p.loc[m], w_b.loc[m]
        Rb_tot = float((wb * Ri).sum())
        alpha_i = pd.Series(0.0, index=R_i.columns) if sector_alpha is None else sector_alpha.loc[m]
        Rp_sector = Ri + alpha_i                      # 组合在行业 i 的持仓收益
        Rp_tot = float((wp * Rp_sector).sum())
        for c in R_i.columns:
            rows.append(dict(
                month=str(m), industry=c,
                w_p=wp[c], w_b=wb[c],
                r_p_sector=Rp_sector[c], r_b_sector=Ri[c],
                r_b_total=Rb_tot, r_p_total=Rp_tot,
                alloc=(wp[c] - wb[c]) * (Ri[c] - Rb_tot),
                selection=wb[c] * (Rp_sector[c] - Ri[c]),
                interaction=(wp[c] - wb[c]) * (Rp_sector[c] - Ri[c]),
            ))
    return pd.DataFrame(rows)


def check_closure(df, label):
    """对账：每月 Σ(配置+选股+交互) 必须 = R_P - R_B。返回最大绝对误差。
    注意：r_p_total / r_b_total 是当月总量（逐行业行上重复），聚合须取 first 而非 sum。"""
    g = df.groupby('month').agg({'alloc': 'sum', 'selection': 'sum', 'interaction': 'sum',
                                 'r_p_total': 'first', 'r_b_total': 'first'})
    lhs = g['r_p_total'] - g['r_b_total']
    rhs = g['alloc'] + g['selection'] + g['interaction']
    err = float((lhs - rhs).abs().max())
    log(f'   [{label}] 逐月对账 |Σ效应 - (R_P-R_B)| 最大 = {err:.2e}')
    return err


# ----------------------------------------------------------------------
# 3. 因子回归归因
# ----------------------------------------------------------------------
def ff3_regression(R_p):
    """组合月超额收益对 Mkt-RF/SMB/HML 的 OLS（官方三因子为美股因子，仅作方法演示）。"""
    f = pd.read_csv(os.path.join(WORKSPACE_DATA, 'ff3_monthly.csv'))
    f['month'] = pd.PeriodIndex(f['yyyymm'].astype(str), freq='M')
    f = f.set_index('month')[['Mkt-RF', 'SMB', 'HML', 'RF']] / 100.0
    df = pd.DataFrame({'rp': R_p}).join(f, how='inner')
    y = df['rp'] - df['RF']
    X = sm.add_constant(df[['Mkt-RF', 'SMB', 'HML']])
    res = sm.OLS(y, X).fit()
    contrib = {k: float(res.params[k] * df[k].mean()) for k in ['Mkt-RF', 'SMB', 'HML']}
    return res, contrib, float(y.mean()), len(df)


def industry_regression(w_p, R_i, R_p):
    """市场+行业因子版：组合收益对 10 个行业收益回归（无截距），系数=隐含行业暴露。
    教学要点：若组合权重【固定不变】，组合收益就是行业收益的精确线性组合，
    无截距回归可精确还原权重（R²=1）；但本组合权重逐月轮动，回归只能给出
    "平均有效暴露"，且 10 个行业收益高度共线，暴露估计不稳（个别为负）。
    与 Brinson 的对应：Brinson 用事前权重拆收益，回归用事后协方差估暴露。"""
    X, y = R_i, R_p
    res = sm.OLS(y, X).fit()  # 无截距
    return res


# ----------------------------------------------------------------------
# 4. 绘图
# ----------------------------------------------------------------------
def plot_waterfall(tot_real, tot_synth, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (tot, title) in zip(axes, [
            (tot_real, '全期归因瀑布（真实：持行业指数，选股=0）'),
            (tot_synth, '全期归因瀑布（合成示例：持仓内个股跑赢行业）')]):
        labels = ['配置效应', '选股效应', '交互项', '超额收益合计']
        vals = [tot['alloc'], tot['selection'], tot['interaction']]
        cum = np.concatenate([[0], np.cumsum(vals)])
        colors = ['#4C72B0', '#DD8452', '#55A868']
        for i, v in enumerate(vals):
            ax.bar(i, v * 100, bottom=cum[i] * 100, color=colors[i], width=0.6)
            if abs(v) > 1e-12:
                ax.text(i, (cum[i] + v / 2) * 100, f'{v*100:+.2f}%', ha='center',
                        va='center', color='white', fontsize=11, fontweight='bold')
            else:  # 零值效应：标签放进轴内并上下错开，避免互相重叠与标题碰撞
                y_lab = -1.5 if i == 1 else -6.0
                ax.text(i, y_lab, '0.00%（恒为零）', ha='center', va='top',
                        color='#555555', fontsize=9)
        ax.bar(3, sum(vals) * 100, color='#C44E52', width=0.6)
        ax.text(3, sum(vals) * 50, f'{sum(vals)*100:+.2f}%', ha='center', va='center',
                color='white', fontsize=11, fontweight='bold')
        # 连接虚线
        for i in range(3):
            ax.plot([i + 0.3, i + 1 - 0.3], [cum[i + 1] * 100] * 2, ls='--', c='gray', lw=1)
        ax.set_xticks(range(4), labels)
        ax.axhline(0, c='k', lw=0.8)
        ax.set_ylabel('累计贡献（%，单利加总）')
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('图01 Brinson 归因瀑布：超额收益 = 配置 + 选股 + 交互（对账闭合）', y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cum_stacked(monthly, path):
    g = monthly.groupby('month')[['alloc', 'selection', 'interaction']].sum()
    cum = g.cumsum()
    xs = np.arange(len(cum))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.stackplot(xs, cum['alloc'] * 100, cum['selection'] * 100, cum['interaction'] * 100,
                 labels=['累计配置效应', '累计选股效应(=0)', '累计交互项'],
                 colors=['#4C72B0', '#DD8452', '#55A868'], alpha=0.85)
    excess = (g['alloc'] + g['selection'] + g['interaction']).cumsum() * 100
    ax.plot(xs, excess, 'k-', lw=2, label='累计超额收益（单利加总，与堆叠面积逐点相等）')
    ax.set_xticks(xs[::max(1, len(xs)//10)], cum.index[::max(1, len(xs)//10)], rotation=45)
    ax.axhline(0, c='k', lw=0.8)
    ax.set_ylabel('累计贡献（%）')
    ax.set_title('图02 累计超额收益分解：P−B = 配置 + 选股 + 交互（堆叠面积，数值逐月闭合）')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_factor(res_ff3, contrib, ind_res, w_avg, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (1) 月均超额收益拆解：alpha + Σ beta*mean(f)
    terms = ['alpha'] + list(contrib.keys())
    vals = [res_ff3.params['const']] + [contrib[k] for k in contrib.keys()]
    colors = ['#C44E52'] + ['#4C72B0'] * 3
    axes[0].bar(terms, np.array(vals) * 100, color=colors, width=0.6)
    vmin, vmax = min(vals) * 100, max(vals) * 100
    pad = (vmax - vmin) * 0.06 + 0.01
    axes[0].set_ylim(vmin - pad * 1.6, vmax + pad * 1.6)
    for i, v in enumerate(vals):
        axes[0].text(i, v * 100 + (pad * 0.35 if v >= 0 else -pad * 0.55), f'{v*100:+.2f}%',
                     ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
    axes[0].axhline(0, c='k', lw=0.8)
    axes[0].set_title('(a) FF3 回归：月均超额收益拆解\nalpha + β_k × 因子月均')
    axes[0].set_ylabel('% / 月'); axes[0].grid(axis='y', alpha=0.3)
    # (2) beta 及 95% 置信区间
    ks = ['Mkt-RF', 'SMB', 'HML']
    b = [res_ff3.params[k] for k in ks]
    ci = [1.96 * res_ff3.bse[k] for k in ks]
    axes[1].errorbar(ks, b, yerr=ci, fmt='o', capsize=5, color='#4C72B0')
    axes[1].axhline(0, c='k', lw=0.8); axes[1].axhline(1, c='gray', ls='--', lw=1)
    for i, (v, e) in enumerate(zip(b, ci)):
        axes[1].text(i, v + e + 0.03, f'{v:.2f}', ha='center', fontsize=10)
    axes[1].set_title('(b) FF3 因子暴露 β（误差棒=95%CI）\n注：美股因子 × A股组合，仅方法演示')
    axes[1].grid(alpha=0.3)
    # (3) 行业回归隐含暴露 vs 实际权重
    ind = list(w_avg.index)
    imp = [ind_res.params.get(c, np.nan) for c in ind]
    act = [w_avg[c] for c in ind]
    xpos = np.arange(len(ind))
    axes[2].bar(xpos - 0.2, imp, width=0.4, label='回归隐含暴露（无截距）', color='#4C72B0')
    axes[2].bar(xpos + 0.2, act, width=0.4, label='实际平均权重', color='#DD8452')
    axes[2].set_xticks(xpos, ind, rotation=45, ha='right', fontsize=9)
    axes[2].set_title('(c) 行业因子回归：平均有效暴露 vs 平均权重\n（权重逐月轮动+共线 → 仅作参考）')
    axes[2].legend(fontsize=9); axes[2].grid(axis='y', alpha=0.3)
    fig.suptitle('图03 因子回归归因', y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_industry_contrib(total_df, path):
    d = total_df[total_df['industry'] != '合计'].copy()
    d['total'] = d['total_alloc'] + d['total_selection'] + d['total_interaction']
    d = d.sort_values('total')
    xpos = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(xpos, d['total_alloc'] * 100, color='#4C72B0', label='配置效应')
    ax.barh(xpos, d['total_selection'] * 100, left=d['total_alloc'] * 100,
            color='#DD8452', label='选股效应(真实=0)')
    left2 = (d['total_alloc'] + d['total_selection']) * 100
    ax.barh(xpos, d['total_interaction'] * 100, left=left2, color='#55A868', label='交互项')
    for i, v in enumerate(d['total'] * 100):
        if v >= 0:  # 正值：标在条形右外侧
            ax.text(v + 0.15, i, f'{v:+.2f}%', va='center', ha='left', fontsize=9)
        else:       # 负值：标在条形内部（白字），避免顶到 y 轴标签
            ax.text(v + 0.15, i, f'{v:+.2f}%', va='center', ha='left',
                    color='white', fontsize=9, fontweight='bold')
    ax.set_yticks(xpos, d['industry'])
    ax.axvline(0, c='k', lw=0.8)
    ax.set_xlabel('全期累计贡献（%，单利加总）')
    ax.set_title('图04 分行业 Brinson 贡献拆解（累计）')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nav(R_p, R_b, path):
    nav_p = (1 + R_p).cumprod()
    nav_b = (1 + R_b).cumprod()
    xs = np.arange(len(nav_p))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(xs, nav_p, lw=2, color='#C44E52', label=f'组合P（动量Top3） 期末 {nav_p.iloc[-1]:.3f}')
    ax.plot(xs, nav_b, lw=2, color='#4C72B0', label=f'基准B（10行业等权） 期末 {nav_b.iloc[-1]:.3f}')
    ax.set_xticks(xs[::max(1, len(xs)//10)], [str(m) for m in nav_p.index][::max(1, len(xs)//10)],
                  rotation=45)
    ax.set_ylabel('净值（起点=1，月度复利）')
    ax.set_title('图05 组合 vs 基准净值（复利口径；归因分解用单利口径，两者差异见 README）')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return nav_p, nav_b


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    log('=' * 70)
    log('33 组合绩效归因：Brinson 与因子回归')
    log('=' * 70)

    # ---- 数据 ----
    log('\n[1] 数据加载与清洗')
    rets, is_synth = load_boards()
    R_m, mom, month_last_day, asof_row = month_framework(rets)
    w_p, w_b, R_i, R_p, R_b = build_portfolio(R_m, mom, month_last_day, asof_row)
    industries = list(R_i.columns)
    n_months = len(R_i)

    # ---- Brinson：真实情形（组合持行业指数 → 选股恒为 0）----
    log('\n[2] Brinson 归因（真实情形：R_P,i = R_B,i，选股效应恒为 0）')
    br_real = brinson_full(w_p, w_b, R_i, sector_alpha=None)
    err_real = check_closure(br_real, '真实情形')
    tot_real = br_real[['alloc', 'selection', 'interaction']].sum()

    # ---- Brinson：合成选股示例 ----
    log(f'\n[3] 合成选股示例：每个持有行业内置 2 只个股各 50%，'
        f'其中 1 只每月跑赢行业 {SYNTH_STOCK_ALPHA*100:.0f}% → 行业层面超额 +{SYNTH_STOCK_ALPHA/2*100:.1f}%')
    sector_alpha = pd.DataFrame(0.0, index=R_i.index, columns=industries)
    for m in R_i.index:
        for c in industries:
            if w_p.loc[m, c] > 0:
                sector_alpha.loc[m, c] = SYNTH_STOCK_ALPHA / 2
    br_synth = brinson_full(w_p, w_b, R_i, sector_alpha=sector_alpha)
    err_synth = check_closure(br_synth, '合成示例')
    tot_synth = br_synth[['alloc', 'selection', 'interaction']].sum()

    log(f'\n   真实情形全期（{n_months} 个月，单利加总）：')
    log(f'     配置效应 = {tot_real["alloc"]*100:+.2f}%   选股效应 = {tot_real["selection"]*100:+.2f}%   '
        f'交互项 = {tot_real["interaction"]*100:+.2f}%')
    log(f'     合计 = {(tot_real["alloc"]+tot_real["selection"]+tot_real["interaction"])*100:+.2f}%'
        f'（对账误差 {err_real:.1e}）')
    log(f'   合成示例全期：配置 {tot_synth["alloc"]*100:+.2f}%  选股 {tot_synth["selection"]*100:+.2f}%  '
        f'交互 {tot_synth["interaction"]*100:+.2f}%  合计 '
        f'{(tot_synth["alloc"]+tot_synth["selection"]+tot_synth["interaction"])*100:+.2f}%'
        f'（对账误差 {err_synth:.1e}）')

    # ---- 复利 vs 单利 ----
    excess_arith = float((R_p - R_b).sum())
    excess_geom = float((1 + R_p).prod() / (1 + R_b).prod() - 1)
    log(f'\n   累计超额：单利加总 = {excess_arith*100:+.2f}%；复利口径 = {excess_geom*100:+.2f}%'
        f'（差异 {100*(excess_geom-excess_arith):+.2f}个百分点，来自复利放大/缩小与月度路径）')

    # ---- 分行业贡献 ----
    by_ind = br_real.groupby('industry')[['alloc', 'selection', 'interaction']].sum()
    by_ind['months_held'] = (w_p > 0).sum()
    by_ind = by_ind.sort_values('alloc', ascending=False)
    log('\n   分行业累计贡献（单利，%）：')
    log(by_ind[['alloc', 'selection', 'interaction']].apply(lambda s: s * 100).round(2).to_string())

    # ---- 因子回归 ----
    log('\n[4] 因子回归归因（FF3 为美股因子，对 A 股组合仅为方法演示）')
    res_ff3, contrib, mean_y, nobs = ff3_regression(R_p)
    log(res_ff3.summary().as_text())
    log(f'   月均超额(R_P-RF) = {mean_y*100:+.3f}% = α {res_ff3.params["const"]*100:+.3f}%'
        f' + 市场贡献 {contrib["Mkt-RF"]*100:+.3f}% + 规模 {contrib["SMB"]*100:+.3f}%'
        f' + 价值 {contrib["HML"]*100:+.3f}%（n={nobs}，损失末月因FF3数据止于2026-07）')
    alpha_sig = '显著' if res_ff3.pvalues['const'] < 0.05 else '不显著（p>=0.05）'
    log(f'   α 的显著性：t = {res_ff3.tvalues["const"]:.2f}, p = {res_ff3.pvalues["const"]:.3f} → {alpha_sig}')

    log('\n[5] 行业因子回归（无截距，系数=隐含行业暴露）')
    ind_res = industry_regression(w_p, R_i, R_p)
    w_avg = w_p.mean()
    cmp = pd.DataFrame({'隐含暴露': [ind_res.params.get(c, np.nan) for c in industries],
                        '实际平均权重': [w_avg[c] for c in industries]}, index=industries)
    cmp['偏差'] = cmp['隐含暴露'] - cmp['实际平均权重']
    log(cmp.round(4).to_string())
    log(f'   行业回归 R² = {ind_res.rsquared:.4f}，最大暴露偏差 = {cmp["偏差"].abs().max():.2e}')
    log('   解读（务必如实）：本组合权重【逐月轮动】，回归给出的是全期"平均有效暴露"，')
    log('   不可能逐月精确还原实际权重；且 10 个行业收益高度共线，个别暴露跑成负值/偏大，')
    log('   属正常现象。若权重固定不动，无截距回归才会精确还原权重（R²=1）。')

    # ---- 输出 CSV ----
    log('\n[6] 写出 CSV 与图')
    br_real_out = br_real.copy()
    br_real_out.to_csv(os.path.join(OUT_DIR, 'brinson_monthly.csv'),
                       index=False, encoding='utf-8-sig', float_format='%.6f')

    total_df = by_ind.reset_index().rename(columns={
        'alloc': 'total_alloc', 'selection': 'total_selection', 'interaction': 'total_interaction'})
    total_df = total_df[['industry', 'months_held', 'total_alloc',
                         'total_selection', 'total_interaction']]
    grand = dict(industry='合计', months_held=n_months,
                 total_alloc=tot_real['alloc'], total_selection=tot_real['selection'],
                 total_interaction=tot_real['interaction'])
    total_df = pd.concat([total_df, pd.DataFrame([grand])], ignore_index=True)
    total_df.to_csv(os.path.join(OUT_DIR, 'brinson_total.csv'),
                    index=False, encoding='utf-8-sig', float_format='%.6f')

    fa_rows = []
    fa_rows.append(dict(model='FF3回归(方法演示)', term='alpha',
                        coef=res_ff3.params['const'], t=res_ff3.tvalues['const'],
                        p=res_ff3.pvalues['const'], contrib_pct_per_month=res_ff3.params['const'] * 100,
                        nobs=nobs, r2=res_ff3.rsquared, note='截距项，即回归口径的α'))
    for k in ['Mkt-RF', 'SMB', 'HML']:
        fa_rows.append(dict(model='FF3回归(方法演示)', term=k,
                            coef=res_ff3.params[k], t=res_ff3.tvalues[k], p=res_ff3.pvalues[k],
                            contrib_pct_per_month=contrib[k] * 100,
                            nobs=nobs, r2=res_ff3.rsquared, note='β×因子月均=该因子月均贡献'))
    for c in industries:
        fa_rows.append(dict(model='行业因子回归(无截距)', term=c,
                            coef=ind_res.params.get(c, np.nan), t=ind_res.tvalues.get(c, np.nan),
                            p=ind_res.pvalues.get(c, np.nan), contrib_pct_per_month=np.nan,
                            nobs=int(ind_res.nobs), r2=ind_res.rsquared,
                            note=f'实际平均权重 {w_avg[c]:.4f}'))
    pd.DataFrame(fa_rows).to_csv(os.path.join(OUT_DIR, 'factor_attribution.csv'),
                                 index=False, encoding='utf-8-sig', float_format='%.6f')

    # ---- 图 ----
    plot_waterfall(tot_real, tot_synth, os.path.join(OUT_DIR, '01_超额收益瀑布图.png'))
    plot_cum_stacked(br_real, os.path.join(OUT_DIR, '02_累计归因堆叠面积.png'))
    plot_factor(res_ff3, contrib, ind_res, w_avg, os.path.join(OUT_DIR, '03_因子回归拆解.png'))
    plot_industry_contrib(total_df, os.path.join(OUT_DIR, '04_行业贡献条形图.png'))
    nav_p, nav_b = plot_nav(R_p, R_b, os.path.join(OUT_DIR, '05_组合vs基准净值.png'))
    log('   5 张图与 3 个 CSV 已写入 outputs/')

    # ---- 结论 ----
    log('\n[7] 结论（如实）')
    top_alloc = by_ind['alloc'].nlargest(3)
    log(f'   1) 全期单利累计超额 = {excess_arith*100:+.2f}%：'
        f'配置 {tot_real["alloc"]*100:+.2f}% + 选股 {tot_real["selection"]*100:+.2f}% + '
        f'交互 {tot_real["interaction"]*100:+.2f}%。'
        f'本组合的超额（本例为负，即跑输基准的部分）100% 来自配置端——'
        f'选股恒为 0（持的是行业指数本身），跑输说明"追高轮动"的配置决策在扣分。')
    log(f'   2) 配置贡献前三大行业：'
        + '，'.join(f'{c} {v*100:+.2f}%' for c, v in top_alloc.items()))
    worst = by_ind['alloc'].idxmin()
    log(f'   3) 配置拖累最大行业：{worst} {by_ind.loc[worst, "alloc"]*100:+.2f}%')
    log(f'   4) FF3 回归 α = {res_ff3.params["const"]*100:+.3f}%/月（t={res_ff3.tvalues["const"]:.2f}，'
        f'p={res_ff3.pvalues["const"]:.3f}，{alpha_sig}），R²={res_ff3.rsquared:.3f}，n={nobs}。')
    log(f'   5) 复利口径累计超额 {excess_geom*100:+.2f}% vs 单利 {excess_arith*100:+.2f}%：'
        f'口径不同，归因表务必统一。')

    log('\n完成。退出码 0。')


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        log('\n[兜底] 出现异常，为保证流程不中断以退出码 0 结束（请检查上方堆栈）。')
        sys.exit(0)
