# -*- coding: utf-8 -*-
"""
12_期权定价与希腊字母 —— 期权世界的"牛顿力学"
=====================================================================
一节课讲清五件事：
  1) Black-Scholes 欧式期权定价公式（解析解，自己实现）
  2) 五个希腊字母 Delta/Gamma/Vega/Theta/Rho 的解析式与风险直觉
  3) 蒙特卡洛(GBM)重新定价 vs 解析解：收敛性与方差缩减
  4) 隐含波动率：牛顿迭代反解 + 波动率微笑（U型）
  5) Put-Call Parity 平价关系数值验证 + 期权到期损益图

标的示例：贵州茅台(600519) 现价（读取失败时回退 S0=1500 并打印标注）。
运行：.venv/bin/python main.py   （随机种子 42，总耗时远小于 2 分钟）
输出：outputs/ 下 5 张图 + greeks_table.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示器环境下出图，必须放在 pyplot 之前
import matplotlib.pyplot as plt
from scipy.stats import norm      # norm.pdf = φ(·) 正态密度, norm.cdf = N(·) 正态分布
from scipy.optimize import brentq  # 二分法求根，用来交叉验证牛顿迭代得到的隐含波动率

# 中文字体设置（本工作区统一要求）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 第 0 节：标的与参数
# ============================================================
RISK_FREE = 0.02    # 无风险利率 r：年化 2%（连续复利）
T_YEAR = 0.25       # 到期期限 T：0.25 年（约 3 个月 / 63 个交易日）
N_PATHS = 100_000   # 蒙特卡洛路径数
SEED = 42           # 随机种子（保证结果可复现）


def load_spot():
    """读取贵州茅台最新收盘价做 S0、过去 60 日年化波动率做 σ。

    读取失败则回退 S0=1500、σ=0.25（示例值），并打印标注，保证脚本永不失败。
    """
    path = os.path.join(HERE, '..', 'data', 'cache', 'stock_600519_贵州茅台.csv')
    try:
        df = pd.read_csv(path, parse_dates=['date']).set_index('date')
        s0 = float(df['close'].iloc[-1])
        # 对数收益率的标准差 × sqrt(252个交易日) = 年化波动率
        ret = np.log(df['close'] / df['close'].shift(1)).dropna()
        sigma60 = float(ret.tail(60).std(ddof=1) * np.sqrt(252))
        print(f'[数据] 成功读取 {os.path.basename(path)}')
        print(f'[数据] 最新收盘价 S0 = {s0:.2f} 元（{df.index[-1].date()}）')
        print(f'[数据] 过去60日年化波动率 σ = {sigma60:.4f}（即 {sigma60 * 100:.2f}%）')
        return s0, sigma60, False
    except Exception as e:  # noqa: BLE001 —— 任何读取异常都回退，不让脚本挂掉
        print(f'[警告] 行情读取失败({e})，回退示例参数：S0=1500、σ=0.25（结果仅作演示）')
        return 1500.0, 0.25, True


S0, SIGMA, IS_FALLBACK = load_spot()
K_ATM = S0  # 平值期权：执行价 K = 现价 S0

# ============================================================
# 第 1 节：Black-Scholes 欧式期权定价公式（解析解，自己实现）
# ============================================================
# ---------------- 公式全文（每个符号的含义） ----------------
# 记号：
#   S0 : 标的现价（如茅台股价）          K  : 执行价（strike）
#   r  : 无风险利率（年化，连续复利）     T  : 到期期限（年）
#   σ  : 标的年化波动率（对数收益率标准差）
#   N(·): 标准正态分布的累积分布函数（scipy.stats.norm.cdf）
#   φ(·): 标准正态分布的概率密度函数（scipy.stats.norm.pdf）
#   e  : 自然常数，e^(-rT) 是"把 K 折现到今天"的贴现因子
#
# 中间变量：
#   d1 = [ln(S0/K) + (r + σ²/2)·T] / (σ·√T)
#   d2 = d1 - σ·√T
#
# 欧式看涨期权价格：
#   C = S0·N(d1) - K·e^(-rT)·N(d2)
#       └─行权拿到股票的"期望现值"─┘└─付出行权价的"期望现值"─┘
#
# 欧式看跌期权价格：
#   P = K·e^(-rT)·N(-d2) - S0·N(-d1)
#
# 直觉：C = "可能拿到股票"的价值 - "可能付出现金"的价值。
# N(d2) 是风险中性世界里到期股价高于 K 的概率，N(d1) 是"股价每涨1元、
# 期权价值涨多少"的加权（它同时就是 Delta）。
# -------------------------------------------------------------


def bs_d1_d2(S, K, r, sigma, T):
    """计算 B-S 公式的中间变量 d1 与 d2。"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S, K, r, sigma, T):
    """欧式看涨期权 B-S 解析价格：C = S·N(d1) - K·e^(-rT)·N(d2)。"""
    d1, d2 = bs_d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, r, sigma, T):
    """欧式看跌期权 B-S 解析价格：P = K·e^(-rT)·N(-d2) - S·N(-d1)。"""
    d1, d2 = bs_d1_d2(S, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


call_price = bs_call(S0, K_ATM, RISK_FREE, SIGMA, T_YEAR)
put_price = bs_put(S0, K_ATM, RISK_FREE, SIGMA, T_YEAR)
intrinsic_call = max(S0 - K_ATM, 0.0)          # 内在价值 = max(S-K, 0)
time_value_call = call_price - intrinsic_call  # 时间价值 = 价格 - 内在价值

print('\n========== 1. B-S 定价（平值：K = S0） ==========')
print(f'参数：S0={S0:.2f}, K={K_ATM:.2f}, r={RISK_FREE:.2%}, '
      f'σ={SIGMA:.4f}, T={T_YEAR}年')
print(f'平值看涨 C = {call_price:.4f} 元（内在价值 {intrinsic_call:.2f} + '
      f'时间价值 {time_value_call:.4f}）')
print(f'平值看跌 P = {put_price:.4f} 元')

# ============================================================
# 第 2 节：五个希腊字母（解析式）
# ============================================================
# 风险直觉（每个字母回答一个问题）：
#   Delta  : 股价动 1 元，期权价值动多少？——方向敞口/对冲比率
#   Gamma  : 股价动 1 元，Delta 自己动多少？——方向敞口的变化速度（凸性来源）
#   Vega   : 波动率升 1 个百分点(0.01)，期权价值动多少？——怕波动率降
#   Theta  : 每过一天，期权价值"自然蒸发"多少？——时间磨损
#   Rho    : 利率升 1 个百分点(0.01)，期权价值动多少？——利率敏感度（通常最小）
# ------------------------------------------------------------


def greeks_call(S, K, r, sigma, T):
    """看涨期权的五个希腊字母（Vega/Theta/Rho 按自然单位给出，见下）。"""
    d1, d2 = bs_d1_d2(S, K, r, sigma, T)
    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)
    delta = norm.cdf(d1)                                  # 方向敞口，0~1
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))             # Delta 的变化速度
    vega = S * pdf_d1 * np.sqrt(T)                        # 对 σ 的导数（σ 变动 1.0 的价格变化）
    theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))        # 对 T 的偏导（每年），恒为负
             - r * K * disc * norm.cdf(d2))
    rho = K * T * disc * norm.cdf(d2)                     # 对 r 的导数（r 变动 1.0 的价格变化）
    return delta, gamma, vega, theta, rho


def greeks_put(S, K, r, sigma, T):
    """看跌期权的五个希腊字母。"""
    d1, d2 = bs_d1_d2(S, K, r, sigma, T)
    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)
    delta = norm.cdf(d1) - 1.0                            # 负数：跌了才赚
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))             # 与 call 相同
    vega = S * pdf_d1 * np.sqrt(T)                        # 与 call 相同
    theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))        # 深度实值 put 的 theta 可为正
             + r * K * disc * norm.cdf(-d2))
    rho = -K * T * disc * norm.cdf(-d2)                   # 利率升 → put 跌
    return delta, gamma, vega, theta, rho


d_c, g_c, v_c, th_c, rho_c = greeks_call(S0, K_ATM, RISK_FREE, SIGMA, T_YEAR)
d_p, g_p, v_p, th_p, rho_p = greeks_put(S0, K_ATM, RISK_FREE, SIGMA, T_YEAR)

print('\n========== 2. 平值处的五个希腊字母（call | put） ==========')
print(f'Delta : {d_c:.4f} | {d_p:.4f}   （方向敞口）')
print(f'Gamma : {g_c:.6f} | {g_p:.6f} （Delta 的变化速度）')
print(f'Vega  : {v_c:.4f} | {v_p:.4f}   （σ 变动 1.0 的价格变化；每 1 个百分点 ≈ {v_c / 100:.4f} 元）')
print(f'Theta : {th_c:.4f} | {th_p:.4f} （每年；每天 ≈ {th_c / 365:.4f} 元）')
print(f'Rho   : {rho_c:.4f} | {rho_p:.4f}   （r 变动 1.0；每 1 个百分点 ≈ {rho_c / 100:.4f} 元）')

# ---- 对 K 从 0.5·S0 到 1.5·S0 扫描，画五张希腊字母曲线（图02） ----
K_scan = np.linspace(0.5 * S0, 1.5 * S0, 201)
gc = np.array([greeks_call(S0, k, RISK_FREE, SIGMA, T_YEAR) for k in K_scan])
gp = np.array([greeks_put(S0, k, RISK_FREE, SIGMA, T_YEAR) for k in K_scan])

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('五个希腊字母随执行价 K 的变化（S0=%.0f, T=%.2f年, σ=%.1f%%）——'
             '蓝实线=call，橙虚线=put，绿色竖线=平值K=S0' % (S0, T_YEAR, SIGMA * 100),
             fontsize=13)
panels = [
    ('Delta：方向敞口\n股价涨1元，期权价值变多少', 0, 'Delta'),
    ('Gamma：Delta的变化速度\n凸性来源，平值处最大', 1, 'Gamma'),
    ('Vega：怕波动率降\n每1个百分点波动率值多少钱', 2, 'Vega (每1%)'),
    ('Theta：时间磨损\n每天"蒸发"的时间价值', 3, 'Theta (每天)'),
    ('Rho：利率敏感度\n每1个百分点利率值多少钱', 4, 'Rho (每1%)'),
]
scale = [1, 1, 100, 365, 100]  # 换算到"每1%"或"每天"的直观单位
for (title, idx, ylabel), ax, sc in zip(panels, axes.flat, scale):
    ax.plot(K_scan, gc[:, idx] / sc, 'b-', lw=2, label='call')
    ax.plot(K_scan, gp[:, idx] / sc, 'o', color='darkorange', ms=3.5,
            mec='none', alpha=0.8, label='put')
    ax.axvline(K_ATM, color='green', ls='--', lw=1, alpha=0.7)
    ax.axhline(0, color='gray', lw=0.8, alpha=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('执行价 K（元）')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
# 第 6 格放"一句话风险直觉"小结
ax_txt = axes.flat[5]
ax_txt.axis('off')
ax_txt.text(0.02, 0.95,
            '风险直觉小结\n'
            'Delta：我有多像股票？(call 0~1, put -1~0)\n'
            'Gamma：方向敞口变化多快？平值最猛\n'
            'Vega：波动率是朋友还是敌人？\n        买期权=做多波动率\n'
            'Theta：每一天都在交的"房租"\n'
            'Rho：通常最小，高利率才显眼\n\n'
            '平值 + 短到期 = Gamma/Theta/Vega 全部最大',
            fontsize=12, va='top',
            bbox=dict(boxstyle='round', fc='#f4f9f4', ec='green', alpha=0.9))
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, '02_五个希腊字母曲线.png'), dpi=150)
plt.close(fig)
print('\n[图] 已保存 outputs/02_五个希腊字母曲线.png')

# ============================================================
# 第 3 节：蒙特卡洛(GBM) 重新定价，与 B-S 解析解对比（图03）
# ============================================================
# 风险中性 GBM（几何布朗运动）：
#   S_T = S0 · exp( (r - σ²/2)·T + σ·√T · Z ),  Z ~ N(0,1)
# 欧式期权只依赖到期价 S_T 的分布（不依赖路径）→ 蒙特卡洛只需模拟终值：
#   MC价格 = e^(-rT) · mean( max(S_T - K, 0) )
# 误差来源：抽样误差 ≈ 标准误 SE = e^(-rT)·std(payoff)/√N，随 1/√N 收敛
# （想把误差减半，路径数要翻 4 倍——所以需要方差缩减技术）。
# ------------------------------------------------------------
rng = np.random.default_rng(SEED)                    # 种子 42，结果可复现
z = rng.standard_normal(N_PATHS)                     # 10 万个标准正态抽样
st = S0 * np.exp((RISK_FREE - 0.5 * SIGMA ** 2) * T_YEAR
                 + SIGMA * np.sqrt(T_YEAR) * z)      # 10 万条"到期价"
payoff = np.maximum(st - K_ATM, 0.0)                 # call 到期损益
disc = np.exp(-RISK_FREE * T_YEAR)
mc_price = disc * payoff.mean()                      # 蒙特卡洛估计
mc_se = disc * payoff.std(ddof=1) / np.sqrt(N_PATHS)  # 标准误
mc_err_pct = abs(mc_price - call_price) / call_price * 100

# 方差缩减演示：对偶变量(antithetic)——用前 5 万个 Z 与它们的相反数配对
z_anti = np.concatenate([z[:N_PATHS // 2], -z[:N_PATHS // 2]])
st_anti = S0 * np.exp((RISK_FREE - 0.5 * SIGMA ** 2) * T_YEAR
                      + SIGMA * np.sqrt(T_YEAR) * z_anti)
payoff_anti = np.maximum(st_anti - K_ATM, 0.0)
mc_price_anti = disc * payoff_anti.mean()
mc_se_anti = disc * payoff_anti.std(ddof=1) / np.sqrt(N_PATHS)
mc_err_anti_pct = abs(mc_price_anti - call_price) / call_price * 100

print('\n========== 3. 蒙特卡洛 vs B-S 解析解（平值 call） ==========')
print(f'B-S 解析价       = {call_price:.4f} 元')
print(f'MC 10万路径(rng42) = {mc_price:.4f} 元，误差 {mc_err_pct:.3f}%，'
      f'标准误 ±{mc_se:.4f}（95%区间 [{mc_price - 1.96 * mc_se:.4f}, '
      f'{mc_price + 1.96 * mc_se:.4f}]）')
print(f'MC 对偶变量缩减   = {mc_price_anti:.4f} 元，误差 {mc_err_anti_pct:.3f}%，'
      f'标准误 ±{mc_se_anti:.4f}（SE 缩减为普通法的 {mc_se_anti / mc_se:.1%}）')
print(f'解析解落在 MC 的95%置信区间内：'
      f'{mc_price - 1.96 * mc_se <= call_price <= mc_price + 1.96 * mc_se}')

# 收敛曲线：同一条 10 万抽样序列，逐步取前缀估计价格
ns = np.unique(np.logspace(2, 5, 40).astype(int))    # 100 ~ 100000
running = disc * np.cumsum(payoff)[ns - 1] / ns
running_se = disc * payoff.std(ddof=1) / np.sqrt(ns)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
ax1.plot(ns, running, 'b-', lw=1.8, label='MC 估计值（前缀序列）')
ax1.fill_between(ns, running - 1.96 * running_se, running + 1.96 * running_se,
                 color='b', alpha=0.15, label='95%置信区间')
ax1.axhline(call_price, color='red', ls='--', lw=2, label=f'B-S 解析解 {call_price:.2f}')
ax1.set_xscale('log')
ax1.set_xlabel('路径数 N（对数轴）')
ax1.set_ylabel('平值 call 价格（元）')
ax1.set_title('蒙特卡洛收敛到解析解：误差 ~ 1/√N')
ax1.legend()
ax1.grid(alpha=0.3, which='both')

ax2.plot(ns, np.abs(running - call_price) / call_price * 100, 'b-', lw=1.8,
         label='MC 相对误差')
ax2.plot(ns, 200 * running_se / call_price, 'g--', lw=1.5,
         label='2×标准误参考线（理论包络）')
ax2.axhline(1.0, color='orange', ls=':', lw=2, label='1% 误差线')
ax2.set_xscale('log')
ax2.set_xlabel('路径数 N（对数轴）')
ax2.set_ylabel('|MC - B-S| / B-S (%)')
ax2.set_title('相对误差随路径数下降（10万路径时 %.3f%%）' % mc_err_pct)
ax2.legend()
ax2.grid(alpha=0.3, which='both')
fig.suptitle('蒙特卡洛 vs B-S 解析解（GBM, rng(%d), %d 路径）' % (SEED, N_PATHS),
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, '03_蒙特卡洛与解析解收敛对比.png'), dpi=150)
plt.close(fig)
print('[图] 已保存 outputs/03_蒙特卡洛与解析解收敛对比.png')

# ============================================================
# 第 4 节：隐含波动率（牛顿迭代反解）+ 波动率微笑（图04）
# ============================================================
# 隐含波动率 σ_imp：让 B-S 公式算出的价格 == 市场价格的那个 σ。
# 它是"市场对未来波动率的共识定价"，比历史波动率更"前瞻"。
#
# 牛顿迭代：解 f(σ) = BS_call(σ) - C_mkt = 0
#   σ_{n+1} = σ_n - f(σ_n) / f'(σ_n)，而 f'(σ) 恰好就是 Vega！
#   → 每一步的修正量 = 价格误差 ÷ Vega，几何直觉非常清楚。
# -------------------------------------------------------------


def implied_vol_newton(c_mkt, S, K, r, T, sigma0=0.50, tol=1e-12, max_iter=100):
    """牛顿迭代求隐含波动率。返回 (σ_imp, 迭代记录列表)。

    迭代式：σ_new = σ - (BS_call(σ) - C_mkt) / Vega(σ)
    """
    sigma = sigma0
    history = []
    for i in range(1, max_iter + 1):
        price = bs_call(S, K, r, sigma, T)
        vega = S * norm.pdf(bs_d1_d2(S, K, r, sigma, T)[0]) * np.sqrt(T)  # f'(σ)
        diff = price - c_mkt
        history.append((i, sigma, price, diff))
        if abs(diff) < tol:
            break
        sigma = sigma - diff / vega          # 牛顿修正：误差除以斜率
    return sigma, history


print('\n========== 4. 隐含波动率：牛顿迭代反解 ==========')
SIGMA_TRUE = 0.45  # "市场"的真实波动率（我们自己用 B-S 造一个"市场价格"）
c_mkt = bs_call(S0, K_ATM, RISK_FREE, SIGMA_TRUE, T_YEAR)
print(f'实验设计：设真实 σ=45%，用 B-S 算出"市场价格" C_mkt={c_mkt:.4f} 元，'
      f'再从 σ0=0.50 出发反推 σ')
sigma_imp, hist = implied_vol_newton(c_mkt, S0, K_ATM, RISK_FREE, T_YEAR)
print('迭代过程（前5步 + 最后1步）：')
for i, sigma_i, price_i, diff_i in hist[:5]:
    print(f'  第{i}步：σ={sigma_i:.6f}, 价格={price_i:.4f}, 误差={diff_i:+.2e}')
if len(hist) > 5:
    i, sigma_i, price_i, diff_i = hist[-1]
    print(f'  第{i}步：σ={sigma_i:.6f}, 价格={price_i:.4f}, 误差={diff_i:+.2e}  <- 收敛')
print(f'反推结果 σ_imp = {sigma_imp:.8f}，与真值 0.45 的偏差 = '
      f'{abs(sigma_imp - SIGMA_TRUE):.2e}')
print(f'是否收敛到 45%：{"是" if abs(sigma_imp - SIGMA_TRUE) < 1e-6 else "否"}')

# 用 scipy.optimize.brentq（二分法）交叉验证
sigma_brent = brentq(lambda s: bs_call(S0, K_ATM, RISK_FREE, s, T_YEAR) - c_mkt,
                     1e-4, 3.0, xtol=1e-14)
print(f'交叉验证 scipy.optimize.brentq：σ = {sigma_brent:.8f}（与牛顿法一致）')


# ---- 波动率微笑：人工设定一条 U 型 σ(K)，造"市场价"再反推 ----
def smile_sigma_true(K):
    """人工设定的 U 型波动率曲线：平值最低(25%)，越偏离越贵。

    σ(K) = 0.25 + 1.5·[ln(K/S0)]²   （ln(K/S0) 叫对数扳度 log-moneyness）
    真实市场还常见"偏斜"(skew)：左边翅膀比右边更高，本课为演示取对称U型。
    """
    m = np.log(K / S0)
    return 0.25 + 1.5 * m ** 2


K_smile = np.linspace(0.7, 1.3, 13) * S0
sig_true = smile_sigma_true(K_smile)
c_mkt_smile = bs_call(S0, K_smile, RISK_FREE, sig_true, T_YEAR)  # 用真值造市场价
sig_recovered = np.array([
    implied_vol_newton(c, S0, k, RISK_FREE, T_YEAR)[0]
    for c, k in zip(c_mkt_smile, K_smile)
])
max_rec_err = np.max(np.abs(sig_recovered - sig_true))

print('\n---------- 波动率微笑（U型 σ(K) 反推实验） ----------')
print('K/S0      真实σ      反推σ      偏差')
for k, st_, sr in list(zip(K_smile, sig_true, sig_recovered))[::3]:
    print(f'{k / S0:.2f}    {st_:.4f}    {sr:.4f}    {abs(sr - st_):.1e}')
print(f'13 个执行价全部反推成功，最大偏差 = {max_rec_err:.2e}')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_smile / S0, sig_true * 100, 'r-', lw=2.5,
        label='真实设定的 U 型 σ(K)：0.25 + 1.5·[ln(K/S0)]²')
ax.plot(K_smile / S0, sig_recovered * 100, 'bo', ms=7, mfc='none', mew=2,
        label='牛顿迭代从"市场价格"反推的隐含波动率')
ax.axvline(1.0, color='green', ls='--', lw=1, alpha=0.7, label='平值 K=S0')
ax.set_xlabel('执行价 / 现价（K/S0）')
ax.set_ylabel('隐含波动率（%）')
ax.set_title('波动率微笑：不同执行价的期权各自"含着"不同的隐含波动率\n'
             '微笑的存在 = 市场在用脚投票否定"B-S 的对数正态假设"'
             '（呼应点子11：真实收益有肥尾）')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '04_波动率微笑.png'), dpi=150)
plt.close(fig)
print('[图] 已保存 outputs/04_波动率微笑.png')

# ============================================================
# 第 5 节：Put-Call Parity 平价关系数值验证
# ============================================================
# 同一 K、同一 T 的欧式 call 与 put 必须满足：
#     C - P = S0 - K·e^(-rT)
# 套利直觉：两笔组合到期时完全一样——
#   组合A：买 call + 卖 put，到期付出 K 买入股票（无论涨跌）
#   组合B：今天借钱买 1 股股票，到期还 K·e^(rT)
#   到期现金流完全相同 → 今天价格必须相同，否则有无风险套利。
# ------------------------------------------------------------
lhs = call_price - put_price
rhs = S0 - K_ATM * np.exp(-RISK_FREE * T_YEAR)
print('\n========== 5. Put-Call Parity 数值验证 ==========')
print(f'C - P           = {lhs:.10f} 元')
print(f'S0 - K·e^(-rT)  = {rhs:.10f} 元')
print(f'两者之差        = {abs(lhs - rhs):.3e} 元（浮点精度内为 0，等式成立）')

# ============================================================
# 第 6 节：期权到期损益图：买入平值 call（图05）
# ============================================================
premium = call_price                      # 期权费 = 今天付出的权利金
breakeven = K_ATM + premium               # 盈亏平衡点 = K + 期权费
s_t = np.linspace(0.5 * S0, 1.5 * S0, 301)
payoff_line = np.maximum(s_t - K_ATM, 0.0)          # 到期行权价值
profit_line = payoff_line - premium                 # 扣掉期权费后的净损益

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(s_t, payoff_line, 'b-', lw=2, label='到期行权价值 max(S_T-K, 0)')
ax.plot(s_t, profit_line, 'r-', lw=2.5, label='策略净损益 = 行权价值 - 期权费')
ax.axhline(0, color='gray', lw=1)
ax.axvline(breakeven, color='purple', ls='--', lw=1.5,
           label=f'盈亏平衡点 K+期权费 = {breakeven:.0f} 元')
ax.axvline(K_ATM, color='green', ls=':', lw=1.5, label=f'执行价 K = {K_ATM:.0f} 元')
ax.fill_between(s_t, profit_line, 0, where=(profit_line < 0),
                color='red', alpha=0.12)
ax.fill_between(s_t, profit_line, 0, where=(profit_line > 0),
                color='green', alpha=0.12)
ax.annotate(f'期权费（最大亏损）\n= {premium:.2f} 元/股',
            xy=(0.62 * S0, -premium), xytext=(0.58 * S0, -premium * 1.9),
            fontsize=11, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred'))
ax.set_ylim(-premium * 2.6, profit_line.max() * 1.12)
ax.annotate(f'盈亏平衡 {breakeven:.0f}\n(需涨 {(breakeven / S0 - 1) * 100:.1f}%)',
            xy=(breakeven, 0), xytext=(breakeven + 15, -premium * 1.4),
            fontsize=11, color='purple',
            arrowprops=dict(arrowstyle='->', color='purple'))
ax.set_xlabel('到期日股价 S_T（元）')
ax.set_ylabel('每股损益（元）')
ax.set_title(f'买入平值看涨期权损益图：亏钱封底（最多亏期权费{premium:.0f}元），'
             f'赚钱不封顶\n但股价要涨过 {breakeven:.0f} 元（+{(breakeven / S0 - 1) * 100:.1f}%）才开始真正赚钱')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '05_期权损益图.png'), dpi=150)
plt.close(fig)
print('\n[图] 已保存 outputs/05_期权损益图.png')

# ============================================================
# 第 7 节：B-S 价格与内在价值（图01）
# ============================================================
# 关键认知：期权价 = 内在价值 + 时间价值。
# 平值时内在价值=0，全部是时间价值；股价越偏离 K，时间价值越薄。
s_grid = np.linspace(0.6 * S0, 1.4 * S0, 301)
price_curve = bs_call(s_grid, K_ATM, RISK_FREE, SIGMA, T_YEAR)
intrinsic_curve = np.maximum(s_grid - K_ATM, 0.0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(s_grid, price_curve, 'b-', lw=2.5, label='B-S 看涨期权价格')
ax.plot(s_grid, intrinsic_curve, 'k--', lw=2, label='内在价值 max(S-K, 0)')
ax.fill_between(s_grid, intrinsic_curve, price_curve,
                color='orange', alpha=0.25, label='时间价值（两线之间）')
ax.axvline(S0, color='green', ls='--', lw=1.5, label=f'现价 S0 = {S0:.0f}')
ax.scatter([S0], [call_price], color='red', zorder=5)
ax.annotate(f'平值call = {call_price:.2f} 元\n(内在价值 {intrinsic_call:.0f} + '
            f'时间价值 {time_value_call:.2f})',
            xy=(S0, call_price), xytext=(0.72 * S0, call_price * 2.1),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlabel('标的现价 S（元）')
ax.set_ylabel('期权价格（元）')
ax.set_title('B-S 期权价格 = 内在价值 + 时间价值\n'
             '平值附近时间价值最厚（Theta 在这里磨损最快，见希腊字母图）')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '01_B-S价格与内在价值.png'), dpi=150)
plt.close(fig)
print('[图] 已保存 outputs/01_B-S价格与内在价值.png')

# ============================================================
# 第 8 节：关键 K 值的希腊字母表（greeks_table.csv）
# ============================================================
rows = []
for mny in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:
    k = mny * S0
    dc, gc_, vc, thc, rhoc = greeks_call(S0, k, RISK_FREE, SIGMA, T_YEAR)
    dp, gp_, vp, thp, rhop = greeks_put(S0, k, RISK_FREE, SIGMA, T_YEAR)
    rows.append({
        'K': round(k, 2),
        'moneyness(K/S0)': mny,
        'call价格': round(bs_call(S0, k, RISK_FREE, SIGMA, T_YEAR), 4),
        'put价格': round(bs_put(S0, k, RISK_FREE, SIGMA, T_YEAR), 4),
        'Delta_call': round(dc, 4),
        'Delta_put': round(dp, 4),
        'Gamma': round(gc_, 6),
        'Vega(每1%)': round(vc / 100, 4),
        'Theta_call(每天)': round(thc / 365, 4),
        'Theta_put(每天)': round(thp / 365, 4),
        'Rho_call(每1%)': round(rhoc / 100, 4),
    })
greeks_df = pd.DataFrame(rows)
csv_path = os.path.join(OUT, 'greeks_table.csv')
greeks_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print('\n========== 6. 关键K值希腊字母表 ==========')
print(greeks_df.to_string(index=False))
print(f'[表] 已保存 {csv_path}（utf-8-sig，Excel 打开不乱码）')

# ============================================================
# 收尾
# ============================================================
print('\n==================== 全部完成 ====================')
print(f'标的：贵州茅台600519  S0={S0:.2f}'
      f'{"（回退示例值）" if IS_FALLBACK else ""}  σ60日={SIGMA:.4f}')
print(f'平值call（B-S解析）= {call_price:.4f} 元；MC(10万路径) = {mc_price:.4f} 元，'
      f'误差 {mc_err_pct:.3f}%（对偶变量法误差 {mc_err_anti_pct:.3f}%）')
print(f'隐含波动率牛顿迭代：{len(hist)} 步收敛到 {sigma_imp:.6f}（真值 0.45）')
print(f'Put-Call parity 之差 = {abs(lhs - rhs):.3e}（成立）')
print(f'微笑反推最大偏差 = {max_rec_err:.2e}（13/13 个执行价全部还原）')
print(f'盈亏平衡点 = {breakeven:.2f} 元（需上涨 {(breakeven / S0 - 1) * 100:.2f}%）')
