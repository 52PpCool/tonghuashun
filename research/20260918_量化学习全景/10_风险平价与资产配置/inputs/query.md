# 入参存档：本节课的任务要求（query 原文要点）

- 任务：为量化初学者构建一节"必须能跑通"的课程，只负责一个点子：
  创建文件夹「10_风险平价与资产配置」——"不要把鸡蛋放在一个篮子里"的工程化：
  等权/最小方差/风险平价/波动率目标四种配置的真实回测对比。
- 工作区：/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/
- Python（绝对路径）：/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/.venv/bin/python
- 已装库：numpy pandas matplotlib scipy scikit-learn statsmodels arch lightgbm cvxpy
- matplotlib 必须 use('Agg')，中文字体 Noto Sans CJK SC / WenQuanYi Micro Hei，axes.unicode_minus=False
- 数据：工作区 data/cache/ 下 10 个行业 board_*.csv（date,open,high,low,close,volume，起始日期不同）
  + index_000300_沪深300.csv；utf-8-sig；读取方式 pd.read_csv(p, parse_dates=['date']).set_index('date')。
  取共同重叠期、月频收益。若数据缺失用 np.random.default_rng(42) 造合成数据并标注"合成数据"；
  脚本绝不允许非 0 退出。

## main.py 要求
1. 风险贡献概念：RC_i = w_i·(Σw)_i/√(w'Σw)，注释+一张图讲直觉（等权≠等风险贡献）。
2. 四种月度再平衡配置策略完整回测（2018起，用共同重叠期）：
   等权；最小方差（滚动24个月协方差）；风险平价（迭代/scipy.optimize 最小化 Σ(RC_i-1/N)²，讲清求解目标）；
   波动率目标（目标年化10%，权重=等权×(10%/已实现波动)，讲杠杆与现金腿）。
3. 对比基准：沪深300买入持有。
4. 绩效表：年化、波动、夏普(rf=2%)、最大回撤、月度胜率、换手 → outputs/portfolios.csv（utf-8-sig）。
5. 风险贡献可视化：等权 vs 风险平价在某时点的风险贡献堆叠/分组条形图。
6. 净值曲线+回撤曲线（多策略一图）。

## 产出
- main.py、outputs/（01~05 五张图 + portfolios.csv）
- README.md（中文教学文档：这节课学什么 / 核心概念 / 代码怎么读（含行号）/ 如何运行 /
  输出怎么读（引用实际运行真实数字，如实呈现跑输年份）/ 常见的坑（≥3个，用实际相关系数说话）/
  延伸思考（2-3个练习））
- 硬性要求：实际运行到退出码 0；README 数字全部来自实际运行；禁止修改 data/cache 与 .venv。

## 实际执行中的数据事实记录（与原要求"2018起"的偏差说明）
- 10 个行业共同重叠期自 2021-12-29 起（电力行业指数发布最晚），叠加 24 个月协方差预热，
  四策略统一评估窗口为 2024-01-31 ~ 2026-08-31（32 个月）。已在 README 中如实说明。
- 原始板块数据存在 29 处"基期跳变"毛刺（单日 ±35%~66% 后部分跳回），已用拼接法清洗并在 README 记录。
