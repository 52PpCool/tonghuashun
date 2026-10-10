/**
 * fetch_ifind.js —— 第29课「PEAD财报漂移」第一步：取数
 *
 * 目标：为 data/cache 里的 30 只A股查询财报的「归母净利润」和「公告日期」。
 * 为了能自己算同比增速 yoy，每个事件期都要带上上一年同期的利润作分母：
 *   事件期：2024年报、2025半年报、2025三季报（公告日做事件日）
 *   分母期：2023年报、2024半年报、2024三季报（只取利润，不取公告日）
 * 所以单次 query 一次要 6 个报告期 × 5 只股票。
 *
 * 措辞踩坑记录（真实调试留痕）：
 *   v1 问法 "...的归母净利润和公告日期" —— "公告日期"被解析器吞掉，
 *       只返回报告期"日期"列 + 利润（见 raw/01_financials_batch1_v1.json）。
 *   v2 问法 "...的归母净利润和财报公告日期" —— 按 cn_stock.md 文档规范
 *       "股票简称+财务指标+财报日期" 的措辞重试。
 *
 * 用法:
 *   node fetch_ifind.js                    # 全量（已存在的批次文件自动复用）
 *   node fetch_ifind.js --from 2 --to 2    # 只跑第2批（试探新问法用）
 *   node fetch_ifind.js --suffix v2        # 批次文件名加后缀（不覆盖旧留痕）
 *   node fetch_ifind.js --fresh            # 忽略缓存全量重取
 *
 * 硬性约束:
 *   - 免费 key 限流 2 请求/秒：每次调用之间 sleep 1.1 秒
 *   - 单次 query 主体 ≤ 5 只（iFinD 自然语言解析对长 query 不稳）
 *   - 总调用 ≤ 8 次（v1试探1次 + v2正式6次 = 7次）
 *   - 每个原始响应原样落盘 raw/（留痕，审计可回溯）
 */
const path = require('path');
const fs = require('fs');

const SKILL = '/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js';
const { call } = require(SKILL);

const DIR = __dirname;
const RAW = path.join(DIR, 'raw');
const CACHE_DIR = '/home/ppku/zcodespace/tonghuashun/research/20260918_量化学习全景/data/cache';
if (!fs.existsSync(RAW)) fs.mkdirSync(RAW, { recursive: true });

const argOf = (name, dft) => process.argv.includes(name)
  ? process.argv[process.argv.indexOf(name) + 1] : dft;
const FROM = parseInt(argOf('--from', '1'), 10);
const TO = parseInt(argOf('--to', '6'), 10);
const SUFFIX = argOf('--suffix', '');
const FRESH = process.argv.includes('--fresh');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const BATCH_SIZE = 5;

/** 从 data/cache/stock_*.csv 文件名解析 30 只股票清单（代码 + 简称） */
function loadStocks() {
  const stocks = fs.readdirSync(CACHE_DIR)
    .filter((f) => f.startsWith('stock_') && f.endsWith('.csv'))
    .map((f) => {
      const stem = f.replace('.csv', '');
      const parts = stem.split('_');
      return { code: parts[1], name: parts.slice(2).join('_') };
    })
    .sort((a, b) => a.code.localeCompare(b.code));
  fs.writeFileSync(path.join(RAW, '00_stock_list.json'),
    JSON.stringify({ source: 'data/cache/stock_*.csv 文件名', count: stocks.length, stocks }, null, 2), 'utf-8');
  return stocks;
}

/** 调一次接口并存盘；文件已存在则复用（幂等：迭代调试不重复消耗调用次数）。
 * 落盘内容 = {tool, params, response} 三件套，query 原文一并留痕。 */
async function fetchOnce(file, serverType, toolName, params, label) {
  const fpath = path.join(RAW, file);
  if (!FRESH && fs.existsSync(fpath)) {
    console.log(`[skip] ${file} 已存在，复用缓存（不消耗接口调用）`);
    return JSON.parse(fs.readFileSync(fpath, 'utf-8'));
  }
  console.log(`[call] ${label} | ${toolName}\n       query=${params.query}`);
  const result = await call(serverType, toolName, params);
  fs.writeFileSync(fpath, JSON.stringify({ tool: toolName, params, response: result }, null, 2), 'utf-8');
  console.log(`[save] ${file} (${fs.statSync(fpath).size} bytes) ok=${result.ok}`);
  await sleep(1100); // 免费key限流: <=2 req/s，间隔至少1.1秒
  return result;
}

async function main() {
  const stocks = loadStocks();
  console.log(`[info] 股票清单 ${stocks.length} 只（来自 data/cache 文件名）`);

  // 分批：每批 ≤5 只主体
  const batches = [];
  for (let i = 0; i < stocks.length; i += BATCH_SIZE) batches.push(stocks.slice(i, i + BATCH_SIZE));

  let okCount = 0, callCount = 0;
  for (let i = FROM - 1; i < Math.min(batches.length, TO); i++) {
    const names = batches[i].map((s) => s.name).join('、');
    // v4 问法（经 4 次试探定型，试探留痕见 raw/01_financials_batch*_v*.json）：
    //  - "定期报告实际披露日期" 是引擎认识的指标名，但它每股只回 1 个标量，
    //    实测映射到"第一个提到的报告期"→ 把 2024年报 放第一位，拿到 2024年报真实公告日
    //    （akshare 披露时间表的交叉核验源，全量3期日期由 fetch_dates_akshare.py 补齐）
    //  - 利润部分会按报告期展开成每期一行（含去年同期分母）
    const query = `${names}的2024年报、2025半年报、2025三季报的定期报告实际披露日期和归母净利润`;
    const file = `01_financials_batch${i + 1}${SUFFIX ? '_' + SUFFIX : ''}.json`;
    const before = fs.existsSync(path.join(RAW, file)) && !FRESH;
    const res = await fetchOnce(
      file, 'stock', 'get_stock_financials', { query },
      `财报批次${i + 1}/${batches.length}（${names}）`
    );
    if (!before) callCount++;
    if (res && res.response && res.response.ok) okCount++;
  }
  console.log(`\n[done] 本次实际新调用接口 ${callCount} 次；成功批次 ${okCount}。raw/ 文件清单:`);
  fs.readdirSync(RAW).sort().forEach((f) => console.log('   ', f));
}

main().catch((e) => {
  console.error('[fatal] 取数失败:', e && e.message ? e.message : e);
  console.error('[fallback] parse_and_study.py 会检测 raw/ 缺失并走降级方案（合成公告日期，教学演示）');
  process.exit(0); // 教学规范：脚本永不非0退出，失败信息留痕
});
