/**
 * fetch_profile.js —— 十倍股画像 第一步：iFinD 拉取 88 只十倍股的基础画像
 *
 * 数据源：research/20260905_最近10年A股十倍股/数据/results_verified.csv（isTenX=1 的 88 只）
 * 指标：① 上市日期 ② 所属同花顺行业（一级） ③ 2016-09-01 总市值（历史锚定）
 *
 * 问法试探结论（probe_ifind.js / probe2_mktcap.js，留痕见 raw/probe*.json）：
 *   - get_stock_info："A、B、C的上市日期和所属同花顺行业" → markdown 表，上市日期格式 YYYYMMDD
 *   - get_stock_performance："A、B的2016年9月1日的总市值" → 单位"亿"，未上市返回 0.0（自然语义：未上市）
 *   - 单次 query 主体 ≤ 5 只（skill 规范），免费 key 限流 2 请求/秒 → 批间 sleep 1.1s
 *
 * 用法:
 *   node fetch_profile.js [--fresh]   # 幂等：raw/ 已存在的批次文件直接复用，不重复消耗调用
 */
const path = require('path');
const fs = require('fs');
const SKILL = '/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js';
const { call } = require(SKILL);

const DIR = __dirname;
const RAW = path.join(DIR, 'raw');
const TENX_CSV = '/home/ppku/zcodespace/tonghuashun/research/20260905_最近10年A股十倍股/数据/results_verified.csv';
if (!fs.existsSync(RAW)) fs.mkdirSync(RAW, { recursive: true });

const FRESH = process.argv.includes('--fresh');
const BATCH_SIZE = 5;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function loadTenX() {
  const lines = fs.readFileSync(TENX_CSV, 'utf-8').trim().split('\n');
  const header = lines[0].split(',');
  const iCode = header.indexOf('股票代码'), iName = header.indexOf('股票简称'), iTen = header.indexOf('isTenX(≥1000%为1)');
  const stocks = [];
  for (const line of lines.slice(1)) {
    const c = line.split(',');
    if (c[iTen] === '1') stocks.push({ code: c[iCode], name: c[iName] });
  }
  fs.writeFileSync(path.join(RAW, '00_tenx_list.json'),
    JSON.stringify({ source: TENX_CSV, count: stocks.length, stocks }, null, 2), 'utf-8');
  return stocks;
}

async function fetchBatched(prefix, toolName, makeQuery, batches) {
  let calls = 0;
  for (let i = 0; i < batches.length; i++) {
    const names = batches[i].map((s) => s.name).join('、');
    const file = `${prefix}_batch${i + 1}.json`;
    const fpath = path.join(RAW, file);
    if (!FRESH && fs.existsSync(fpath)) {
      console.log(`[skip] ${file} 已存在，复用`);
      continue;
    }
    const params = { query: makeQuery(names) };
    console.log(`[call] ${file} | ${toolName}\n       query=${params.query}`);
    const result = await call('stock', toolName, params);
    fs.writeFileSync(fpath, JSON.stringify({ tool: toolName, params, response: result }, null, 2), 'utf-8');
    calls++;
    await sleep(1100); // 免费key限流 <=2 req/s
  }
  return calls;
}

async function main() {
  const stocks = loadTenX();
  console.log(`[info] 十倍股 ${stocks.length} 只`);
  const batches = [];
  for (let i = 0; i < stocks.length; i += BATCH_SIZE) batches.push(stocks.slice(i, i + BATCH_SIZE));

  const c1 = await fetchBatched('01_info', 'get_stock_info',
    (names) => `${names}的上市日期和所属同花顺行业`, batches);
  const c2 = await fetchBatched('02_mktcap', 'get_stock_performance',
    (names) => `${names}2016年9月1日的总市值`, batches);

  console.log(`\n[done] 本次新调用接口 ${c1 + c2} 次（info ${c1} + mktcap ${c2}）。raw/ 文件:`);
  fs.readdirSync(RAW).sort().forEach((f) => console.log('   ', f));
}
main().catch((e) => {
  console.error('[fatal]', e && e.message ? e.message : e);
  process.exit(0); // 教学规范：失败信息留痕，不非0退出
});
