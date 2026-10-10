/**
 * fetch_control_profile.js —— 伪十倍股对照画像 第一步：iFinD 拉 28 只对照组基础画像
 *
 * 对照组定义：research/20260905_最近10年A股十倍股/数据/results_verified.csv 中
 *   核验区间涨幅%（前复权）∈ [500, 1000) 且 isTenX=0 的 28 只
 *   （同一次智能选股初筛、同一引擎逐只核验口径，唯一差异是涨幅档位——理想的对照组）。
 *
 * 管道与 20261001_十倍股画像/fetch_profile.js 完全一致（问法已经上一轮验证）：
 *   - get_stock_info："A、B、C的上市日期和所属同花顺行业"（5只/批 → 6批）
 *   - get_stock_performance："A、B、C2016年9月1日的总市值"（老股锚点，5只/批 → 6批）
 *   - 上市日市值对窗口内上市股补查 → fetch_control_listday_cap.js
 *
 * 用法: node fetch_control_profile.js [--fresh]   # 幂等，原始响应落盘 raw/
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

function loadControl() {
  const lines = fs.readFileSync(TENX_CSV, 'utf-8').trim().split('\n');
  const header = lines[0].split(',');
  const iCode = header.indexOf('股票代码'), iName = header.indexOf('股票简称'),
    iGain = header.indexOf('核验区间涨幅%(前复权)'), iTen = header.indexOf('isTenX(≥1000%为1)');
  const stocks = [];
  for (const line of lines.slice(1)) {
    const c = line.split(',');
    if (c[iTen] === '0') {
      const gain = parseFloat(c[iGain]);
      if (gain >= 500 && gain < 1000) stocks.push({ code: c[iCode], name: c[iName], gain });
    }
  }
  fs.writeFileSync(path.join(RAW, '00_control_list.json'),
    JSON.stringify({ source: TENX_CSV, rule: '500<=gain<1000 & isTenX=0', count: stocks.length, stocks }, null, 2), 'utf-8');
  return stocks;
}

async function fetchBatched(prefix, toolName, makeQuery, batches) {
  let calls = 0;
  for (let i = 0; i < batches.length; i++) {
    const names = batches[i].map((s) => s.name).join('、');
    const file = `${prefix}_batch${i + 1}.json`;
    const fpath = path.join(RAW, file);
    if (!FRESH && fs.existsSync(fpath)) { console.log(`[skip] ${file}`); continue; }
    const params = { query: makeQuery(names) };
    console.log(`[call] ${file} | ${toolName}\n       query=${params.query}`);
    const result = await call('stock', toolName, params);
    fs.writeFileSync(fpath, JSON.stringify({ tool: toolName, params, response: result }, null, 2), 'utf-8');
    calls++;
    await sleep(1100);
  }
  return calls;
}

async function main() {
  const stocks = loadControl();
  console.log(`[info] 对照组 ${stocks.length} 只`);
  const batches = [];
  for (let i = 0; i < stocks.length; i += BATCH_SIZE) batches.push(stocks.slice(i, i + BATCH_SIZE));
  const c1 = await fetchBatched('01_info', 'get_stock_info',
    (names) => `${names}的上市日期和所属同花顺行业`, batches);
  const c2 = await fetchBatched('02_mktcap', 'get_stock_performance',
    (names) => `${names}2016年9月1日的总市值`, batches);
  console.log(`\n[done] 本次新调用 ${c1 + c2} 次（info ${c1} + mktcap ${c2}）`);
}
main().catch((e) => { console.error('[fatal]', e && e.message ? e.message : e); process.exit(0); });
