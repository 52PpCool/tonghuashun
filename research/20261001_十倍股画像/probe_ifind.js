/**
 * probe_ifind.js —— 十倍股画像：iFinD 问法试探（正式取数前的探针）
 * 试探三件事：上市日期 / 所属同花顺行业 / 2016-09-01 历史总市值。
 * 原始响应原样落盘 raw/（留痕，含失败试探）。
 */
const path = require('path');
const fs = require('fs');
const SKILL = '/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js';
const { call } = require(SKILL);

const DIR = __dirname;
const RAW = path.join(DIR, 'raw');
if (!fs.existsSync(RAW)) fs.mkdirSync(RAW, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function probe(file, toolName, params, label) {
  const fpath = path.join(RAW, file);
  console.log(`[probe] ${label} | ${toolName}\n        query=${params.query}`);
  const result = await call('stock', toolName, params);
  fs.writeFileSync(fpath, JSON.stringify({ tool: toolName, params, response: result }, null, 2), 'utf-8');
  console.log(`[save] ${file} (${fs.statSync(fpath).size} bytes) ok=${result && result.ok}`);
  await sleep(1100);
}

async function main() {
  // 试探1：get_stock_info 上 2 只股票的上市日期+行业
  await probe('probe1_info_2stocks.json', 'get_stock_info', {
    query: '长川科技、中际旭创的上市日期和所属同花顺行业',
  }, '上市日期+行业');
  // 试探2：get_stock_shareholders 历史总市值（股本结构类）
  await probe('probe2_mktcap_shareholders.json', 'get_stock_shareholders', {
    query: '长川科技2016年9月1日的总市值和总股本',
  }, '历史总市值-股本工具');
  // 试探3：get_stock_performance 历史总市值（行情类）
  await probe('probe3_mktcap_performance.json', 'get_stock_performance', {
    query: '长川科技2016年9月1日的总市值',
  }, '历史总市值-行情工具');
  console.log('[done] 三次试探完成，请查看 raw/ 下返回结构');
}

main().catch((e) => {
  console.error('[fatal]', e && e.message ? e.message : e);
  process.exit(0);
});
