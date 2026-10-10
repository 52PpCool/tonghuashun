/**
 * probe2_mktcap.js —— 验证已上市股票的 2016-09-01 历史总市值是否返回真值
 * 长川科技(2017-04-17上市)返回0.0符合预期；此处用中际旭创(2012-04-10上市)验证。
 */
const path = require('path');
const fs = require('fs');
const SKILL = '/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js';
const { call } = require(SKILL);
const RAW = path.join(__dirname, 'raw');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const params = { query: '中际旭创2016年9月1日的总市值' };
  const result = await call('stock', 'get_stock_performance', params);
  fs.writeFileSync(path.join(RAW, 'probe4_mktcap_listed.json'),
    JSON.stringify({ tool: 'get_stock_performance', params, response: result }, null, 2), 'utf-8');
  console.log(JSON.stringify(result, null, 2).slice(0, 1200));
  await sleep(1100);
  // 顺带试探两只股票一次 query 的总市值（正式批量要用）
  const p2 = { query: '中际旭创、新易盛2016年9月1日的总市值' };
  const r2 = await call('stock', 'get_stock_performance', p2);
  fs.writeFileSync(path.join(RAW, 'probe5_mktcap_2stocks.json'),
    JSON.stringify({ tool: 'get_stock_performance', params: p2, response: r2 }, null, 2), 'utf-8');
  console.log(JSON.stringify(r2, null, 2).slice(0, 1200));
}
main().catch((e) => { console.error('[fatal]', e && e.message ? e.message : e); process.exit(0); });
