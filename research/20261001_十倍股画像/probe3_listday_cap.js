/**
 * probe3_listday_cap.js —— 验证"次新股按上市日查总市值"是否返回真值
 * 德明利(20220701)、凯莱英(20161118)、英维克(20161229) 三只不同年份的上市日。
 */
const path = require('path');
const fs = require('fs');
const SKILL = '/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js';
const { call } = require(SKILL);
const RAW = path.join(__dirname, 'raw');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const cases = [
    ['probe6_cap_listday_1.json', '德明利2022年7月1日的总市值'],
    ['probe6_cap_listday_2.json', '凯莱英2016年11月18日的总市值'],
    ['probe6_cap_listday_3.json', '英维克2016年12月29日的总市值'],
  ];
  for (const [file, query] of cases) {
    const fpath = path.join(RAW, file);
    if (fs.existsSync(fpath)) { console.log('[skip]', file); continue; }
    const params = { query };
    const result = await call('stock', 'get_stock_performance', params);
    fs.writeFileSync(path.join(RAW, file), JSON.stringify({ tool: 'get_stock_performance', params, response: result }, null, 2), 'utf-8');
    const txt = JSON.stringify(result);
    const m = txt.match(/\|(\d{6}\.(SZ|SH))\|[^|]*\|([^|]+)\|/);
    console.log('[probe]', query, '→', m ? m[3] : '(未匹配到表行)');
    await sleep(1100);
  }
}
main().catch((e) => { console.error('[fatal]', e && e.message ? e.message : e); process.exit(0); });
