/**
 * fetch_listday_cap.js —— 十倍股画像 补充取数：75 只窗口内上市股按各自上市日查总市值
 *
 * 背景：02_mktcap 批次以 2016-09-01 锚定，未上市返回 0.0（已解析为 NaN）。
 *       为让"起点市值"覆盖全部 88 只，对窗口内上市的 75 只，以各自上市日为锚点逐只查询。
 *       问法已由 probe3_listday_cap.js 验证（德明利2022年7月1日→30.576亿 等3例）。
 *
 * 用法: node fetch_listday_cap.js [--fresh]   # 幂等，逐只落盘 raw/03_cap_at_listday/
 */
const path = require('path');
const fs = require('fs');
const SKILL = '/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js';
const { call } = require(SKILL);

const DIR = __dirname;
const RAW = path.join(DIR, 'raw', '03_cap_at_listday');
const PROFILE = path.join(DIR, 'data', 'profile_basic.csv');
if (!fs.existsSync(RAW)) fs.mkdirSync(RAW, { recursive: true });
const FRESH = process.argv.includes('--fresh');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function cnDate(yyyymmdd) {
  const y = yyyymmdd.slice(0, 4), m = parseInt(yyyymmdd.slice(4, 6), 10), d = parseInt(yyyymmdd.slice(6, 8), 10);
  return `${y}年${m}月${d}日`;
}

async function main() {
  const lines = fs.readFileSync(PROFILE, 'utf-8').trim().split('\n');
  const header = lines[0].split(',');
  const idx = { code: header.indexOf('code'), name: header.indexOf('name'),
    listDate: header.indexOf('list_date'), before: header.indexOf('listed_before_window') };
  const stocks = lines.slice(1).map((l) => {
    const c = l.split(',');
    return { code: c[idx.code], name: c[idx.name], listDate: c[idx.listDate], before: c[idx.before] === 'True' };
  }).filter((s) => !s.before); // 只查窗口内上市的
  console.log(`[info] 窗口内上市需补查市值 ${stocks.length} 只`);

  let calls = 0, ok = 0;
  for (const s of stocks) {
    const file = path.join(RAW, `${s.code.replace('.', '_')}.json`);
    if (!FRESH && fs.existsSync(file)) { ok++; continue; }
    const date = cnDate(s.listDate);
    const params = { query: `${s.name}${date}的总市值` };
    try {
      const result = await call('stock', 'get_stock_performance', params);
      fs.writeFileSync(file, JSON.stringify({ tool: 'get_stock_performance', params, response: result }, null, 2), 'utf-8');
      calls++;
      const txt = JSON.stringify(result);
      const m = txt.match(/\|(\d{6}\.(SZ|SH))\|[^|]*\|([^|]+)\|/);
      console.log(`[call] ${s.code} ${s.name} ${date} → ${m ? m[3] : '(未匹配)'}`);
      if (result && result.ok) ok++;
    } catch (e) {
      fs.writeFileSync(file, JSON.stringify({ tool: 'get_stock_performance', params, error: String(e && e.message || e) }, null, 2), 'utf-8');
      console.log(`[FAIL] ${s.code} ${s.name}: ${e && e.message}`);
    }
    await sleep(1100); // 免费key限流 <=2 req/s
  }
  console.log(`\n[done] 本次新调用 ${calls} 次，成功 ${ok}/${stocks.length}。文件在 raw/03_cap_at_listday/`);
}
main().catch((e) => { console.error('[fatal]', e && e.message ? e.message : e); process.exit(0); });
