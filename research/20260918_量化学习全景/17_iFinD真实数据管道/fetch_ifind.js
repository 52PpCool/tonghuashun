/**
 * fetch_ifind.js —— 第17课「iFinD真实数据管道」第一步：取数
 *
 * 机构真实工作流微缩版：智能选股圈定候选池 -> 下载全量CSV -> 分批查财务 -> 分批查行情
 *
 * 用法:
 *   node fetch_ifind.js            # 完整取数（已存在的 raw/*.json 会复用，不重复消耗接口调用）
 *   node fetch_ifind.js --fresh    # 忽略缓存，全量重新取数
 *
 * 硬性约束:
 *   - 免费 key 限流 2 请求/秒，每次调用之间 sleep 1.1 秒
 *   - 单次 query 主体不超过 5 个（iFinD 自然语言解析对长 query 不稳）
 *   - 每个原始响应原样落盘 raw/ 目录（入参出参留痕，审计可回溯）
 *   - 全流程接口调用 ≤ 10 次：1(选股) + 4(财务) + 4(行情) = 9 次
 */
const path = require('path');
const fs = require('fs');
const http = require('http');

const SKILL = '/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js';
const { call } = require(SKILL);

const DIR = __dirname;
const RAW = path.join(DIR, 'raw');
if (!fs.existsSync(RAW)) fs.mkdirSync(RAW, { recursive: true });

const FRESH = process.argv.includes('--fresh');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DETAIL_TOP_N = 20; // 候选池按市值取前 N 只做深挖（控制在 10 次调用预算内：1+4+4=9）

/** 调一次接口并存盘；文件已存在则复用（幂等，方便迭代调试不重复消耗调用次数） */
async function fetchOnce(file, serverType, toolName, params, label) {
  const fpath = path.join(RAW, file);
  if (!FRESH && fs.existsSync(fpath)) {
    console.log(`[skip] ${file} 已存在，复用缓存（不消耗接口调用）`);
    return JSON.parse(fs.readFileSync(fpath, 'utf-8'));
  }
  console.log(`[call] ${label} | ${toolName} | query=${JSON.stringify(params.query)}`);
  const result = await call(serverType, toolName, params);
  fs.writeFileSync(fpath, JSON.stringify(result, null, 2), 'utf-8');
  console.log(`[save] ${file} (${fs.statSync(fpath).size} bytes) ok=${result.ok}`);
  await sleep(1100); // 免费key限流: <=2 req/s，间隔至少1秒
  return result;
}

/**
 * 拆开 iFinD 的"俄罗斯套娃"结构，取出 markdown 正文：
 * MCP层: result.content[0].text -> 一段JSON字符串
 * 业务层: {"code":1,"msg":"success","data":"再一层JSON字符串"}
 * 最内层: {"answer":"# 选股结果...\n| 股票代码 | ...", "dataTotalVolume":1048, ...}
 */
function unwrapToMarkdown(rawResponse) {
  if (!rawResponse || !rawResponse.ok) return null;
  let text = rawResponse?.data?.result?.content?.[0]?.text || null;
  if (!text) return null;
  for (let i = 0; i < 2; i++) {
    try {
      const obj = JSON.parse(text);
      if (obj && typeof obj.data === 'string') { text = obj.data; continue; }
      if (obj && typeof obj.answer === 'string') { text = obj.answer; break; }
      break;
    } catch { break; }
  }
  return text;
}

/** 全量CSV下载（选股结果>100行被截断时，iFinD给24小时有效的csv下载链接） */
function downloadCsv(url, fpath) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      if (res.statusCode !== 200) { resolve(false); res.resume(); return; }
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        fs.writeFileSync(fpath, Buffer.concat(chunks));
        resolve(true);
      });
    });
    req.on('error', () => resolve(false));
    req.setTimeout(30000, () => { req.destroy(); resolve(false); });
  });
}

async function main() {
  // ---------- 第 1 步：智能选股，圈定候选池 ----------
  const searchFile = '01_search.json';
  if (FRESH && fs.existsSync(path.join(RAW, searchFile))) fs.unlinkSync(path.join(RAW, searchFile));
  const searchRaw = await fetchOnce(
    searchFile, 'stock', 'search_stocks',
    { query: '沪深300成分股中市值大于500亿的股票' },
    '智能选股'
  );
  const answer = unwrapToMarkdown(searchRaw);
  if (!answer) {
    console.error('[fatal] search_stocks 未返回有效文本，原始响应已存 raw/01_search.json，请人工检查');
    console.error('降级方案：parse_and_screen.py 会检测 raw/ 缺失并改用 data/cache 完成同口径流程');
    return;
  }
  fs.writeFileSync(path.join(RAW, '01_search_text.txt'), answer, 'utf-8');

  // 记录业务层的元信息（返回截断信息、总行数）
  try {
    const outer = JSON.parse(searchRaw.data.result.content[0].text);
    const inner = JSON.parse(outer.data);
    fs.writeFileSync(path.join(RAW, '01_search_meta.json'),
      JSON.stringify({ dataTotalVolume: inner.dataTotalVolume, selectedSecuritiesCount: inner.selectedSecuritiesCount }, null, 2), 'utf-8');
    console.log(`[info] 业务元信息: 命中${inner.selectedSecuritiesCount}只, 数据单元${inner.dataTotalVolume}个`);
  } catch { /* 元信息解析失败不影响主流程 */ }

  // 选股结果被截断时，下载全量 CSV（链接24小时有效，下载不算接口调用、不受限流）
  const mUrl = answer.match(/https?:\/\/o\.thsi\.cn[^\s]+/);
  const fullCsv = path.join(RAW, '01_search_full.csv');
  if (mUrl && !fs.existsSync(fullCsv)) {
    const okDl = await downloadCsv(mUrl[0], fullCsv);
    console.log(`[info] 全量CSV下载: ${okDl ? '成功' : '失败(降级用表格内的前100行)'} -> ${path.basename(fullCsv)}`);
  }

  // ---------- 第 2 步：确定深挖名单 ----------
  // 优先用全量CSV；否则用表格内前100行。列: 股票代码|股票简称|股票市场类型|总市值[日期]
  let csvLines = [];
  if (fs.existsSync(fullCsv)) {
    csvLines = fs.readFileSync(fullCsv, 'utf-8').trim().split('\n');
  } else {
    csvLines = answer.split('\n').filter((l) => l.trim().startsWith('|'));
  }
  const parseCap = (s) => parseFloat(String(s).replace(/[^\d.eE]/g, '')); // "2.27E11"(元) -> 数值
  const stocks = [];
  for (const line of csvLines.slice(1)) { // 跳过表头
    const cells = line.split(/[|,]/).map((c) => c.trim()).filter((c) => c !== '');
    if (cells.length >= 4 && /\d{6}\.(SZ|SH)/.test(cells[0])) {
      stocks.push({ code: cells[0], name: cells[1], mktcap: parseCap(cells[3]) });
    }
  }
  console.log(`[info] 解析出候选池 ${stocks.length} 只`);
  const pool = stocks
    .slice()
    .sort((a, b) => (isNaN(b.mktcap) ? -1 : b.mktcap) - (isNaN(a.mktcap) ? -1 : a.mktcap))
    .slice(0, DETAIL_TOP_N);
  console.log('[info] 深挖名单(按总市值降序前20):');
  pool.forEach((s, i) => console.log(`  ${String(i + 1).padStart(2)}. ${s.name}(${s.code}) ${(s.mktcap / 1e8).toFixed(0)}亿元`));
  fs.writeFileSync(path.join(RAW, 'pool.json'), JSON.stringify(pool, null, 2), 'utf-8');
  fs.writeFileSync(path.join(RAW, 'pool.txt'), pool.map((s) => s.name).join('\n'), 'utf-8');

  // ---------- 第 3 步：分批查财务（每批最多 5 个主体）----------
  const chunk = (arr, n) => arr.map((_, i) => (i % n === 0 ? arr.slice(i, i + n) : null)).filter(Boolean);
  const batches = chunk(pool, 5);
  let finCount = 0, perfCount = 0;
  for (let i = 0; i < batches.length; i++) {
    const names = batches[i].map((s) => s.name).join('、');
    const fin = await fetchOnce(
      `02_financials_${i + 1}.json`, 'stock', 'get_stock_financials',
      { query: `${names}的最新ROE、净利润增速、市盈率、资产负债率` },
      `财务批次${i + 1}/${batches.length}`
    );
    const finText = unwrapToMarkdown(fin);
    if (finText) fs.writeFileSync(path.join(RAW, `02_financials_${i + 1}_text.txt`), finText, 'utf-8');
    if (fin && fin.ok) finCount++;
  }

  // ---------- 第 4 步：分批查行情（最新收盘价 + 近60日涨跌幅）----------
  for (let i = 0; i < batches.length; i++) {
    const names = batches[i].map((s) => s.name).join('、');
    const perf = await fetchOnce(
      `03_performance_${i + 1}.json`, 'stock', 'get_stock_performance',
      { query: `${names}的最新收盘价、近60日涨跌幅` },
      `行情批次${i + 1}/${batches.length}`
    );
    const perfText = unwrapToMarkdown(perf);
    if (perfText) fs.writeFileSync(path.join(RAW, `03_performance_${i + 1}_text.txt`), perfText, 'utf-8');
    if (perf && perf.ok) perfCount++;
  }

  console.log(`\n[done] 取数完成: 财务成功 ${finCount}/${batches.length} 批, 行情成功 ${perfCount}/${batches.length} 批`);
  console.log('[done] raw/ 目录文件清单:');
  fs.readdirSync(RAW).sort().forEach((f) => console.log('   ', f));
}

main().catch((e) => {
  console.error('[fatal] 取数失败:', e && e.message ? e.message : e);
  console.error('降级方案：parse_and_screen.py 会检测 raw/ 缺失并改用 data/cache 完成同口径流程');
  process.exit(0); // 教学规范：脚本永不非0退出，失败信息留痕
});
