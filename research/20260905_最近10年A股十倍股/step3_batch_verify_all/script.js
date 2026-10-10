// step3 批量核验：对 137 只候选逐只用 get_stock_performance 核验前复权区间涨幅
// 每批 5 只、批间间隔 600ms（免费版并发上限 2/s），共 28 批。
// 数据来源：../数据/candidates.csv（step1 选股结果完整导出）
const { call } = require('/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js');
const fs = require('fs');
const path = require('path');

const CSV_PATH = path.join(__dirname, '..', '数据', 'candidates.csv');

const lines = fs.readFileSync(CSV_PATH, 'utf8').split('\n').filter(l => l.trim());
const rows = lines.slice(1).map(l => {
    const m = l.match(/^([^,]+),(.+?),([^,]*)$/);
    return m ? { code: m[1], name: m[2], searchChange: parseFloat(m[3] || '') } : null;
}).filter(Boolean);

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function verifyChunk(chunk) {
    // 注意：引号/逗号需走 JSON 转义，此处按会话实际使用的方式构造（代码+空格均可被引擎解析）
    const query = chunk.map(r => r.code).join('、') + ' 2016-09-01至2026-08-31区间涨跌幅（前复权）';
    try {
        const res = await call("stock", "get_stock_performance", { query });
        const raw = res?.data?.result?.content?.[0]?.text || '';
        const outer = JSON.parse(raw);
        const inner = JSON.parse(outer.data);
        const answer = inner.answer || '';
        const data = {};
        for (const line of answer.split('\n')) {
            const cells = line.split('|').map(c => c.trim()).filter(Boolean);
            if (cells.length >= 3 && cells[0].includes('.') && /\d/.test(cells[0]) && !cells[0].startsWith('证券')) {
                data[cells[0]] = { name: cells[1], change: parseFloat(cells[2]) };
            }
        }
        return data;
    } catch (e) {
        return { error: 'chunk-error: ' + String(e).slice(0, 120) };
    }
}

(async () => {
    const results = {};
    let failedChunks = 0;
    for (let i = 0; i < rows.length; i += 5) {
        const chunk = rows.slice(i, i + 5);
        const r = await verifyChunk(chunk);
        if (r.error) { console.error(`批 ${i / 5 + 1} 失败: ${r.error}`); failedChunks++; continue; }
        for (const code of Object.keys(r)) results[code] = r[code];
        console.log(`进度 ${i + chunk.length}/${rows.length}, 已核验 ${Object.keys(results).length}`);
        await sleep(600);
    }
    console.log('核验成功数:', Object.keys(results).length, '失败批次数:', failedChunks);
    // 汇总输出
    const merged = rows.map(r => ({
        code: r.code,
        name: r.name,
        searchChange: r.searchChange || null,
        verifiedChange: results[r.code] ? results[r.code].change : null
    })).filter(r => r.verifiedChange !== null);
    const real = merged.filter(r => r.verifiedChange >= 1000).sort((a, b) => b.verifiedChange - a.verifiedChange);
    const fake = merged.filter(r => r.verifiedChange < 1000).sort((a, b) => b.verifiedChange - a.verifiedChange);
    const summary = {
        window: "20160901-20260831",
        adjust: "前复权",
        candidates: rows.length,
        verified: merged.length,
        tenXCount: real.length,
        falsePositiveCount: fake.length,
        note: "engine_search_change 列存在高分红股复权虚增问题，以 verifiedChange 为准；verifiedChange>=1000(即10倍) 判定为十倍股"
    };
    fs.writeFileSync(path.join(__dirname, 'output.json'), JSON.stringify({ summary, tenX: real, falsePositive: fake }, null, 2));
    console.log('十倍股数量:', real.length, '| 误选数量:', fake.length);
})();