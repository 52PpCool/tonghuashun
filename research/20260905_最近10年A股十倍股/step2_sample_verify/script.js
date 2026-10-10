// step2 抽查核验：用 get_stock_performance 独立核验引擎的涨幅是否可信
const { call } = require('/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js');

async function main() {
    const result = await call("stock", "get_stock_performance", {
        query: "贵州茅台、中际旭创、重庆啤酒、潍柴动力、广和通 2016-09-01至2026-08-31区间涨跌幅（前复权）"
    });
    console.log(JSON.stringify(result, null, 2));
}

main().catch(console.error);