// step1 智能选股：最近10年涨幅超1000%的A股（iFinD search_stocks）
// 调用方式：node script.js（需在工作区内、call-node.js 可访问）
const { call } = require('/home/ppku/zcodespace/tonghuashun/.zcode/skills/ifind-finance-data/call-node.js');

async function main() {
    const result = await call("stock", "search_stocks", {
        query: "最近10年涨幅超过1000%的A股股票"
    });
    console.log(JSON.stringify(result, null, 2));
}

main().catch(console.error);