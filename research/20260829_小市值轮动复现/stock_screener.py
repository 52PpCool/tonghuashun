#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同花顺 iFinD 智能选股：小市值轮动策略初筛（可复用脚本）

用法：
    python3 stock_screener.py [输出CSV路径]

条件（与原帖一致）：
    中小板股票，总市值 10~100 亿，单季度扣非净利润 > 0，单季度营收 > 1 亿

说明：
    - 单次智能选股返回行数有限（约 100 行）且结果不稳定，故按市值分段查询再合并去重；
    - 输出按总市值从小到大排序（小市值策略按最小市值一批选股）。
"""
import csv
import json
import os
import sys
import time

# 脚本位于 research/20260829_小市值轮动复现/，工作区根目录为再上两级
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKILL_DIR = os.path.join(WORKSPACE, ".zcode", "skills", "ifind-finance-data")
sys.path.insert(0, SKILL_DIR)
from call import call  # noqa: E402

SEGMENTS = [
    ("10-30", "中小板股票中总市值在10亿到30亿之间且单季度扣非净利润大于0且单季度营业收入大于1亿元的股票"),
    ("30-40", "中小板股票中总市值在30亿到40亿之间且单季度扣非净利润大于0且单季度营业收入大于1亿元的股票"),
    ("40-50", "中小板股票中总市值在40亿到50亿之间且单季度扣非净利润大于0且单季度营业收入大于1亿元的股票"),
    ("50-60", "中小板股票中总市值在50亿到60亿之间且单季度扣非净利润大于0且单季度营业收入大于1亿元的股票"),
    ("60-70", "中小板股票中总市值在60亿到70亿之间且单季度扣非净利润大于0且单季度营业收入大于1亿元的股票"),
    ("70-80", "中小板股票中总市值在70亿到80亿之间且单季度扣非净利润大于0且单季度营业收入大于1亿元的股票"),
    ("80-100", "中小板股票中总市值在80亿到100亿之间且单季度扣非净利润大于0且单季度营业收入大于1亿元的股票"),
]

MKT_CAP_COL_PREFIX = "总市值["


def parse_answer(answer):
    """按表头列名解析 markdown 表格，返回记录列表。"""
    lines = [l for l in answer.splitlines() if l.strip().startswith("|")]
    if not lines:
        return []
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for l in lines[2:]:  # 跳过表头行与分隔行
        cells = [c.strip() for c in l.strip("|").split("|")]
        if len(cells) < 2:
            continue
        rec = dict(zip(header, cells))
        if rec.get("股票代码"):
            rows.append(rec)
    return rows


def main(out_path="search_all.csv"):
    all_rows = {}
    for seg, q in SEGMENTS:
        time.sleep(3.5)  # 免费版限流，间隔 3 秒以上
        r = call("stock", "search_stocks", {"query": q})
        text = (r.get("data", {}).get("result", {}).get("content", [{}])[0]
                .get("text", ""))
        d = json.loads(text)
        data = json.loads(d.get("data", "{}"))
        answer = data.get("answer", "")
        cnt = data.get("selectedSecuritiesCount", "?")
        rows = parse_answer(answer)
        for rec in rows:
            code = rec["股票代码"]
            name = rec.get("股票简称", "")
            mv_col = next((c for c in rec if c.startswith(MKT_CAP_COL_PREFIX)), None)
            try:
                mv = float(rec.get(mv_col, "")) / 1e8
            except (TypeError, ValueError):
                mv = None
            all_rows[code] = (name, mv)
        print(f"[{seg}] 声明 {cnt} 条 / 解析 {len(rows)} 行",
              "✓" if rows else "✗ 解析失败")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["股票代码", "股票简称", "总市值(亿)"])
        for code, (name, mv) in sorted(all_rows.items(),
                                       key=lambda kv: (kv[1][1] is None, kv[1][1] or 0)):
            w.writerow([code, name, f"{mv:.2f}" if mv else ""])
    print(f"合并去重后共 {len(all_rows)} 只，已保存 {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "search_all.csv")