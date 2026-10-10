#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_profile.py —— 十倍股画像 第三步：解析 iFinD 返回的 markdown 表 → 结构化 CSV

输入：raw/01_info_batch*.json（上市日期+同花顺一级行业）、raw/02_mktcap_batch*.json（2016-09-01 总市值）
输出：data/profile_basic.csv（code,name,list_date,ths_industry,mktcap_20160901_yi）

解析坑（真实留痕）：
  - 返回是 markdown 管道表套在两层 JSON 转义里；列数因批次而异（info 表含竞争公司等长文本列，
    内部没有转义的 | 但公司名里可能出现英文逗号，按 | 切分是安全的）。
  - 市值单位是"亿"（如 26.6773亿）也有"万"级兜底；未上市股票返回 0.0 → 记 NaN 并标记未上市。
  - 上市日期为 YYYYMMDD 字符串。
"""
import glob
import json
import os
import re

import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(DIR, "raw")
OUT = os.path.join(DIR, "data", "profile_basic.csv")


def extract_table(payload):
    """从 iFinD 响应的嵌套 JSON 里抠出 markdown 表文本。"""
    try:
        text = payload["response"]["data"]["result"]["content"][0]["text"]
    except (KeyError, TypeError, IndexError):
        try:
            text = payload["response"]["data"]["result"]["content"][0]["text"]
        except Exception:  # noqa: BLE001
            return None
    inner = json.loads(text)
    return inner.get("data", {}).get("answer", "")


def parse_md_table(md):
    rows = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):  # 分隔行
            continue
        rows.append(cells)
    return rows


def main():
    # 1) 上市日期 + 行业
    info_rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "01_info_batch*.json"))):
        md = extract_table(json.load(open(f, encoding="utf-8")))
        if not md:
            print(f"[warn] {os.path.basename(f)} 无表")
            continue
        rows = parse_md_table(md)
        header = rows[0]
        idx = {h: i for i, h in enumerate(header)}
        for r in rows[1:]:
            info_rows.append({
                "code": r[idx["证券代码"]],
                "name": r[idx["证券简称"]],
                "list_date": r[idx["上市日期"]],
                "ths_industry": r[idx["所属同花顺行业"]],
            })
    df_info = pd.DataFrame(info_rows)
    print(f"[info] 上市日期+行业：{len(df_info)} 条（应为88）")

    # 2) 2016-09-01 总市值
    cap_rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "02_mktcap_batch*.json"))):
        md = extract_table(json.load(open(f, encoding="utf-8")))
        if not md:
            print(f"[warn] {os.path.basename(f)} 无表")
            continue
        rows = parse_md_table(md)
        header = rows[0]
        idx = {h: i for i, h in enumerate(header)}
        for r in rows[1:]:
            raw_cap = r[idx["总市值（单位：元）"]]
            m = re.match(r"([\d.]+)(亿|万|元)?", raw_cap)
            if not m:
                cap_rows.append({"code": r[idx["证券代码"]], "name": r[idx["证券简称"]],
                                 "mktcap_raw": raw_cap, "mktcap_20160901_yi": float("nan")})
                continue
            val = float(m.group(1))
            unit = m.group(2) or "元"
            yi = val if unit == "亿" else (val / 1e4 if unit == "万" else val / 1e8)
            cap_rows.append({"code": r[idx["证券代码"]], "name": r[idx["证券简称"]],
                             "mktcap_raw": raw_cap, "mktcap_20160901_yi": yi})
    df_cap = pd.DataFrame(cap_rows)
    print(f"[info] 2016-09-01 总市值：{len(df_cap)} 条（应为88）")

    df = df_info.merge(df_cap[["code", "mktcap_raw", "mktcap_20160901_yi"]], on="code", how="outer")
    # 未上市 → 市值 0.0 视为缺失（起点市值改用上市时点，分析脚本处理）
    n_zero = int((df["mktcap_20160901_yi"] == 0).sum())
    df.loc[df["mktcap_20160901_yi"] == 0, "mktcap_20160901_yi"] = float("nan")
    df["listed_before_window"] = df["mktcap_raw"].notna() & (df["mktcap_raw"] != "0.0")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[done] 写出 {OUT}：{len(df)} 行；2016-09-01 前已上市 {int(df['listed_before_window'].sum())} 只，"
          f"窗口内才上市（市值=0标记为NaN）{n_zero} 只")
    print(df.head(8).to_string())
    print("行业分布（前12）：")
    print(df["ths_industry"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
