#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_control.py —— 伪十倍股对照画像 第二步：解析 iFinD markdown 表 → data/profile_control.csv
解析逻辑与 20261001_十倍股画像/parse_profile.py 一致（含"亿/万"单位与未上市=0.0→NaN 处理）。
"""
import glob
import json
import os
import re

import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(DIR, "raw")
OUT = os.path.join(DIR, "data", "profile_control.csv")


def extract_table(payload):
    text = payload["response"]["data"]["result"]["content"][0]["text"]
    inner = json.loads(text)
    return inner.get("data", {}).get("answer", "")


def parse_md_table(md):
    rows = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def main():
    info_rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "01_info_batch*.json"))):
        md = extract_table(json.load(open(f, encoding="utf-8")))
        rows = parse_md_table(md)
        idx = {h: i for i, h in enumerate(rows[0])}
        for r in rows[1:]:
            info_rows.append({"code": r[idx["证券代码"]], "name": r[idx["证券简称"]],
                              "list_date": r[idx["上市日期"]], "ths_industry": r[idx["所属同花顺行业"]]})
    df_info = pd.DataFrame(info_rows)

    cap_rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "02_mktcap_batch*.json"))):
        md = extract_table(json.load(open(f, encoding="utf-8")))
        rows = parse_md_table(md)
        idx = {h: i for i, h in enumerate(rows[0])}
        for r in rows[1:]:
            raw_cap = r[idx["总市值（单位：元）"]]
            m = re.match(r"([\d.]+)(亿|万|元)?", raw_cap)
            yi = float("nan")
            if m:
                val, unit = float(m.group(1)), m.group(2) or "元"
                yi = val if unit == "亿" else (val / 1e4 if unit == "万" else val / 1e8)
            cap_rows.append({"code": r[idx["证券代码"]], "name": r[idx["证券简称"]],
                             "mktcap_raw": raw_cap, "mktcap_20160901_yi": yi})
    df_cap = pd.DataFrame(cap_rows)

    df = df_info.merge(df_cap[["code", "mktcap_raw", "mktcap_20160901_yi"]], on="code", how="outer")
    n_zero = int((df["mktcap_20160901_yi"] == 0).sum())
    df.loc[df["mktcap_20160901_yi"] == 0, "mktcap_20160901_yi"] = float("nan")
    df["listed_before_window"] = df["mktcap_raw"].notna() & (df["mktcap_raw"] != "0.0")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[done] {OUT}: {len(df)} 行；2016-09-01 前已上市 {int(df['listed_before_window'].sum())} 只，"
          f"窗口内上市 {n_zero} 只")
    print(df["ths_industry"].value_counts().to_string())


if __name__ == "__main__":
    main()
