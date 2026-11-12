#!/usr/bin/env python3
"""用官方站点列表校验此前用 Nominatim 补的经纬度。

官方文件在 全国空气质量/_站点列表/，含监测点编码、名称、城市、经纬度。
国控站在北京只有 12 个，与市级 35 站网交集 12 个，正好可做对照。
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LIST_DIR = REPO / "全国空气质量" / "_站点列表"


def haversine(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def read_list(path: Path):
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main():
    latest = sorted(LIST_DIR.glob("站点列表-*.csv"))
    print("可用站点列表：")
    for p in latest:
        print(f"  {p.name} ({p.stat().st_size/1024:.0f} KB)")

    target = LIST_DIR / "站点列表-2020.01.01起.csv"
    rows = read_list(target)
    print(f"\n使用 {target.name}：{len(rows)} 站，字段 {list(rows[0].keys())}")

    cities = {}
    for r in rows:
        c = r["城市"].strip()
        cities[c] = cities.get(c, 0) + 1
    print(f"城市数 {len(cities)}；含『北京』的城市键：{[k for k in cities if '北京' in k]}")

    bj = [r for r in rows if r["城市"].strip().startswith("北京")]
    print(f"北京国控站 {len(bj)} 个：{[r['监测点名称'] for r in bj]}")

    official = {r["监测点名称"].strip(): (float(r["经度"]), float(r["纬度"]), r["监测点编码"]) for r in bj}

    st = pd.read_csv(REPO / "data" / "meta" / "stations_beijing.csv")
    st = st[st["network_era"] == "A"]

    print(f"\n{'站名':<10}{'官方经纬度':<22}{'Nominatim经纬度':<22}{'偏差km':>8}")
    print("-" * 64)
    diffs, matched = [], 0
    for _, s in st.iterrows():
        name = s["name_cn"]
        if name not in official:
            continue
        olon, olat, code = official[name]
        d = haversine(olon, olat, s["lon"], s["lat"])
        diffs.append((name, d))
        matched += 1
        print(f"{name:<10}{f'{olon:.3f},{olat:.3f}':<22}{f'{s['lon']:.3f},{s['lat']:.3f}':<22}{d:8.2f}")

    if diffs:
        ds = sorted(x[1] for x in diffs)
        print(f"\n可比对 {matched} 站 | 偏差中位 {ds[len(ds)//2]:.2f} km | 最大 {ds[-1]:.2f} km")
        print(f"偏差 >5km 的站：{[n for n, d in diffs if d > 5]}")

    print(f"\n未在官方表出现的市级站（共 {len(st) - matched} 个）：")
    print("  " + "、".join(sorted(set(st['name_cn']) - set(official))))


if __name__ == "__main__":
    main()
