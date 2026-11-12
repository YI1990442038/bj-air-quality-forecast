#!/usr/bin/env python3
"""站点元数据构建：从数据本体取站点名，地理编码补经纬度。

权威性说明（重要）：
- 站点名称来自 quotsoft 日文件表头，是数据对齐的唯一依据，权威。
- 经纬度官方文件《站点列表.csv》在百度网盘，需人工下载，暂不可得。
  本脚本用 OpenStreetMap Nominatim 地理编码补齐，标注 coord_source=nominatim_osm，
  并保留 raw_response，便于日后被官方表覆盖时逐条比对。
- 下游按经纬度做 station_key 映射，而非按列顺序，避免站点表变更导致静默错位。
"""
from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = "aircast-research/1.0 (academic project; contact: local)"

# 官方站点名中可能被 OSM 误解的别名
ALIAS = {
    "万寿西宫": "万寿西宫 北京",
    "北部新区": "北部新区 北京 昌平",
    "丰台花园": "丰台花园 北京",
    "云岗": "云岗 北京 丰台",
    "古城": "古城 北京 石景山",
    "东高村": "东高村 北京 平谷",
    "永乐店": "永乐店 北京 通州",
    "榆垡": "榆垡 北京 大兴",
    "琉璃河": "琉璃河 北京 房山",
    "前门": "前门 北京 东城",
    "永定门内": "永定门 北京 西城",
    "西直门北": "西直门 北京",
    "南三环": "南三环 北京 丰台",
    "东四环": "东四环 北京 朝阳",
    "密云水库": "密云水库 北京",
    "八达岭": "八达岭 北京 延庆",
    "定陵": "定陵 北京 昌平",
    "植物园": "北京植物园",
    "农展馆": "全国农业展览馆 北京",
    "奥体中心": "奥体中心 北京 朝阳",
}


DISTRICTS = [
    "东城", "西城", "朝阳", "海淀", "丰台", "石景山", "通州", "顺义",
    "昌平", "大兴", "房山", "门头沟", "平谷", "怀柔", "密云", "延庆",
]


def query_for(name: str) -> str:
    """把「区名+站名」拆成可检索的地名。"""
    if name in ALIAS:
        return ALIAS[name]
    base = name.replace("(对照点)", "").strip()
    if base in ALIAS:
        return ALIAS[base]
    for d in DISTRICTS:
        if base.startswith(d) and len(base) > len(d):
            rest = base[len(d) :]
            return ALIAS.get(rest, f"{rest} {d}区 北京")
    return f"{base} 北京"


def geocode(query: str, cache: dict, cache_path: Path):
    key = query
    if key in cache:
        return cache[key]
    url = NOMINATIM + "?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 3, "accept-language": "zh"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:
        data = [{"error": str(e)[:100]}]
    time.sleep(1.1)  # Nominatim 使用政策：不超过 1 req/s
    cache[key] = data
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def pick(data):
    """返回 (lon, lat, display_name, confidence)。"""
    for item in data:
        if "error" in item:
            continue
        try:
            lat, lon = float(item["lat"]), float(item["lon"])
        except (KeyError, ValueError):
            continue
        name = item.get("display_name", "")
        # 是否落在北京市行政范围内（粗略包围盒）
        in_bj = 115.4 <= lon <= 117.6 and 39.4 <= lat <= 41.2
        conf = "high" if (in_bj and "北京" in name) else ("medium" if in_bj else "low")
        return lon, lat, name, conf
    return None, None, "", "none"


def main():
    repo = Path(__file__).resolve().parent.parent
    meta = repo / "data" / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    cache_path = meta / "nominatim_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    # 站网在 2021-01-24 换代：前后都是 35 站，但只有 10 个是纯改名，
    # 其余为新设站。因此两个纪元分别取表头、分别地理编码，不可混用。
    files = sorted((repo / "data" / "raw" / "beijing_all").glob("*.csv"))
    era_b_file = files[-1]
    era_a_file = max(f for f in files if f.stem < "20210124")

    rows = []
    for era, src in (("A", era_a_file), ("B", era_b_file)):
        with src.open(encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh))
        names = header[3:]
        print(f"纪元 {era}（{src.stem}）站点 {len(names)} 个")
        for i, name in enumerate(names, 1):
            q = query_for(name)
            data = geocode(q, cache, cache_path)
            lon, lat, disp, conf = pick(data)
            rows.append(
                {
                    "station_key": f"BJ{era}{1000+i}",
                    "network_era": era,
                    "name_cn": name,
                    "city": "北京",
                    "lon": lon,
                    "lat": lat,
                    "col_index": i,
                    "coord_source": "nominatim_osm",
                    "coord_confidence": conf,
                    "query": q,
                    "matched_name": disp[:80],
                    "is_target": 1,
                }
            )

    out = meta / "stations_beijing.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if r["coord_confidence"] in ("low", "none")]
    print(f"\n写入 {out} | 共 {len(rows)} 条")
    for era in ("A", "B"):
        sub = [r for r in rows if r["network_era"] == era]
        print(f"  纪元 {era}: 高置信 {sum(1 for r in sub if r['coord_confidence']=='high')}/{len(sub)}")
    if bad:
        print(f"  需人工复核 {len(bad)} 条: {[r['name_cn'] for r in bad]}")


if __name__ == "__main__":
    main()
