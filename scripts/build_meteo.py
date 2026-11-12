#!/usr/bin/env python3
"""Open-Meteo JSON → 气象长表（按站点逐点取数）。

输出：
  data/warehouse/dim_meteo/point_id=<id>/part.parquet
  data/meta/meteo_coverage.csv          每点每年每变量覆盖率
  data/meta/meteo_snap.csv              请求坐标与 ERA5 吸附坐标的距离

关键点：
- 返回坐标是 ERA5 原生网格吸附值，与请求坐标不同，必须如实记录 geo_lon/geo_lat 与偏差。
- 边界层高度在个别时段有缺口，本脚本统计覆盖率，不插值、不填补。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_meteo import BATCH, build_points  # noqa: E402

VAR_CN = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "rh",
    "dew_point_2m": "dewp",
    "surface_pressure": "pres",
    "wind_speed_10m": "wspd",
    "wind_direction_10m": "wdir",
    "precipitation": "rain",
    "cloud_cover": "cloud",
    "boundary_layer_height": "blh",
    "shortwave_radiation": "ssrd",
    "vapour_pressure_deficit": "vpd",
}


def point_id(kind: str, key) -> str:
    if kind == "bjsite":
        return f"bj_{int(key):02d}"
    if kind == "grid":
        return f"grid_{int(key):02d}"
    if kind == "station":
        return f"st_{key}"
    return f"city_{key}"


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main():
    repo = Path(__file__).resolve().parent.parent
    raw = repo / "data" / "raw" / "meteo"
    out = repo / "data" / "warehouse" / "dim_meteo"
    meta = repo / "data" / "meta"
    out.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    points = build_points(repo)
    files = sorted(raw.glob("*.json"))
    print(f"[meteo] {len(files)} 个原始文件 | 点位 {len(points)}")

    frames, coverage, snap = [], [], []
    for f in files:
        parts = f.stem.split("_b")
        year, bidx = parts[0], int(parts[1])
        pts = points[bidx * BATCH : (bidx + 1) * BATCH]
        data = json.loads(f.read_text(encoding="utf-8"))
        if len(data) != len(pts):
            print(f"  ! {f.name} 返回 {len(data)} 点，期望 {len(pts)}")
        for pt, spec in zip(data, pts):
            kind, key, req_lat, req_lon = spec
            h = pt["hourly"]
            df = pd.DataFrame({"ts_local": pd.to_datetime(h["time"])})
            for en, cn in VAR_CN.items():
                df[cn] = pd.to_numeric(pd.Series(h.get(en)), errors="coerce")
            pid = point_id(kind, key)
            df["point_id"] = pid
            df["point_kind"] = kind
            df["geo_lat"] = pt["latitude"]
            df["geo_lon"] = pt["longitude"]
            df["elevation"] = pt.get("elevation")
            df["src_file"] = f.name
            frames.append(df)
            snap.append(
                {
                    "point_id": pid,
                    "req_lat": req_lat,
                    "req_lon": req_lon,
                    "geo_lat": pt["latitude"],
                    "geo_lon": pt["longitude"],
                    "snap_km": round(haversine(req_lat, req_lon, pt["latitude"], pt["longitude"]), 2),
                }
            )
            coverage.append(
                {
                    "point_id": pid,
                    "year": year,
                    **{cn: round(float(df[cn].notna().mean()), 4) for cn in VAR_CN.values()},
                }
            )

    if not frames:
        raise SystemExit("没有可展平的气象文件")
    allm = pd.concat(frames, ignore_index=True)
    allm["year"] = allm["ts_local"].dt.year
    allm["ts_utc"] = allm["ts_local"] - pd.Timedelta(hours=8)

    for old in out.glob("point_id=*"):
        for p in old.glob("*.parquet"):
            p.unlink()

    total = 0
    for pid, g in allm.groupby("point_id"):
        d = out / f"point_id={pid}"
        d.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["year"]).to_parquet(d / "part.parquet", index=False, compression="zstd")
        total += len(g)

    pd.DataFrame(coverage).to_csv(meta / "meteo_coverage.csv", index=False, encoding="utf-8")
    sp = pd.DataFrame(snap).drop_duplicates("point_id")
    sp.to_csv(meta / "meteo_snap.csv", index=False, encoding="utf-8")

    # 空气站 → 最近的北京气象格点（9 个），供下游按站取气象
    bj = sp[sp["point_id"].str.startswith("bj_")]
    st_path = meta / "stations_beijing.csv"
    if st_path.exists() and not bj.empty:
        st = pd.read_csv(st_path)
        rows = []
        for _, s in st.iterrows():
            if pd.isna(s["lat"]) or pd.isna(s["lon"]):
                continue
            d2 = bj.apply(lambda g: haversine(s["lat"], s["lon"], g["geo_lat"], g["geo_lon"]), axis=1)
            i = d2.idxmin()
            rows.append(
                {
                    "station_key": s["station_key"],
                    "name_cn": s["name_cn"],
                    "station_lat": s["lat"],
                    "station_lon": s["lon"],
                    "meteo_point_id": bj.loc[i, "point_id"],
                    "dist_km": round(float(d2.loc[i]), 2),
                }
            )
        m = pd.DataFrame(rows)
        m.to_csv(meta / "station_meteo_map.csv", index=False, encoding="utf-8")
        print(f"  站点→气象格点: 中位距离 {m['dist_km'].median():.1f} km，最大 {m['dist_km'].max():.1f} km")

    print(f"[meteo] 写出 {total:,} 行 → {out}")
    print(f"  时段 {allm['ts_local'].min()} ~ {allm['ts_local'].max()} | 点位 {allm['point_id'].nunique()}")
    print(f"  ERA5 吸附距离: 中位 {sp['snap_km'].median():.1f} km，最大 {sp['snap_km'].max():.1f} km")
    cov = pd.DataFrame(coverage)
    worst = cov[list(VAR_CN.values())].min().sort_values()
    print(f"  覆盖率最低变量: { {k: round(v,3) for k,v in worst.head(4).items()} }")


if __name__ == "__main__":
    main()
