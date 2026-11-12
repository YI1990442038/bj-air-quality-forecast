#!/usr/bin/env python3
"""Open-Meteo 历史气象采集（ERA5 存档，小时级）。

为什么不用 quotsoft 自带气象数据：那份在百度网盘，需人工下载且为单站 NCDC 记录，
无法按站点位置对齐。Open-Meteo 的 archive 接口可按任意经纬度取 ERA5 网格值，
天然支持"每个空气站取最近格点"，且可脚本化、可重放。

覆盖点：北京 5x5 网格（25 点）+ 京津冀 13 城中心（13 点）= 38 点
输出：data/raw/meteo/{year}_{batch}.json，清单记入 data/manifests/meteo.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

API = "https://archive-api.open-meteo.com/v1/archive"
UA = "aircast-research/1.0"

VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "boundary_layer_height",
    "shortwave_radiation",
    "vapour_pressure_deficit",
]

# 北京用 0.25° 对齐网格。ERA5 原生网格即 0.25°，取 0.25° 整数倍点位可保证
# 每个空气站拿到「包含它的那个 ERA5 格点」，站点到格点中心不超过半格对角线（~19 km）。
# 早先用 0.5° 间隔（9 点）导致中位偏差 15.5 km、最大 39 km，等于可能取到邻格，已废弃。
BEIJING_GRID = [
    (round(lat * 4) / 4, round(lon * 4) / 4)
    for lat in (39.5, 39.75, 40.0, 40.25, 40.5)
    for lon in (116.0, 116.25, 116.5, 116.75, 117.0)
]

# 京津冀 13 城中心（城市中心坐标，公开常识值）
CITIES = {
    "北京": (39.9042, 116.4074),
    "天津": (39.3434, 117.3616),
    "石家庄": (38.0428, 114.5149),
    "唐山": (39.6305, 118.1802),
    "秦皇岛": (39.9354, 119.5996),
    "邯郸": (36.6255, 114.5391),
    "保定": (38.8740, 115.4646),
    "张家口": (40.8118, 114.8879),
    "承德": (40.9515, 117.9634),
    "廊坊": (39.5378, 116.6835),
    "沧州": (38.3045, 116.8388),
    "衡水": (37.7389, 115.6707),
    "邢台": (37.0708, 114.5048),
}

# 每请求点数。实测 22 点 × 11 变量 × 全年 只需 5 秒，一次请求拿完一年最优：
# 请求数越少，越不容易触发免费额度的「按请求计数」限流。
BATCH = 64


def build_points(repo: Path):
    """取点顺序必须稳定，因为结果按批内顺序回填。

    北京用 3x3 网格（9 点）+ 京津冀 13 城 = 22 点。
    免费额度按「点位数 × 变量数 × 年数」加权，22x11x14 = 3388，控制在每小时 5000 以内；
    早先按 35 个站点逐点取（48 点 x 15 变量 = 10080）会直接被限流，且超出 ERA5 分辨率毫无收益。
    """
    pts = [("bjsite", i, lat, lon) for i, (lat, lon) in enumerate(BEIJING_GRID)]
    seen = {(round(p[2], 4), round(p[3], 4)) for p in pts}
    for name, (lat, lon) in CITIES.items():
        key = (round(lat, 4), round(lon, 4))
        if key in seen:
            continue
        seen.add(key)
        pts.append(("city", name, lat, lon))
    return pts


def year_windows(start: str, end: str):
    """按自然年切分，返回 [(label, start_date, end_date)]。"""
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end, "%Y%m%d").date()
    out = []
    for y in range(s.year, e.year + 1):
        cs = max(s, date(y, 1, 1))
        ce = min(e, date(y, 12, 31))
        out.append((str(y), cs.isoformat(), ce.isoformat()))
    return out


def fetch_batch(dest: Path, label: str, s: str, e: str, pts, timeout=300):
    lats = ",".join(f"{p[2]:.4f}" for p in pts)
    lons = ",".join(f"{p[3]:.4f}" for p in pts)
    q = {
        "latitude": lats,
        "longitude": lons,
        "start_date": s,
        "end_date": e,
        "hourly": ",".join(VARS),
        "timezone": "Asia/Shanghai",
        # 锁定最终版 ERA5。默认的 best_match 会在 ERA5 缺位时静默回退到其他模式，
        # 实测它会给「今天」返回包含尚未发生时段的数据——那等于把未来信息当已知。
        # 锁定的代价是最后约三周为空（ERA5 有发布延迟），缺口保留为 NaN，不插值。
        "models": "era5",
    }
    url = API + "?" + urllib.parse.urlencode(q, safe=",")

    if dest.exists() and dest.stat().st_size > 1000:
        try:
            old = json.loads(dest.read_text(encoding="utf-8"))
            return {"file": dest.name, "status": "cached", "points": len(old)}
        except Exception:
            dest.unlink(missing_ok=True)

    last = ""
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("error"):
                last = str(data.get("reason"))
                time.sleep(30)
                continue
            if not isinstance(data, list):
                last = "unexpected_shape"
                time.sleep(5)
                continue
            tmp = dest.with_suffix(".json.part")
            tmp.write_bytes(raw)
            os.replace(tmp, dest)
            return {"file": dest.name, "status": "ok", "points": len(data), "bytes": len(raw)}
        except urllib.error.HTTPError as h:
            last = f"http_{h.code}"
            if h.code == 429:
                # 免费额度按小时重置，硬等一小时不如按窗口轮询
                print(f"    [429] {dest.name} 触发小时配额，等待 300s 后重试（第 {attempt+1} 次）", flush=True)
                time.sleep(300)
            else:
                time.sleep(8 * (attempt + 1))
        except Exception as ex:
            last = type(ex).__name__ + ":" + str(ex)[:100]
            time.sleep(8 * (attempt + 1))
    return {"file": dest.name, "status": "failed", "reason": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20131201")
    ap.add_argument("--end", default=None, help="默认昨天")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = Path(args.root) if args.root else repo / "data"
    outdir = root / "raw" / "meteo"
    man_dir = root / "manifests"
    outdir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    end = args.end or (date.today() - timedelta(days=2)).strftime("%Y%m%d")
    windows = year_windows(args.start, end)
    points = build_points(repo)
    batches = [points[i : i + BATCH] for i in range(0, len(points), BATCH)]

    tasks = []
    for label, s, e in windows:
        for bi, pts in enumerate(batches):
            dest = outdir / f"{label}_b{bi}.json"
            tasks.append((dest, f"{label}_b{bi}", s, e, pts))

    print(
        f"[meteo] {len(windows)} 个年段 × {len(batches)} 批 = {len(tasks)} 个请求 "
        f"| 点位 {len(points)} | workers={args.workers}",
        flush=True,
    )
    t0 = time.time()
    stats = {"ok": 0, "cached": 0, "failed": 0}
    man = man_dir / "meteo.jsonl"

    with man.open("a", encoding="utf-8") as mf, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_batch, *t): t[1] for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            stats[rec["status"]] = stats.get(rec["status"], 0) + 1
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            print(f"  {i}/{len(tasks)} {rec['file']} {rec['status']} pts={rec.get('points')}", flush=True)
            time.sleep(3)  # 主动降速，避免把小时额度瞬间打满

    print(
        f"[meteo] 完成 用时 {(time.time()-t0)/60:.1f}min ok={stats['ok']} "
        f"cached={stats['cached']} failed={stats['failed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
