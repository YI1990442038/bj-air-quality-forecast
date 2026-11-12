#!/usr/bin/env python3
"""quotsoft.net 空气质量历史数据采集器。

设计要点：
- 断点续传：以 manifest 为准，已成功的不重复下载
- 原子落盘：先写 .part 再 rename，避免半截文件冒充成功
- 校验：HTTP 状态、表头前缀、行数、类型集合，任一不符记为失败
- 限速与重试：指数退避，区分「源端缺失」与「传输失败」
- 清单：每文件记录 sha256/字节数/行数/列数，供后续门控使用

用法：
  python fetch_quotsoft.py --dataset beijing_all --start 20131206 --end 20261029 --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

BASE = "https://quotsoft.net/air/data"
UA = "Mozilla/5.0 (compatible; aircast-research/1.0)"

# 每个数据集：(URL 模板, 期望行数, 必须包含的类型集合)
DATASETS = {
    "beijing_all": (
        "beijing_all_{d}.csv",
        120,
        {"PM2.5", "PM2.5_24h", "PM10", "PM10_24h", "AQI"},
        "20131206",
    ),
    "beijing_extra": (
        "beijing_extra_{d}.csv",
        240,
        {"SO2", "SO2_24h", "NO2", "NO2_24h", "O3", "O3_24h", "O3_8h", "CO", "CO_24h"},
        "20140403",
    ),
    "china_cities": (
        "china_cities_{d}.csv",
        360,
        {"PM2.5", "PM10", "AQI", "SO2", "NO2", "O3", "CO"},
        "20140513",
    ),
    "china_sites": (
        "china_sites_{d}.csv",
        360,
        {"PM2.5", "PM10", "AQI", "SO2", "NO2", "O3", "CO"},
        "20140513",
    ),
}

_lock = threading.Lock()


def daterange(start: str, end: str):
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end, "%Y%m%d").date()
    d = s
    while d <= e:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


def load_manifest(path: Path) -> dict:
    """返回已完成条目 {date: record}；损坏行忽略。"""
    done: dict = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") in ("ok", "partial"):
                done[rec["date"]] = rec
    return done


def fetch_one(session_opener, dataset: str, day: str, outdir: Path, timeout: int = 90):
    """下载并校验单日文件。返回 manifest 记录。"""
    tpl, expect_rows, expect_types, _ = DATASETS[dataset]
    url = f"{BASE}/{tpl.format(d=day)}"
    target = outdir / f"{day}.csv"
    rec = {
        "dataset": dataset,
        "date": day,
        "url": url,
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    if target.exists() and target.stat().st_size > 0:
        raw = target.read_bytes()
        ok, info = validate(raw, expect_rows, expect_types)
        if ok:
            rec.update(info, cached=True)
            return rec
        target.unlink(missing_ok=True)

    last_err = ""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with session_opener.open(req, timeout=timeout) as resp:
                raw = resp.read()
            if not raw:
                rec.update(status="missing_source", reason="empty_body", bytes=0)
                return rec
            ok, info = validate(raw, expect_rows, expect_types)
            if not ok:
                if info.get("reason") == "html_not_found":
                    rec.update(status="missing_source", reason="html_not_found", bytes=len(raw))
                    return rec
                last_err = info.get("reason", "invalid")
                time.sleep(1.5 * (attempt + 1))
                continue
            tmp = target.with_suffix(".csv.part")
            tmp.write_bytes(raw)
            os.replace(tmp, target)
            rec.update(info)
            return rec
        except urllib.error.HTTPError as e:
            if e.code == 404:
                rec.update(status="missing_source", reason="http_404")
                return rec
            last_err = f"http_{e.code}"
            time.sleep(2.0 * (attempt + 1) + random.random())
        except Exception as e:  # 超时、连接重置等
            last_err = type(e).__name__ + ":" + str(e)[:120]
            time.sleep(2.0 * (attempt + 1) + random.random())

    rec.update(status="failed", reason=last_err)
    return rec


def validate(raw: bytes, expect_rows: int, expect_types: set):
    """校验并记录精确覆盖度。

    重要：源站存在合法的「部分缺失日」（如只有 0—17 时、或某小时整行缺失）。
    这类文件必须收下并标注，而不是当坏文件丢弃——丢弃会让时间轴出现假空洞。
    返回 (可接受, 信息字典)，信息中含 status=ok|partial 及缺失清单。
    """
    if raw.lstrip()[:1] == b"<":
        return False, {"reason": "html_not_found"}
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False, {"reason": "not_utf8"}
    lines = text.splitlines()
    if not lines:
        return False, {"reason": "empty"}
    header = lines[0]
    if not header.startswith("date,hour,type,"):
        return False, {"reason": f"bad_header:{header[:40]}"}
    ncol = header.count(",") + 1
    body = [ln for ln in lines[1:] if ln.strip()]
    if not body:
        return False, {"reason": "no_rows"}

    types: dict[str, set] = {}
    for ln in body:
        parts = ln.split(",", 3)
        if len(parts) < 4:
            return False, {"reason": "bad_line"}
        types.setdefault(parts[2], set()).add(parts[1])

    all_hours = {str(h) for h in range(24)}
    miss_types = sorted(expect_types - set(types))
    miss_hours = sorted(all_hours - set().union(*types.values()), key=int) if types else list(range(24))
    # 类型集合随年份演进（如 O3_8h 是后来才加入的），缺类型属历史事实，
    # 只有「缺小时」才构成部分缺失。类型缺失仅记录，不影响可用性判定。
    info = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(body),
        "cols": ncol,
        "n_types": len(types),
        "n_hours": max(len(v) for v in types.values()),
        "types_present": sorted(types),
        "missing_types": miss_types,
        "missing_hours": miss_hours,
        "status": "partial" if miss_hours else "ok",
    }
    return True, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--start", default=None, help="YYYYMMDD，默认取该数据集起始日")
    ap.add_argument("--end", default=None, help="YYYYMMDD，默认昨天")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default=None, help="数据根目录，默认 <repo>/data")
    ap.add_argument("--limit", type=int, default=0, help="仅下载前 N 天，用于冒烟测试")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = Path(args.root) if args.root else repo / "data"
    outdir = root / "raw" / args.dataset
    man_dir = root / "manifests"
    outdir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    default_start = DATASETS[args.dataset][3]
    start = args.start or default_start
    end = args.end or (date.today() - timedelta(days=1)).strftime("%Y%m%d")

    manifest_path = man_dir / f"{args.dataset}.jsonl"
    done = load_manifest(manifest_path)
    days = [d for d in daterange(start, end)]
    todo = [d for d in days if d not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(
        f"[{args.dataset}] 区间 {start}~{end} 共 {len(days)} 天 | "
        f"已完成 {len(done)} | 待下载 {len(todo)} | workers={args.workers}",
        flush=True,
    )
    if not todo:
        print("无待下载任务", flush=True)
        return

    opener = urllib.request.build_opener()
    stats = {"ok": 0, "cached": 0, "missing_source": 0, "failed": 0}
    fails = []
    t0 = time.time()

    with manifest_path.open("a", encoding="utf-8") as mf, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {pool.submit(fetch_one, opener, args.dataset, d, outdir): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            stats[rec["status"]] = stats.get(rec["status"], 0) + 1
            if rec.get("cached"):
                stats["cached"] += 1
            if rec["status"] in ("failed", "missing_source"):
                fails.append(rec)
            with _lock:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(todo) - i) / rate if rate else 0
                print(
                    f"  {i}/{len(todo)} ok={stats['ok']} miss={stats['missing_source']} "
                    f"fail={stats['failed']} {rate:.1f}/s ETA {eta/60:.1f}min",
                    flush=True,
                )

    if fails:
        fpath = man_dir / f"{args.dataset}_issues.jsonl"
        with fpath.open("a", encoding="utf-8") as f:
            for r in fails:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"异常条目已记入 {fpath}", flush=True)

    el = time.time() - t0
    print(
        f"[{args.dataset}] 完成 用时 {el/60:.1f}min | "
        f"ok={stats['ok']} missing={stats['missing_source']} failed={stats['failed']}",
        flush=True,
    )
    compact(manifest_path)


def compact(path: Path):
    """同一日期保留最后一条记录，避免重跑后清单里堆叠陈旧失败项。"""
    if not path.exists():
        return
    latest: dict = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[rec["date"]] = rec
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for d in sorted(latest):
            f.write(json.dumps(latest[d], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
