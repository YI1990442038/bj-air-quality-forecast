#!/usr/bin/env python3
"""从百度网盘年度打包中提取逐日文件。

为什么要用它：站点级数据（china_sites）逐日下载需约 4500 次请求 / 7.6 GB，
打包下载仅 2.9 GB 压缩包，且内容与逐日抓取逐字节一致（已抽样验证）。

处理要点：
- zip 内文件名是 GBK 编码，被 Python 按 cp437 解码成乱码，需回转
- 跳过 macOS 垃圾（.DS_Store / __MACOSX）
- 已存在且内容相同的文件不重写；本地缺失的才补，并单独报告
"""
from __future__ import annotations

import argparse
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "全国空气质量"


def fix_name(raw: str) -> str:
    try:
        return raw.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def scan_package(pattern: str):
    """返回 {日期: (zip 路径, zip 内条目名)}。

    必须剔除 macOS 的 AppleDouble 条目（形如 `._china_sites_20260101.csv`，
    固定 4096 字节，头为 `\\x00\\x05\\x16\\x07`）。它们与真数据同名且常排在后面，
    一旦入选就会把可用数据替换成二进制垃圾——本项目已实际踩过一次。
    """
    index = {}
    for zp in sorted(PKG.glob(pattern)):
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                name = fix_name(info.filename)
                base = Path(name).name
                if not name.endswith(".csv") or "__MACOSX" in name:
                    continue
                if base.startswith("._") or base in (".DS_Store", "Thumbs.db"):
                    continue
                # 不按体积过滤：2018-12-22~26 有 5 个仅 169 字节的合法空文件
                # （源站整日故障），按尺寸过滤会误删真实缺口证据。
                stem = Path(base).stem
                parts = stem.split("_")
                if len(parts) < 3:
                    continue
                index[parts[-1]] = (zp, info.filename)
    return index


def extract(kind: str, target: Path, only_missing: bool):
    pattern = "站点_*.zip" if kind == "china_sites" else "城市_*.zip"
    index = scan_package(pattern)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[{kind}] 包内共 {len(index)} 天")

    have = {p.stem for p in target.glob("*.csv")}
    todo = sorted(d for d in index if d not in have) if only_missing else sorted(index)
    print(f"  本地已有 {len(have)} 天 | 需处理 {len(todo)} 天")

    added, replaced, same = 0, 0, 0
    for d in todo:
        zp, inner = index[d]
        with zipfile.ZipFile(zp) as z:
            data = z.read(inner)
        dest = target / f"{d}.csv"
        if dest.exists():
            if dest.read_bytes() == data:
                same += 1
                continue
            replaced += 1
        else:
            added += 1
        tmp = dest.with_suffix(".csv.part")
        tmp.write_bytes(data)
        tmp.replace(dest)

    total = len(list(target.glob("*.csv")))
    print(f"  新增 {added} | 覆盖 {replaced} | 相同跳过 {same} | 目标目录共 {total} 天")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["china_sites", "china_cities", "both"], default="both")
    ap.add_argument("--overwrite", action="store_true", help="已存在也重写（默认只补缺失）")
    args = ap.parse_args()

    t0 = datetime.now()
    if args.kind in ("china_sites", "both"):
        extract("china_sites", REPO / "data" / "raw" / "china_sites", not args.overwrite)
    if args.kind in ("china_cities", "both"):
        extract("china_cities", REPO / "data" / "raw" / "china_cities", not args.overwrite)
    print(f"用时 {(datetime.now()-t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
