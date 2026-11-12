#!/usr/bin/env python3
"""全国站点级数据 → 京津冀范围站点小时长表。

与 beijing_all 的区别：
- beijing_all / beijing_extra 是北京市级 35 站网
- china_sites 是全国国控站，北京仅 12 个，河北 11 市共 58 个

站点编码会随年份增删改，官方提供多版《站点列表》。本脚本按「日期 → 当时生效的
最近一版站点表」解析编码归属，避免用 2020 年的表去解释 2014 年的编码。
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long import TYPE_MAP, finalize  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LIST_DIR = REPO / "全国空气质量" / "_站点列表"

JINGJINJI_CITIES = {
    "北京", "天津", "石家庄", "唐山", "秦皇岛", "邯郸", "保定",
    "张家口", "承德", "廊坊", "沧州", "衡水", "邢台",
}


def load_snapshots():
    """返回 [(生效日期 YYYYMMDD, {编码: (名称, 城市, 经度, 纬度)})]，按日期升序。

    官方站点表有两种形态：
    - 常规版（2016.11 起）：CSV/XLSX，列含 编码/名称/城市/经度/纬度
    - 旧版（946 站）：转置格式，编码一行、名称一行、城市一行，**无经纬度**
    旧版缺的坐标从后续版本按（名称, 城市）补齐。
    """
    snaps = []
    for p in LIST_DIR.iterdir():
        if p.suffix.lower() not in (".csv", ".xlsx"):
            continue
        m = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})起", p.name)
        m2 = re.search(r"(\d{4})\.(\d{1,2})起", p.name)
        if m:
            eff = f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}"
        elif m2:
            eff = f"{m2.group(1)}{int(m2.group(2)):02d}01"
        elif "旧" in p.name:
            eff = "20140513"
        else:
            continue
        try:
            mapping = _read_transposed(p) if "旧" in p.name else _read_standard(p)
        except Exception as e:
            print(f"  ! 跳过 {p.name}: {type(e).__name__} {e}")
            continue
        if mapping:
            snaps.append((eff, mapping))

    snaps.sort(key=lambda x: x[0])
    # 去重：同一天多版时保留条目最多的
    dedup = {}
    for eff, mp in snaps:
        if eff not in dedup or len(mp) > len(dedup[eff]):
            dedup[eff] = mp
    snaps = sorted(dedup.items())

    # 旧版无坐标，用后续版本按（名称, 城市）回填
    coord_index = {}
    for _, mp in snaps:
        for name, city, lon, lat in mp.values():
            if lon is not None and city:
                coord_index[(name, city)] = (lon, lat)
    for eff, mp in snaps:
        fixed = {}
        for code, (name, city, lon, lat) in mp.items():
            if lon is None:
                hit = coord_index.get((name, city)) or coord_index.get((name, ""))
                if hit:
                    lon, lat = hit
            fixed[code] = (name, city, lon, lat)
        snaps[snaps.index((eff, mp))] = (eff, fixed)
    return snaps


def _read_standard(path: Path):
    mapping = {}
    if path.suffix.lower() == ".xlsx":
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = list(wb[wb.sheetnames[0]].iter_rows(values_only=True))
        header = [str(c).strip() if c is not None else "" for c in rows[0]]
        idx = {h: i for i, h in enumerate(header)}
        body = rows[1:]
    else:
        with path.open(encoding="utf-8-sig") as fh:
            rd = csv.reader(fh)
            header = [h.strip() for h in next(rd)]
            idx = {h: i for i, h in enumerate(header)}
            body = list(rd)

    def cell(r, key):
        i = idx.get(key)
        if i is None or i >= len(r) or r[i] is None:
            return ""
        return str(r[i]).strip()

    for r in body:
        code = cell(r, "监测点编码")
        if not code or code == "监测点编码":
            continue
        try:
            lon = float(cell(r, "经度"))
            lat = float(cell(r, "纬度"))
        except ValueError:
            lon = lat = None
        mapping[code] = (cell(r, "监测点名称"), cell(r, "城市"), lon, lat)
    return mapping


def _read_transposed(path: Path):
    """旧版转置表：第 1 行编码、第 2 行名称、第 3 行城市，无经纬度。"""
    with path.open(encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if len(rows) < 3:
        return {}
    codes, names, cities = rows[0], rows[1], rows[2]
    mapping = {}
    for c, n, city in zip(codes, names, cities):
        c = c.strip()
        if c:
            mapping[c] = (n.strip(), city.strip(), None, None)
    return mapping


def snapshot_for(day: str, snaps):
    chosen = snaps[0]
    for eff, mapping in snaps:
        if eff <= day:
            chosen = (eff, mapping)
        else:
            break
    return chosen


def melt_sites(path: Path, codes: list[str]) -> pd.DataFrame:
    from build_long import _is_html_junk

    if _is_html_junk(path):
        return pd.DataFrame()
    head = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    usecols = [c for c in ["hour", "type"] + list(codes) if c in head]
    if len(usecols) < 3:
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=usecols)
    if df.empty:
        return pd.DataFrame()
    df["type"] = df["type"].astype(str).str.strip()
    df = df[df["type"].isin(TYPE_MAP)]
    if df.empty:
        return pd.DataFrame()
    long = df.melt(id_vars=["hour", "type"], var_name="station_code", value_name="raw")
    long["hour"] = pd.to_numeric(long["hour"], errors="coerce").astype("Int8")
    long = long[long["hour"].between(0, 23)]
    long["raw"] = long["raw"].astype(str).str.strip()
    miss = long["raw"].isin(["", "nan", "None", "-", "N/A", "NA", "--"])
    long["value"] = pd.to_numeric(long["raw"].where(~miss), errors="coerce")
    wide = long.pivot_table(
        index=["hour", "station_code"], columns="type", values="value", aggfunc="first", dropna=False
    ).reset_index()
    wide.columns.name = None
    return wide.rename(columns=TYPE_MAP)


def main():
    snaps = load_snapshots()
    print(f"站点表版本 {len(snaps)} 个：{snaps[0][0]} ~ {snaps[-1][0]}")

    raw = REPO / "data" / "raw" / "china_sites"
    files = sorted(raw.glob("*.csv"))
    print(f"站点数据 {len(files)} 天")
    if not files:
        raise SystemExit("未找到站点数据，先运行 extract_zips.py --kind china_sites")

    out = REPO / "data" / "warehouse" / "fact_obs" / "source=china_sites"
    out.mkdir(parents=True, exist_ok=True)

    # 先算出所有版本中属于京津冀的编码全集，作为读列白名单
    all_codes = set()
    for _, mapping in snaps:
        for code, (_, city, _, _) in mapping.items():
            if city in JINGJINJI_CITIES:
                all_codes.add(code)
    print(f"京津冀编码全集 {len(all_codes)} 个")

    frames, meta_rows = [], []
    for i, f in enumerate(files, 1):
        day = f.stem
        eff, mapping = snapshot_for(day, snaps)
        codes = [c for c in all_codes if c in mapping]
        wide = melt_sites(f, codes)
        if wide.empty:
            continue
        wide["name_cn"] = wide["station_code"].map(lambda c: mapping.get(c, ("", "", None, None))[0])
        wide["city"] = wide["station_code"].map(lambda c: mapping.get(c, ("", "", None, None))[1])
        wide["station_list_version"] = eff
        frames.append(finalize(wide.rename(columns={"station_code": "name_cn_raw"}), day, "china_sites"))
        if i % 300 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    if not frames:
        raise SystemExit("没有可用数据")
    df = pd.concat(frames, ignore_index=True)
    # name_cn 用站点中文名，编码单独保留，避免同名站冲突
    df["station_code"] = df["name_cn_raw"]
    df["year"] = df["hour_start_local"].dt.year

    for old in out.glob("year=*"):
        for p in old.glob("*.parquet"):
            p.unlink()
    total = 0
    for year, part in df.groupby("year"):
        d = out / f"year={year}"
        d.mkdir(parents=True, exist_ok=True)
        part.drop(columns=["year"]).to_parquet(d / "part.parquet", index=False, compression="zstd")
        total += len(part)
        print(f"  year={year} 行数 {len(part):>9,}", flush=True)

    print(f"\n[china_sites] 写出 {total:,} 行 → {out}")
    print(f"  城市 {sorted(df['city'].dropna().unique())}")
    print(f"  站点 {df['station_code'].nunique()} 个 | 状态 {df['value_status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
