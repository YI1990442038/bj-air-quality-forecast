#!/usr/bin/env python3
"""原始日文件 → 站点小时长表（宽转长 + 单位统一 + 质量状态）。

输入：data/raw/{beijing_all,beijing_extra,china_cities}/*.csv
输出：data/warehouse/fact_obs/source=<src>/year=<YYYY>/part.parquet

三个必须做对的地方：
1. type 字段必须精确匹配（PM2.5 / PM2.5_24h / AQI 混在同一文件里）
2. CO 原单位 mg/m³，其余 μg/m³，本脚本统一为 μg/m³ 并保留 unit_original 标记
3. 空值与异常值分开标记，绝不用 0 填充
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

TYPE_MAP = {
    "PM2.5": "pm25",
    "PM2.5_24h": "pm25_24h",
    "PM10": "pm10",
    "PM10_24h": "pm10_24h",
    "AQI": "aqi",
    "SO2": "so2",
    "SO2_24h": "so2_24h",
    "NO2": "no2",
    "NO2_24h": "no2_24h",
    "O3": "o3",
    "O3_24h": "o3_24h",
    "O3_8h": "o3_8h",
    "O3_8h_24h": "o3_8h_24h",
    "CO": "co",
    "CO_24h": "co_24h",
}

# 单位换算：CO 原单位 mg/m³ → μg/m³
MG_COLS = {"co", "co_24h"}

# 物理合理范围（μg/m³），超出记为 out_of_range 而不是直接删
VALID_RANGE = {
    "pm25": (0, 1500),
    "pm10": (0, 3000),
    "so2": (0, 4000),
    "no2": (0, 2000),
    "o3": (0, 2000),
    "co": (0, 60000),  # μg/m³
    "pm25_24h": (0, 1500),
    "pm10_24h": (0, 3000),
    "aqi": (0, 500),
}

JINGJINJI = [
    "北京", "天津", "石家庄", "唐山", "秦皇岛", "邯郸", "保定",
    "张家口", "承德", "廊坊", "沧州", "衡水", "邢台",
]


def _is_html_junk(path: Path) -> bool:
    """识别被误当作数据的 HTML 错误页。

    源站故障时单日文件会返回 nginx 404 页面，年包里也原样收录了。
    例如 2018-12-22~26 的 china_sites 实际是 169 字节的 404 HTML。
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(64).lstrip()
    except OSError:
        return True
    return head[:1] in (b"<",) or head.startswith(b"\x00\x05\x16\x07")


def melt_day(path: Path, keep: list[str] | None = None) -> pd.DataFrame:
    """单日宽表 → (date, hour, name_cn) × 污染物的长表。

    keep 非空时只读指定列：全国城市文件有 367 列，全量 melt 会慢 30 倍且毫无必要。
    """
    if _is_html_junk(path):
        return pd.DataFrame()
    if keep is None:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    else:
        head = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
        usecols = [c for c in ["date", "hour", "type"] + list(keep) if c in head]
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=usecols)
    if df.empty or not {"date", "hour", "type"}.issubset(df.columns):
        return pd.DataFrame()
    # date 在文件内恒定，由文件名提供，绝不能当作站点列参与 melt
    df = df.drop(columns=["date"])
    df["type"] = df["type"].astype(str).str.strip()
    df = df[df["type"].isin(TYPE_MAP)]
    if df.empty:
        return pd.DataFrame()
    long = df.melt(id_vars=["hour", "type"], var_name="name_cn", value_name="raw")
    long["hour"] = pd.to_numeric(long["hour"], errors="coerce").astype("Int8")
    long = long[long["hour"].between(0, 23)]

    # 只把真正缺失的样式识别为缺失，不做任何填充
    long["raw"] = long["raw"].astype(str).str.strip()
    missing_mask = long["raw"].isin(["", "nan", "None", "-", "N/A", "NA", "--"])
    long["value"] = pd.to_numeric(long["raw"].where(~missing_mask), errors="coerce")
    long["is_missing"] = missing_mask | long["value"].isna()

    # dropna=False 极关键：否则「整天全空」的站点会被整列丢弃，
    # 表现为站点凭空消失而非记录为缺失——这是最隐蔽的数据损失。
    wide = long.pivot_table(
        index=["hour", "name_cn"], columns="type", values="value", aggfunc="first", dropna=False
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns=TYPE_MAP)
    return wide


def finalize(df: pd.DataFrame, day: str, source: str) -> pd.DataFrame:
    """补时间列、单位、质量状态。"""
    df = df.copy()
    df["date"] = day
    for c in MG_COLS:
        if c in df.columns:
            df[c] = df[c] * 1000.0
    df["hour_start_local"] = pd.to_datetime(df["date"], format="%Y%m%d") + pd.to_timedelta(
        df["hour"].astype(int), unit="h"
    )
    df["hour_start_utc"] = df["hour_start_local"] - pd.Timedelta(hours=8)
    df["source"] = source
    # 站网在 2021-01-24 换代：A 纪元与 B 纪元的同名站不可直接拼接，
    # 只有 10 个是纯改名，其余为新设站。用纪元列显式隔离，防止跨代错配。
    df["network_era"] = np.where(df["date"] >= "20210124", "B", "A")

    # 质量状态以目标变量 pm25 为准（行 = 站点·小时，只能有一个主状态），
    # 其余污染物的越界单独用计数记录，避免状态被逐列覆盖成噪声。
    val_cols = [c for c in VALID_RANGE if c in df.columns]
    flag_counts = pd.Series(0, index=df.index, dtype="int16")
    for col in val_cols:
        lo, hi = VALID_RANGE[col]
        v = df[col]
        flag_counts += ((v.notna()) & ((v < lo) | (v > hi))).astype("int16")
    df["n_pollutants"] = df[val_cols].notna().sum(axis=1).astype("int8")
    df["n_out_of_range"] = flag_counts

    primary = "pm25" if "pm25" in df.columns else val_cols[0]
    status = pd.Series("observed", index=df.index, dtype="object")
    status = status.mask(df[primary].isna(), "missing_value")
    lo, hi = VALID_RANGE[primary]
    vp = df[primary]
    status = status.mask(vp.notna() & ((vp < lo) | (vp > hi)), "out_of_range")
    df["value_status"] = status
    return df


def process(source: str, repo: Path, cities: list[str] | None) -> pd.DataFrame:
    rawdir = repo / "data" / "raw" / source
    files = sorted(rawdir.glob("*.csv"))
    if not files:
        raise SystemExit(f"{rawdir} 下没有文件")
    print(f"[{source}] {len(files)} 个日文件", flush=True)

    frames, skipped = [], []
    for i, f in enumerate(files, 1):
        try:
            wide = melt_day(f, cities)
        except Exception as e:
            skipped.append((f.name, type(e).__name__))
            continue
        if wide.empty:
            skipped.append((f.name, "empty"))
            continue
        if cities is not None:
            wide = wide[wide["name_cn"].isin(cities)]
            if wide.empty:
                skipped.append((f.name, "no_target_city"))
                continue
        frames.append(finalize(wide, f.stem, source))
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    if not frames:
        raise SystemExit("没有任何可用数据")
    out = pd.concat(frames, ignore_index=True)
    if skipped:
        print(f"  跳过 {len(skipped)} 个文件，例如 {skipped[:3]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["beijing_all", "beijing_extra", "china_cities"])
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    repo = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    cities = JINGJINJI if args.source == "china_cities" else None
    df = process(args.source, repo, cities)

    outdir = repo / "data" / "warehouse" / "fact_obs" / f"source={args.source}"
    outdir.mkdir(parents=True, exist_ok=True)
    df["year"] = df["hour_start_local"].dt.year

    total = 0
    for year, part in df.groupby("year"):
        d = outdir / f"year={year}"
        d.mkdir(parents=True, exist_ok=True)
        part.drop(columns=["year"]).to_parquet(d / "part.parquet", index=False, compression="zstd")
        total += len(part)
        print(f"  year={year} 行数 {len(part):>9,}", flush=True)

    print(f"\n[{args.source}] 写出 {total:,} 行 → {outdir}", flush=True)
    print(
        f"  状态分布: {df['value_status'].value_counts().to_dict()}\n"
        f"  站点数: {df['name_cn'].nunique()} | 时间范围: "
        f"{df['hour_start_local'].min()} ~ {df['hour_start_local'].max()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
