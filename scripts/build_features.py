#!/usr/bin/env python3
"""特征层 V1（不含气象）。

设计原则
1. 一切特征只能用到 [T-1, T) 及更早的观测。取数时刻全部由 protocol 反推，不手写偏移。
2. 邻域特征用留一法均值，避免把本站自己的值混进「区域背景」。
3. 时间编码用**目标小时**的时钟，不是发布时点的——污染有日循环，h=6 的目标在 15 点而非 10 点。
4. 气象暂缺（Open-Meteo 日额度耗尽），留出接口，补采后加 V2 做消融。

输出
  data/features/protocol=p1/h={h}/part.parquet
  data/features/protocol=p1/feature_list.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aircast import protocol as P  # noqa: E402

EXTRA_COLS = ["so2", "no2", "o3", "co"]  # PM10 在 beijing_all 里，不在 extra


def _wide(parquet_dir: Path, value_col: str, era: str | None = None) -> pd.DataFrame:
    fs = sorted(parquet_dir.glob("year=*/part.parquet"))
    if not fs:
        raise SystemExit(f"缺少 {parquet_dir}")
    cols = ["name_cn", "network_era", "hour_start_local", value_col]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    if era:
        df = df[df["network_era"] == era]
    df = df.drop_duplicates(["name_cn", "hour_start_local"], keep="first")
    return df.pivot(index="hour_start_local", columns="name_cn", values=value_col)


def build_era_b(repo: Path):
    """返回 (pm25 矩阵, pm10 矩阵, 其他污染物 {name: 矩阵}, 城市矩阵)。

    PM10 与 PM2.5 同在 beijing_all；SO2/NO2/O3/CO 在 beijing_extra。
    """
    base = repo / "data" / "warehouse" / "fact_obs"
    pm = _wide(base / "source=beijing_all", "pm25", P.ERA)
    grid = pd.date_range(pm.index.min(), pm.index.max(), freq="h")
    pm = pm.reindex(grid)

    try:
        pm10 = _wide(base / "source=beijing_all", "pm10", P.ERA).reindex(index=grid, columns=pm.columns)
    except Exception:
        pm10 = pd.DataFrame(np.nan, index=grid, columns=pm.columns)

    extra = {}
    for c in EXTRA_COLS:
        try:
            m = _wide(base / "source=beijing_extra", c, P.ERA)
        except Exception:
            continue
        extra[c] = m.reindex(index=grid, columns=pm.columns)

    cities = _wide(base / "source=china_cities", "pm25")
    cities = cities.reindex(grid)
    return pm, pm10, extra, cities


def make_features(pm: pd.DataFrame, pm10: pd.DataFrame, extra: dict, cities: pd.DataFrame) -> dict:
    """返回 {特征名: (时间 × 站点) 矩阵 或 一维数组}，全部对齐到发布时点 T。"""
    f: dict[str, pd.DataFrame] = {}

    # 本站滞后：y(T-1), y(T-2) …
    for k in (1, 2, 3, 6, 12, 24, 25, 48):
        f[f"pm25_lag{k}"] = pm.shift(k)

    # 本站滚动统计，窗口以 T-1 结束
    for w in (6, 24, 72):
        f[f"pm25_roll{w}_mean"] = pm.rolling(w, min_periods=max(2, w // 3)).mean().shift(1)
    f["pm25_roll24_std"] = pm.rolling(24, min_periods=8).std().shift(1)
    f["pm25_roll72_max"] = pm.rolling(72, min_periods=24).max().shift(1)

    # 变化量
    f["pm25_diff1"] = pm.shift(1) - pm.shift(2)
    f["pm25_diff3"] = pm.shift(1) - pm.shift(4)
    f["pm25_delta24"] = pm.shift(1) - pm.shift(25)
    f["pm25_trend"] = pm.shift(1).rolling(3, min_periods=2).mean() - pm.shift(4).rolling(
        3, min_periods=2
    ).mean()

    # 同站其他污染物（同一小时 T-1）
    for c, m in extra.items():
        f[f"{c}_lag1"] = m.shift(1)
    f["pm10_lag1"] = pm10.shift(1)
    f["pm10_pm25_ratio"] = (pm10.shift(1) / pm.shift(1).replace(0, np.nan)).clip(0, 20)

    # 邻域：北京其余站点的留一法均值（区域背景）
    lag1 = pm.shift(1)
    total = lag1.sum(axis=1, min_count=1)
    count = lag1.notna().sum(axis=1)
    loo = (total.values[:, None] - lag1.values) / (count.values[:, None] - 1)
    f["bj_bg_loo"] = pd.DataFrame(loo, index=pm.index, columns=pm.columns)

    # 邻域：京津冀 13 城均值与其 24 小时前值（区域输送）
    f["region_city_mean"] = pd.concat([cities.mean(axis=1)], axis=1).reindex(pm.index).values
    f["region_city_mean_lag6"] = cities.mean(axis=1).shift(6).reindex(pm.index).values
    f["region_city_mean_lag24"] = cities.mean(axis=1).shift(24).reindex(pm.index).values

    return f


def attach_time_encoding(df: pd.DataFrame):
    """时间编码取自目标小时，而非发布时点。"""
    tgt = df["target_start"]
    hour = tgt.dt.hour
    df["tgt_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["tgt_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    df["tgt_dow"] = tgt.dt.dayofweek.astype("int8")
    df["tgt_is_weekend"] = (tgt.dt.dayofweek >= 5).astype("int8")
    mo = tgt.dt.month
    df["tgt_month_sin"] = np.sin(2 * np.pi * mo / 12).astype("float32")
    df["tgt_month_cos"] = np.cos(2 * np.pi * mo / 12).astype("float32")
    df["issue_hour"] = df["issue_time"].dt.hour.astype("int8")
    return df


def main():
    pm, pm10, extra, cities = build_era_b(REPO)
    print(f"pm25 矩阵 {pm.shape[0]:,} 小时 × {pm.shape[1]} 站；其他污染物 {list(extra)}")

    feats = make_features(pm, pm10, extra, cities)
    print(f"构建 {len(feats)} 组基础特征")

    outdir = REPO / "data" / "features" / f"protocol={P.PROTOCOL_ID}"
    outdir.mkdir(parents=True, exist_ok=True)
    feature_names = None
    meta_cols = ["issue_time", "station_name", "h", "target_start", "target_end",
                 "split", "y_true", "b1_persistence", "b2_yesterday"]

    for h in P.HORIZONS:
        idx = pd.read_parquet(
            REPO / "data" / "index" / f"protocol={P.PROTOCOL_ID}" / f"h={h}" / "part.parquet"
        )
        n_s = pm.shape[1]
        stations = pm.columns.to_numpy()
        times = pd.DatetimeIndex(sorted(idx["issue_time"].unique()))
        n_t = len(times)

        # 索引是「时间主序、站点次序」铺排的。这里显式断言，避免按位赋值错位。
        assert len(idx) == n_t * n_s, f"行数 {len(idx)} != {n_t}×{n_s}"
        assert (idx["issue_time"].to_numpy() == np.repeat(times.to_numpy(), n_s)).all(), "时间铺排不符"
        assert (idx["station_name"].to_numpy() == np.tile(stations, n_t)).all(), "站点次序不符"

        out = idx[meta_cols].copy()
        for name, mat in feats.items():
            if isinstance(mat, pd.DataFrame):
                vals = mat.reindex(index=times, columns=pm.columns).to_numpy(dtype="float32").ravel(order="C")
            else:
                # 区域级一维特征：先按时间取值，再为每个时间广播到 35 个站点
                arr = pd.Series(np.asarray(mat, dtype="float32").ravel(), index=pm.index)
                per_time = arr.reindex(times).to_numpy(dtype="float32")
                vals = np.repeat(per_time, n_s)
            assert len(vals) == len(out), f"{name} 长度 {len(vals)} != {len(out)}"
            out[name] = vals

        out = attach_time_encoding(out)
        d = outdir / f"h={h}"
        d.mkdir(parents=True, exist_ok=True)
        out.to_parquet(d / "part.parquet", index=False, compression="zstd")
        if feature_names is None:
            feature_names = [c for c in out.columns if c not in meta_cols]
        print(f"  h={h} 写出 {len(out):,} 行 × {len(out.columns)} 列")

    (outdir / "feature_list.json").write_text(
        json.dumps({"protocol_id": P.PROTOCOL_ID, "n_features": len(feature_names),
                    "features": feature_names, "meteo_included": False}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n特征 {len(feature_names)} 个 → {outdir / 'feature_list.json'}")
    print(feature_names)


if __name__ == "__main__":
    main()
