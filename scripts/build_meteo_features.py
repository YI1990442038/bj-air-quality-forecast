#!/usr/bin/env python3
"""把 ERA5 气象特征并入特征表（等 dim_meteo 就绪后运行）。

两种取数时刻，必须在报告里区分清楚：
- `met_*_lag1`：T-1 时刻的气象，发布时点**确实已知**，严格无泄漏
- `met_*_tgt` ：目标小时的气象，现实中需要依赖天气预报，属于**假设气象预报可用**

把两者都做出来，是为了后续能做消融：如果只用 lag1 就够，结论更稳；
如果 tgt 带来大幅提升，就要在报告里明确声明「假设预报可得」这一前提。

风向用 sin/cos 编码，避免 359°→0° 的假跳变。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aircast import protocol as P  # noqa: E402

FEATDIR = REPO / "data" / "features" / f"protocol={P.PROTOCOL_ID}"
METEODIR = REPO / "data" / "warehouse" / "dim_meteo"
MAPFILE = REPO / "data" / "meta" / "station_meteo_map.csv"

# 变量 → 输出列名前缀
VARS = ["temp", "rh", "wspd", "wdir", "blh", "pres", "ssrd", "vpd", "rain"]
# 只有这几个做 T-1 版本，其余只取目标小时，控制特征数量
LAG1_VARS = ["temp", "wspd", "blh"]


def load_meteo() -> dict[str, pd.DataFrame]:
    """返回 {point_id: 以 ts_local 为索引的宽表}，避免逐站重复切片。"""
    fs = sorted(METEODIR.glob("point_id=*/part.parquet"))
    if not fs:
        raise SystemExit("dim_meteo 为空，先跑 scripts/build_meteo.py")
    out = {}
    for f in fs:
        pid = f.parent.name.split("=", 1)[1]
        cols = ["ts_local"] + [v for v in VARS if v in pd.read_parquet(f).columns]
        d = pd.read_parquet(f, columns=cols).drop_duplicates("ts_local")
        rad = np.radians(d["wdir"].astype("float64"))
        d["wdir_sin"] = np.sin(rad)
        d["wdir_cos"] = np.cos(rad)
        out[pid] = d.set_index("ts_local").sort_index()
    return out


def main():
    if not MAPFILE.exists():
        raise SystemExit(f"缺少站点-格点映射 {MAPFILE}")
    mp = pd.read_csv(MAPFILE)
    met = load_meteo()
    all_ts = pd.DatetimeIndex(sorted(set().union(*[d.index for d in met.values()])))
    print(f"气象格点 {len(met)} 个，时段 {all_ts.min()} ~ {all_ts.max()}")

    out_vars = [v for v in VARS if v != "wdir"] + ["wdir_sin", "wdir_cos"]
    added = []
    for h in P.HORIZONS:
        f = FEATDIR / f"h={h}" / "part.parquet"
        df = pd.read_parquet(f)
        times = pd.DatetimeIndex(sorted(df["issue_time"].unique()))
        n_t = len(times)
        n_s = len(df) // n_t
        stations = df["station_name"].to_numpy()[:n_s]
        assert (df["station_name"].to_numpy() == np.tile(stations, n_t)).all()

        st2pt = dict(zip(mp["name_cn"], mp["meteo_point_id"]))
        tgt_times = times + pd.Timedelta(hours=h - 1)
        lag_times = times - pd.Timedelta(hours=1)

        def build(var: str, when: pd.DatetimeIndex) -> np.ndarray:
            mat = np.full((n_t, n_s), np.nan, dtype="float32")
            for j, st in enumerate(stations):
                pid = st2pt.get(st)
                if pid is None or pid not in met or var not in met[pid].columns:
                    continue
                mat[:, j] = met[pid][var].reindex(when).to_numpy(dtype="float32")
            return mat.ravel(order="C")

        new_cols = {f"met_{v}_tgt": build(v, tgt_times) for v in out_vars}
        for v in LAG1_VARS:
            new_cols[f"met_{v}_lag1"] = build(v, lag_times)

        df = df.drop(columns=[c for c in df.columns if c.startswith("met_")])
        for k, arr in new_cols.items():
            df[k] = arr
        df.to_parquet(f, index=False, compression="zstd")
        if not added:
            added = list(new_cols)
        cov = float(np.isfinite(df["met_temp_tgt"]).mean())
        print(f"  h={h} 并入 {len(new_cols)} 个气象特征，可用率 {cov:.1%}")

    print(f"\n新增特征 {len(added)} 个：{added}")
    print("注意：met_*_tgt 依赖目标小时气象，报告中须声明「假设气象预报可用」。")


if __name__ == "__main__":
    main()
