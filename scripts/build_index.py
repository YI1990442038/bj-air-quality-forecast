#!/usr/bin/env python3
"""构建固定评价索引与两套基线。

产出一张「谁在什么时候、预测哪一小时、真值多少、两个基线各给多少」的表。
这张表一旦生成就冻结：所有模型共用同一批 (station, issue_time, h) 行，
不得各自过滤样本，否则指标不可比。

输出
  data/index/protocol=p1/h={h}/part.parquet
  data/index/protocol=p1/summary.csv
  data/index/protocol=p1/fingerprint.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aircast import protocol as P  # noqa: E402


def load_pm25(repo: Path) -> pd.DataFrame:
    fs = sorted((repo / "data" / "warehouse" / "fact_obs" / "source=beijing_all").glob("year=*/part.parquet"))
    if not fs:
        raise SystemExit("缺少 beijing_all 仓库表")
    df = pd.concat(
        [pd.read_parquet(f, columns=["name_cn", "network_era", "hour_start_local", "pm25"]) for f in fs],
        ignore_index=True,
    )
    df = df[df["network_era"] == P.ERA]
    dup = df.duplicated(["name_cn", "hour_start_local"]).sum()
    if dup:
        print(f"  ! 发现 {dup} 条重复的 (站点, 小时)，按首条保留")
        df = df.drop_duplicates(["name_cn", "hour_start_local"], keep="first")
    return df


def station_coverage(df: pd.DataFrame) -> pd.Series:
    """每站在整个纪元内的 pm25 可用率。"""
    return df.groupby("name_cn")["pm25"].apply(lambda s: float(s.notna().mean()))


def build(repo: Path):
    df = load_pm25(repo)
    cov = station_coverage(df)
    keep = cov[cov >= P.MIN_STATION_COVERAGE].index
    dropped = sorted(set(cov.index) - set(keep))
    print(f"站点 {len(cov)} 个，可用率≥{P.MIN_STATION_COVERAGE:.0%} 的 {len(keep)} 个")
    if dropped:
        print(f"  剔除低可用率站点：{dropped}")

    df = df[df["name_cn"].isin(keep)]
    full = pd.date_range(df["hour_start_local"].min(), df["hour_start_local"].max(), freq="h")
    wide = df.pivot(index="hour_start_local", columns="name_cn", values="pm25").reindex(full)
    print(f"  时间轴 {full[0]} ~ {full[-1]} 共 {len(full):,} 小时 × {wide.shape[1]} 站")

    # 索引只在协议区间内产生；warmup 之前的发布时点不进入评价
    issue_times = full[full >= pd.Timestamp(P.SPLITS["warmup"][1])]
    off_target = {h: (P.target_hour(issue_times[0], h) - issue_times[0]).total_seconds() / 3600 for h in P.HORIZONS}
    off_yest = {h: (P.yesterday_hour(issue_times[0], h) - issue_times[0]).total_seconds() / 3600 for h in P.HORIZONS}
    for h in P.HORIZONS:
        assert off_target[h] == h - 1, off_target
        assert off_yest[h] == h - 25, off_yest
    print(f"  偏移自检通过：目标 +{off_target}，昨日同期 {off_yest}")

    outdir = repo / "data" / "index" / f"protocol={P.PROTOCOL_ID}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []

    for h in P.HORIZONS:
        off_t = int(off_target[h])
        off_y = int(off_yest[h])
        y_true = wide.shift(-off_t).reindex(issue_times)
        b1 = wide.shift(1).reindex(issue_times)
        b2 = wide.shift(-off_y).reindex(issue_times)

        n_t, n_s = y_true.shape
        stations = wide.columns.to_numpy()
        splits = np.array([P.split_of(t) for t in issue_times], dtype=object)

        # 三张矩阵都按行主序展平：先时间后站点，与 station/issue_time 的铺设方式一致。
        frame = pd.DataFrame(
            {
                "issue_time": np.repeat(issue_times.to_numpy(), n_s),
                "station_name": np.tile(stations, n_t),
                "y_true": y_true.to_numpy(dtype="float32").ravel(order="C"),
                "b1_persistence": b1.to_numpy(dtype="float32").ravel(order="C"),
                "b2_yesterday": b2.to_numpy(dtype="float32").ravel(order="C"),
                "split": np.repeat(splits, n_s),
            }
        )
        frame["h"] = np.int8(h)
        frame["target_start"] = frame["issue_time"] + pd.Timedelta(hours=h - 1)
        frame["target_end"] = frame["target_start"] + pd.Timedelta(hours=1)
        frame = frame[frame["split"].notna()].copy()
        frame["split"] = frame["split"].astype("category")
        frame["station_name"] = frame["station_name"].astype("category")

        # 全量断言（不是抽样）：目标窗口必须严格等于契约定义，且长度恒为 1 小时
        assert (frame["target_start"] - frame["issue_time"] == pd.Timedelta(hours=h - 1)).all()
        assert (frame["target_end"] - frame["target_start"] == pd.Timedelta(hours=1)).all()
        for ts in frame["issue_time"].iloc[[0, len(frame) // 2, -1]]:
            s, e = P.target_window(ts, h)
            assert s == ts + pd.Timedelta(hours=h - 1) and e - s == pd.Timedelta(hours=1)

        d = outdir / f"h={h}"
        d.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(d / "part.parquet", index=False, compression="zstd")

        for sp in ("train", "val", "test"):
            sub = frame[frame["split"] == sp]
            if sub.empty:
                continue
            ok = sub.dropna(subset=["y_true"])
            summary.append(
                {
                    "h": h,
                    "split": sp,
                    "rows": len(sub),
                    "y_true_available": len(ok),
                    "coverage": round(len(ok) / len(sub), 4),
                    "B1_MAE": round(float((ok["b1_persistence"] - ok["y_true"]).abs().mean()), 3)
                    if len(ok.dropna(subset=["b1_persistence"]))
                    else None,
                    "B2_MAE": round(float((ok["b2_yesterday"] - ok["y_true"]).abs().mean()), 3)
                    if len(ok.dropna(subset=["b2_yesterday"]))
                    else None,
                    "B1_n": int(len(ok.dropna(subset=["b1_persistence"]))),
                    "B2_n": int(len(ok.dropna(subset=["b2_yesterday"]))),
                }
            )
        print(f"  h={h} 写出 {len(frame):,} 行 → {d}")

    sm = pd.DataFrame(summary)
    sm.to_csv(outdir / "summary.csv", index=False, encoding="utf-8")

    fp = {
        "protocol_id": P.PROTOCOL_ID,
        "era": P.ERA,
        "horizons": list(P.HORIZONS),
        "splits": P.SPLITS,
        "stations": sorted(keep.tolist()),
        "n_stations": len(keep),
        "issue_time_min": str(issue_times[0]),
        "issue_time_max": str(issue_times[-1]),
        "n_issue_times": len(issue_times),
        "min_station_coverage": P.MIN_STATION_COVERAGE,
    }
    blob = json.dumps(fp, ensure_ascii=False, sort_keys=True).encode("utf-8")
    fp["fingerprint_sha256"] = hashlib.sha256(blob).hexdigest()
    (outdir / "fingerprint.json").write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n指纹 {fp['fingerprint_sha256'][:16]}…  站点 {len(keep)}  发布时点 {len(issue_times):,}")
    print(sm.to_string(index=False))
    return sm


if __name__ == "__main__":
    build(REPO)
