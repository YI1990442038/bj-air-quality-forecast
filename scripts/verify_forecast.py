#!/usr/bin/env python3
"""预报对账：把已发布的预测与随后到来的真值逐条比对。

真值直接读原始日文件，不走仓库、不走索引——这是独立第三方口径，
避免"用自己的管道验自己的结果"。

用法
  python scripts/verify_forecast.py --pred data/predictions/live_20261031_2300.parquet
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aircast import protocol as P  # noqa: E402

RAW = REPO / "data" / "raw" / "beijing_all"
_cache: dict[str, dict] = {}


def day_table(day: str) -> dict:
    """读某日原始文件的 PM2.5，返回 {(station, hour): value}。"""
    if day in _cache:
        return _cache[day]
    path = RAW / f"{day}.csv"
    out = {}
    if path.exists():
        with path.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row["type"] != "PM2.5":
                    continue
                h = int(row["hour"])
                for k, v in row.items():
                    if k in ("date", "hour", "type"):
                        continue
                    v = (v or "").strip()
                    if v not in ("", "-"):
                        try:
                            out[(k, h)] = float(v)
                        except ValueError:
                            pass
    _cache[day] = out
    return out


def value_at(station: str, ts: pd.Timestamp):
    return day_table(f"{ts:%Y%m%d}").get((station, ts.hour))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.pred)
    T = df["issue_time"].iloc[0]
    print(f"预测文件 {Path(args.pred).name}")
    print(f"发布时点 T = {T:%Y-%m-%d %H:%M}　共 {len(df)} 条\n")

    rows = []
    for _, r in df.iterrows():
        ts = pd.Timestamp(r["target_start"])
        rows.append({
            "h": int(r["h"]), "station_name": r["station_name"], "target": ts,
            "y_pred": float(r["y_pred"]),
            "y_true": value_at(r["station_name"], ts),
            "b1": value_at(r["station_name"], T - pd.Timedelta(hours=1)),
        })
    v = pd.DataFrame(rows)

    print(f"{'h':>2} {'目标时刻':<13}{'可对账':>7}{'缺失':>6}{'MAE':>8}{'B1 MAE':>9}{'优于B1':>8}{'RMSE':>8}")
    print("-" * 68)
    summary = []
    for h in P.HORIZONS:
        s = v[v["h"] == h]
        ok = s.dropna(subset=["y_true", "y_pred"])
        miss = len(s) - len(ok)
        mae = float((ok["y_pred"] - ok["y_true"]).abs().mean()) if len(ok) else np.nan
        rmse = float(np.sqrt(((ok["y_pred"] - ok["y_true"]) ** 2).mean())) if len(ok) else np.nan
        okb = ok.dropna(subset=["b1"])
        b1 = float((okb["b1"] - okb["y_true"]).abs().mean()) if len(okb) else np.nan
        better = (okb["y_pred"] - okb["y_true"]).abs() < (okb["b1"] - okb["y_true"]).abs()
        tgt = s["target"].iloc[0]
        print(f"{h:>2} {tgt:%m-%d %H:%M}  {len(ok):>7}{miss:>6}{mae:>8.2f}{b1:>9.2f}"
              f"{better.mean():>8.1%}{rmse:>8.2f}")
        summary.append({"h": h, "n": len(ok), "MAE": round(mae, 3), "B1_MAE": round(b1, 3),
                        "win_rate": round(float(better.mean()), 4)})

    ok = v.dropna(subset=["y_true", "y_pred"])
    if len(ok):
        err = (ok["y_pred"] - ok["y_true"])
        print(f"\n整体：{len(ok)} 条可对账，MAE {err.abs().mean():.2f} μg/m³，"
              f"平均偏差 {err.mean():+.2f}（正=高估）")
        print(f"  最大高估 {err.max():+.1f}，最大低估 {err.min():+.1f}")

        worst = ok.reindex(err.abs().sort_values(ascending=False).index).head(5)
        print("\n误差最大的 5 条：")
        for _, r in worst.iterrows():
            print(f"  h={int(r['h'])} {r['station_name']:<10} {r['target']:%m-%d %H:%M}  "
                  f"预测 {r['y_pred']:6.1f}  实况 {r['y_true']:6.1f}  偏差 {r['y_pred']-r['y_true']:+6.1f}")

    d = REPO / "data" / "predictions"
    v.to_csv(d / f"verify_{T:%Y%m%d_%H%M}.csv", index=False, encoding="utf-8")
    print(f"\n明细已存 {d / f'verify_{T:%Y%m%d_%H%M}.csv'}")


if __name__ == "__main__":
    main()
