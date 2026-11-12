#!/usr/bin/env python3
"""特征消融实验：逐个拿掉特征组，看各自的边际贡献。

对报告的意义：证明「加进来的数据/特征确实有用」，而不是模型自带的能力。

用法
  python scripts/experiment_ablation.py --h 6 --sets full,no_neighbor,no_pollutant,lag_only,no_time
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aircast import metrics as M  # noqa: E402
from aircast import protocol as P  # noqa: E402

FEATDIR = REPO / "data" / "features" / f"protocol={P.PROTOCOL_ID}"
META = {"issue_time", "station_name", "h", "target_start", "target_end", "split",
        "y_true", "b1_persistence", "b2_yesterday"}
PARAMS = dict(objective="regression_l1", metric="mae", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, num_threads=14)
HEAVY = 150.0

NEIGHBOR = ["bj_bg_loo", "region_city_mean", "region_city_mean_lag6", "region_city_mean_lag24"]
GRAPH = ["nb5_lag1", "nbinv_lag1", "nb5_lag24", "nbinv_lag24", "nb5_minus_bg"]
POLLUTANT = ["so2_lag1", "no2_lag1", "o3_lag1", "co_lag1", "pm10_lag1", "pm10_pm25_ratio"]
LAGS = [f"pm25_lag{k}" for k in (1, 2, 3, 6, 12, 24, 25, 48)]
TIME = ["tgt_hour_sin", "tgt_hour_cos", "tgt_dow", "tgt_is_weekend",
        "tgt_month_sin", "tgt_month_cos", "issue_hour"]


def select(allf: list[str], which: str) -> list[str]:
    if which == "full":
        return allf
    if which == "no_neighbor":
        return [c for c in allf if c not in NEIGHBOR]
    if which == "no_meteo":
        return [c for c in allf if not c.startswith("met_")]
    if which == "met_lag1_only":
        return [c for c in allf if not c.startswith("met_") or c.endswith("_lag1")]
    if which == "met_tgt_only":
        return [c for c in allf if not c.startswith("met_") or c.endswith("_tgt")]
    if which == "no_graph":
        return [c for c in allf if c not in GRAPH]
    if which == "graph_only":
        return [c for c in allf if c in GRAPH or c in LAGS]
    if which == "no_pollutant":
        return [c for c in allf if c not in POLLUTANT]
    if which == "lag_only":
        return [c for c in allf if c in LAGS]
    if which == "no_time":
        return [c for c in allf if c not in TIME]
    if which == "no_rolling":
        return [c for c in allf if not c.startswith("pm25_roll") and not c.startswith("pm25_diff")
                and c not in ("pm25_delta24", "pm25_trend")]
    raise ValueError(which)


def weight_of(y: np.ndarray) -> np.ndarray:
    w = np.ones_like(y, dtype="float64")
    w[y >= 75] = 2.0
    w[y >= HEAVY] = 5.0
    return w


def run(which: str, h: int, df: pd.DataFrame, allf: list[str], seed: int):
    feats = select(allf, which)
    tr = df[(df["split"] == "train") & df["y_true"].notna()]
    va = df[(df["split"] == "val") & df["y_true"].notna()]
    te = df[(df["split"] == "test") & df["y_true"].notna()].copy()

    dtr = lgb.Dataset(tr[feats], label=tr["y_true"], weight=weight_of(tr["y_true"].to_numpy()),
                      free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va["y_true"], reference=dtr, free_raw_data=False)
    p = dict(PARAMS)
    p["seed"] = seed
    bo = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    te["y_pred"] = np.clip(bo.predict(te[feats], num_iteration=bo.best_iteration), 0, None)

    b = M.bootstrap_gain(te, n_boot=800)
    hv = te[te["y_true"] >= HEAVY]
    hv_mae = float((hv["y_pred"] - hv["y_true"]).abs().mean())
    hv_b1 = float((hv["b1_persistence"] - hv["y_true"]).abs().mean())
    pk = M.peak_prf(te, M.station_thresholds(df[df["split"] == "train"]))
    row = pk[pk["model"] == "LGBM"].iloc[0]

    return {"h": h, "set": which, "n_feat": len(feats), "seed": seed,
            "MAE": b["MAE"], "gain_pct": b["gain_pct"],
            "ci_lo": b["gain_ci"][0], "ci_hi": b["gain_ci"][1],
            "heavy_MAE": round(hv_mae, 2), "heavy_gain": round((1 - hv_mae / hv_b1) * 100, 2),
            "peak_F1": row["F1_micro"], "peak_R": row["R_micro"], "best_iter": bo.best_iteration}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=6)
    ap.add_argument("--sets", default="full,no_graph,no_neighbor,no_pollutant,no_rolling,lag_only,no_time")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ablation")
    args = ap.parse_args()

    df = pd.read_parquet(FEATDIR / f"h={args.h}" / "part.parquet")
    allf = [c for c in df.columns if c not in META]
    rows = []
    for s in args.sets.split(","):
        r = run(s.strip(), args.h, df, allf, args.seed)
        rows.append(r)
        print(f"  {r['set']:14s} 特征 {r['n_feat']:2d} | MAE {r['MAE']:.3f} ({r['gain_pct']:+.2f}%) "
              f"| 重度 {r['heavy_gain']:+.1f}% | 峰值F1 {r['peak_F1']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    d = REPO / "data" / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{args.out}_h{args.h}_s{args.seed}.csv"
    out.to_csv(f, index=False, encoding="utf-8")
    print(f"\n{out[['set','n_feat','MAE','gain_pct','heavy_gain','peak_F1','peak_R']].to_string(index=False)}")
    print(f"\n写入 {f}")


if __name__ == "__main__":
    main()
