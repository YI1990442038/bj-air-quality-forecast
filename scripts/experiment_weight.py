#!/usr/bin/env python3
"""权重 × 气象 交叉实验：解释「加气象后重度档增益反而下降」的反常。

假设：样本加权参数（≥75×2、≥150×5）是为『无气象』模型调的。
那时重度档靠 pm25_lag1 撑住，重加权有效；
加入气象后模型决策依据转向边界层高度/风速，原权重可能失配。

做法：固定特征集，只改权重档位，看整体与重度档如何此消彼长。

用法
  python scripts/experiment_weight.py --h 6 --sets no_meteo,full
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

# (标签, 中度阈值权重, 重度阈值权重)；None 表示不加权
WEIGHTS = [("none", None, None), ("2x5x", 2.0, 5.0), ("3x10x", 3.0, 10.0), ("5x20x", 5.0, 20.0)]


def make_w(y: np.ndarray, w_mid, w_heavy):
    if w_mid is None:
        return None
    w = np.ones_like(y, dtype="float64")
    w[y >= 75] = w_mid
    w[y >= HEAVY] = w_heavy
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=6)
    ap.add_argument("--sets", default="no_meteo,full")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(FEATDIR / f"h={args.h}" / "part.parquet")
    allf = [c for c in df.columns if c not in META]
    sets = {
        "no_meteo": [c for c in allf if not c.startswith("met_")],
        "met_lag1_only": [c for c in allf if not c.startswith("met_") or c.endswith("_lag1")],
        "met_tgt_only": [c for c in allf if not c.startswith("met_") or c.endswith("_tgt")],
        "full": allf,
    }

    rows = []
    for sname in args.sets.split(","):
        feats = sets[sname]
        tr = df[(df["split"] == "train") & df["y_true"].notna()]
        va = df[(df["split"] == "val") & df["y_true"].notna()]
        te = df[(df["split"] == "test") & df["y_true"].notna()].copy()
        for label, wm, wh in WEIGHTS:
            p = dict(PARAMS)
            p["seed"] = args.seed
            dtr = lgb.Dataset(tr[feats], label=tr["y_true"],
                              weight=make_w(tr["y_true"].to_numpy(), wm, wh), free_raw_data=False)
            dva = lgb.Dataset(va[feats], label=va["y_true"], reference=dtr, free_raw_data=False)
            bo = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            te["y_pred"] = np.clip(bo.predict(te[feats], num_iteration=bo.best_iteration), 0, None)

            b = M.bootstrap_gain(te, n_boot=600)
            hv = te[te["y_true"] >= HEAVY]
            hm = float((hv["y_pred"] - hv["y_true"]).abs().mean())
            hb = float((hv["b1_persistence"] - hv["y_true"]).abs().mean())
            pk = M.peak_prf(te, M.station_thresholds(df[df["split"] == "train"]))
            row = pk[pk["model"] == "LGBM"].iloc[0]
            rows.append({"h": args.h, "set": sname, "weight": label, "n_feat": len(feats),
                         "MAE": b["MAE"], "gain_pct": b["gain_pct"],
                         "heavy_MAE": round(hm, 2), "heavy_gain": round((1 - hm / hb) * 100, 2),
                         "peak_F1": row["F1_micro"], "peak_R": row["R_micro"]})
            print(f"  {sname:9s} 权重 {label:6s} | MAE {b['MAE']:.3f} ({b['gain_pct']:+.2f}%) "
                  f"| 重度 {rows[-1]['heavy_gain']:+.1f}% | F1 {row['F1_micro']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    d = REPO / "data" / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"weight_x_meteo_h{args.h}.csv"
    out.to_csv(f, index=False, encoding="utf-8")
    print(f"\n{out[['set','weight','MAE','gain_pct','heavy_gain','peak_F1']].to_string(index=False)}")
    print(f"\n写入 {f}")


if __name__ == "__main__":
    main()
