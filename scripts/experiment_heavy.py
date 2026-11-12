#!/usr/bin/env python3
"""重污染低估的三种矫正实验。

对比四个变体，看「重度以上」档的增益能否从负转正，同时不牺牲整体指标：
  base  L1 回归（现有基线模型）
  w     高浓度样本加权
  q60   分位数损失 α=0.60（向上偏置）
  q75   分位数损失 α=0.75（更强偏置）
  log   log1p 变换 + L2

用法
  python scripts/experiment_heavy.py --h 6
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
BASE = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16)
HEAVY = 150.0


def sample_weight(y: np.ndarray) -> np.ndarray:
    """高浓度加权：75 以上 ×2，150 以上 ×5。中度以下保持原权重，避免整体失真。"""
    w = np.ones_like(y, dtype="float64")
    w[y >= 75] = 2.0
    w[y >= HEAVY] = 5.0
    return w


def run_variant(name: str, h: int, df: pd.DataFrame, feats: list[str]):
    tr = df[(df["split"] == "train") & df["y_true"].notna()]
    va = df[(df["split"] == "val") & df["y_true"].notna()]
    te = df[(df["split"] == "test") & df["y_true"].notna()].copy()

    params = dict(BASE)
    tr_label, va_label = tr["y_true"].to_numpy(), va["y_true"].to_numpy()
    te_trans = None

    if name == "base":
        params.update(objective="regression_l1", metric="mae")
    elif name == "w":
        params.update(objective="regression_l1", metric="mae")
    elif name.startswith("q"):
        params.update(objective="quantile", alpha=float(name[1:]) / 100, metric="quantile")
    elif name == "log":
        params.update(objective="regression_l2", metric="l2")
        tr_label = np.log1p(tr_label)
        va_label = np.log1p(va_label)
        te_trans = True
    else:
        raise ValueError(name)

    w = sample_weight(tr["y_true"].to_numpy()) if name == "w" else None
    dtr = lgb.Dataset(tr[feats], label=tr_label, weight=w, free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va_label, reference=dtr, free_raw_data=False)
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    pred = booster.predict(te[feats], num_iteration=booster.best_iteration)
    if te_trans:
        pred = np.expm1(pred)
    te["y_pred"] = np.clip(pred, 0, None)

    b = M.bootstrap_gain(te, n_boot=1000)
    heavy = te[te["y_true"] >= HEAVY]
    h_mae = float((heavy["y_pred"] - heavy["y_true"]).abs().mean()) if len(heavy) else np.nan
    h_b1 = float((heavy["b1_persistence"] - heavy["y_true"]).abs().mean()) if len(heavy) else np.nan

    thr = M.station_thresholds(df[df["split"] == "train"])
    pk = M.peak_prf(te, thr)
    lgbm_pk = pk[pk["model"] == "LGBM"].iloc[0]

    return {
        "variant": name, "h": h,
        "overall_MAE": b["MAE"], "B1_MAE": b["B1_MAE"], "gain_pct": b["gain_pct"],
        "gain_ci_lo": b["gain_ci"][0], "gain_ci_hi": b["gain_ci"][1],
        "heavy_n": len(heavy), "heavy_MAE": round(h_mae, 2), "heavy_B1": round(h_b1, 2),
        "heavy_gain": round((1 - h_mae / h_b1) * 100, 2) if len(heavy) and h_b1 else np.nan,
        "peak_F1": lgbm_pk["F1_micro"], "peak_R": lgbm_pk["R_micro"], "peak_P": lgbm_pk["P_micro"],
        "best_iter": booster.best_iteration,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=6)
    ap.add_argument("--variants", default="base,w,q60,q75,log")
    args = ap.parse_args()

    df = pd.read_parquet(FEATDIR / f"h={args.h}" / "part.parquet")
    feats = [c for c in df.columns if c not in META]
    print(f"h={args.h} | 特征 {len(feats)} | 重度以上样本 {int((df[df.split=='test'].y_true>=HEAVY).sum())}")

    rows = []
    for v in args.variants.split(","):
        r = run_variant(v.strip(), args.h, df, feats)
        rows.append(r)
        print(f"  {r['variant']:5s} 整体 MAE {r['overall_MAE']:.3f} ({r['gain_pct']:+.2f}%) | "
              f"重度 MAE {r['heavy_MAE']} vs {r['heavy_B1']} ({r['heavy_gain']:+.1f}%) | "
              f"峰值 F1 {r['peak_F1']:.3f} R {r['peak_R']:.3f}")

    out = pd.DataFrame(rows)
    d = REPO / "data" / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"heavy_h{args.h}.csv", index=False, encoding="utf-8")
    print(f"\n{out[['variant','overall_MAE','gain_pct','heavy_MAE','heavy_B1','heavy_gain','peak_F1','peak_R']].to_string(index=False)}")
    print(f"\n写入 {d / f'heavy_h{args.h}.csv'}")


if __name__ == "__main__":
    main()
