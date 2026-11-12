#!/usr/bin/env python3
"""LightGBM 基线模型：以 B1 持久性为标尺。

只用 train 段拟合、val 段定轮数、test 段不参与任何选择。
输出
  data/runs/{run_id}/metrics.csv      分跨度分切分的 MAE/RMSE 与基线对比
  data/runs/{run_id}/station_mae.csv  分站点误差（宏平均用）
  data/runs/{run_id}/feature_importance.csv
  data/runs/{run_id}/run.json         可复现记录
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aircast import protocol as P  # noqa: E402

FEATDIR = REPO / "data" / "features" / f"protocol={P.PROTOCOL_ID}"
META = {"issue_time", "station_name", "h", "target_start", "target_end", "split",
        "y_true", "b1_persistence", "b2_yesterday"}

PARAMS = dict(
    objective="regression_l1",  # MAE 是主指标，直接优化 L1
    metric="mae",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=16,
)


def heavy_weight(y: np.ndarray, mid: float = 2.0, heavy: float = 5.0) -> np.ndarray:
    """高浓度样本加权。

    experiment_weight 实测：加入气象特征后最优权重会变。
    无气象时 2x5 最优；含气象时需 3x10 以上才能同时保住整体与重度档。
    """
    w = np.ones_like(y, dtype="float64")
    w[y >= 75] = mid
    w[y >= 150] = heavy
    return w


def load(h: int) -> pd.DataFrame:
    f = FEATDIR / f"h={h}" / "part.parquet"
    if not f.exists():
        raise SystemExit(f"缺少特征文件 {f}")
    return pd.read_parquet(f)


def train_one(h: int, run_dir: Path, seed: int, use_weight: bool = False,
              w_mid: float = 2.0, w_heavy: float = 5.0) -> tuple[list, lgb.Booster]:
    df = load(h)
    feats = [c for c in df.columns if c not in META]
    tr = df[(df["split"] == "train") & df["y_true"].notna()]
    va = df[(df["split"] == "val") & df["y_true"].notna()]
    w = heavy_weight(tr["y_true"].to_numpy(), w_mid, w_heavy) if use_weight else None
    print(f"  h={h}: train {len(tr):,} / val {len(va):,} / 特征 {len(feats)} / 加权 {use_weight}")

    dtr = lgb.Dataset(tr[feats], label=tr["y_true"], weight=w, free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va["y_true"], reference=dtr, free_raw_data=False)
    booster = lgb.train(
        PARAMS,
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"    最佳迭代 {booster.best_iteration}，val MAE {booster.best_score['valid_0']['l1']:.3f}")
    # LightGBM 的 C++ 写文件接口不支持含中文的路径，改用字符串导出由 Python 落盘
    (run_dir / f"model_h{h}.txt").write_text(
        booster.model_to_string(num_iteration=booster.best_iteration), encoding="utf-8"
    )

    imp = pd.DataFrame(
        {"feature": feats, "gain": booster.feature_importance("gain")}
    ).sort_values("gain", ascending=False)
    imp["h"] = h
    imp.to_csv(run_dir / f"importance_h{h}.csv", index=False, encoding="utf-8")
    return feats, booster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="lgbm_v1")
    ap.add_argument("--weight", action="store_true", help="启用高浓度样本加权")
    ap.add_argument("--w-mid", type=float, default=2.0)
    ap.add_argument("--w-heavy", type=float, default=5.0)
    args = ap.parse_args()

    run_id = f"{datetime.now():%Y%m%dT%H%M%S}_{args.tag}"
    run_dir = REPO / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id = {run_id}")

    metrics, station_rows = [], []
    for h in P.HORIZONS:
        feats, booster = train_one(h, run_dir, args.seed, args.weight, args.w_mid, args.w_heavy)
        df = load(h)
        for sp in ("val", "test"):
            sub = df[(df["split"] == sp) & df["y_true"].notna()].copy()
            sub["y_pred"] = booster.predict(sub[feats], num_iteration=booster.best_iteration)

            mae = float((sub["y_pred"] - sub["y_true"]).abs().mean())
            rmse = float(np.sqrt(((sub["y_pred"] - sub["y_true"]) ** 2).mean()))
            b1_ok = sub.dropna(subset=["b1_persistence"])
            b1_mae = float((b1_ok["b1_persistence"] - b1_ok["y_true"]).abs().mean())
            gain = (1 - mae / b1_mae) * 100
            # 逐样本胜负必须在同一批样本上比较，否则基线缺值的行会被算成赢
            win_rate = float(
                ((b1_ok["y_pred"] - b1_ok["y_true"]).abs()
                 < (b1_ok["b1_persistence"] - b1_ok["y_true"]).abs()).mean()
            )

            metrics.append(
                {"h": h, "split": sp, "n": len(sub), "MAE": round(mae, 3), "RMSE": round(rmse, 3),
                 "B1_MAE": round(b1_mae, 3), "vs_B1": f"{gain:+.1f}%",
                 "win_rate": round(win_rate, 4)}
            )
            print(f"    {sp}: MAE {mae:.3f} vs B1 {b1_mae:.3f} ({gain:+.1f}%)")

            # 落盘预测，供 evaluate.py 做区间估计、峰值指标与分组分析
            sub[["station_name", "issue_time", "target_start", "y_true", "y_pred", "b1_persistence"]].to_parquet(
                run_dir / f"pred_h{h}_{sp}.parquet", index=False, compression="zstd"
            )

            s = sub.groupby("station_name", observed=True).apply(
                lambda g: pd.Series({"MAE": float((g["y_pred"] - g["y_true"]).abs().mean()),
                                     "n": len(g)}), include_groups=False
            ).reset_index()
            s["h"], s["split"] = h, sp
            station_rows.append(s)

    m = pd.DataFrame(metrics)
    m.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8")
    pd.concat(station_rows).to_csv(run_dir / "station_mae.csv", index=False, encoding="utf-8")

    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_id, "model": "lightgbm", "params": PARAMS, "seed": args.seed,
        "protocol_id": P.PROTOCOL_ID, "horizons": list(P.HORIZONS),
        "features": json.loads((FEATDIR / "feature_list.json").read_text(encoding="utf-8"))["features"],
        "meteo_included": any(c.startswith("met_") for c in json.loads((FEATDIR / "feature_list.json").read_text(encoding="utf-8"))["features"]), "heavy_weight": args.weight, "w_mid": args.w_mid, "w_heavy": args.w_heavy,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + m.to_string(index=False))
    print(f"\n产物 {run_dir}")


if __name__ == "__main__":
    main()
