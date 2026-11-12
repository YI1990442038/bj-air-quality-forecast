#!/usr/bin/env python3
"""发布预测：给定发布时点 T，输出各站 h=1/3/6 的 PM2.5 预报。

与评价脚本的区别：这是**推理入口**，只需要特征，不需要真值。
真值若存在一并列出便于核对，不存在也不影响出结果。

用法
  python scripts/predict.py                       # 默认取特征表里最晚的发布时点
  python scripts/predict.py --time "2026-10-29 18:00"
  python scripts/predict.py --station 东城东四 --time "2026-10-29 18:00"
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

from aircast import protocol as P  # noqa: E402

FEATDIR = REPO / "data" / "features" / f"protocol={P.PROTOCOL_ID}"
DEFAULT_RUN = REPO / "data" / "runs" / "20261031T225104_lgbm_v3_graph"


def load_booster(run: Path, h: int) -> lgb.Booster:
    f = run / f"model_h{h}.txt"
    if not f.exists():
        raise SystemExit(f"缺少模型 {f}")
    # LightGBM 的 C++ 文件接口不支持中文路径，用字符串加载
    return lgb.Booster(model_str=f.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--time", default=None, help="发布时点，默认取最晚可用")
    ap.add_argument("--station", default=None, help="只输出指定站点")
    ap.add_argument("--top", type=int, default=35, help="按预报值排序输出前 N 站；0 表示全部")
    args = ap.parse_args()
    run = Path(args.run)

    frames = []
    for h in P.HORIZONS:
        df = pd.read_parquet(FEATDIR / f"h={h}" / "part.parquet")
        T = pd.Timestamp(args.time) if args.time else df["issue_time"].max()
        sub = df[df["issue_time"] == T]
        if sub.empty:
            raise SystemExit(f"发布时点 {T} 不在特征表范围内（{df.issue_time.min()} ~ {df.issue_time.max()}）")

        bo = load_booster(run, h)
        feats = bo.feature_name()
        missing = [c for c in feats if c not in sub.columns]
        if missing:
            raise SystemExit(f"特征缺失 {missing}，模型与特征表不匹配")

        pred = np.clip(bo.predict(sub[feats]), 0, None)
        frames.append(pd.DataFrame({
            "issue_time": T, "h": h,
            "station_name": sub["station_name"].to_numpy(),
            "target_start": sub["target_start"].to_numpy(),
            "y_pred": np.round(pred, 1),
            "y_true": sub["y_true"].to_numpy(),
            "b1_persistence": sub["b1_persistence"].to_numpy(),
        }))

    out = pd.concat(frames, ignore_index=True)
    T = out["issue_time"].iloc[0]

    if args.station:
        view = out[out["station_name"] == args.station].sort_values("h")
    else:
        last = out[out["h"] == out["h"].max()].sort_values("y_pred", ascending=False)
        keep = set(last["station_name"].head(args.top)) if args.top else set(last["station_name"])
        view = out[out["station_name"].isin(keep)]

    print(f"发布时点 T = {T:%Y-%m-%d %H:%M}（北京时间）　模型 {run.name}")
    print(f"可用观测截至 T-1 = {T - pd.Timedelta(hours=1):%m-%d %H:%M}\n")

    piv = view.pivot_table(index="station_name", columns="h", values="y_pred")
    piv.columns = [f"h={c}" for c in piv.columns]
    tgt = out.drop_duplicates("h").set_index("h")["target_start"]
    print("目标小时：" + "　".join(f"h={h} → {tgt[h]:%m-%d %H:%M}" for h in P.HORIZONS))
    print()
    print(piv.to_string())

    hi = piv.max()
    print(f"\n全城最高预报：h=1 {hi['h=1']:.1f}　h=3 {hi['h=3']:.1f}　h=6 {hi['h=6']:.1f} μg/m³")
    have = out.dropna(subset=["y_true"])
    if len(have):
        mae = (have["y_pred"] - have["y_true"]).abs().mean()
        print(f"该时点有真值可核对，MAE = {mae:.2f} μg/m³")

    d = REPO / "data" / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"pred_{T:%Y%m%d_%H%M}.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    print(f"\n明细已存 {path}")


if __name__ == "__main__":
    main()
