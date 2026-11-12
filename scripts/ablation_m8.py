#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M8 消融：量化京津冀邻域特征对 LightGBM 的贡献（本地）。

对比同一特征表下：
  - with_neighbor：保留 nb5_* / nbinv_* / bj_bg_loo / region_city_mean*
  - no_neighbor  ：删除上述邻域特征
在 h=1/3/6 的 test 上比较 MAE，得出邻域特征的相对增益。

产出：data/experiments/ablation_m8.csv
用法：python scripts/ablation_m8.py
"""
from __future__ import annotations

import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EXCLUDE = {"issue_time", "station_name", "h", "target_start", "target_end",
           "split", "y_true", "b1_persistence", "b2_yesterday"}
NB_KEYS = ("nb5", "nbinv", "bj_bg", "region_city")


def is_nb(c: str) -> bool:
    return any(k in c for k in NB_KEYS)


def run():
    rows = []
    for h in (1, 3, 6):
        f = glob.glob(os.path.join(DATA, "features/protocol=p1", f"h={h}", "*.parquet"))[0]
        t = pq.read_table(f).to_pandas()
        # 目标缺失行（y_true/b1 为 NaN）会导致 MAE=NaN，先剔除
        t = t[t["y_true"].notna() & t["b1_persistence"].notna()]
        tr = t[t["split"] == "train"]
        te = t[t["split"] == "test"]
        if len(tr) > 500_000:
            tr = tr.sample(n=500_000, random_state=42)
        for name, drop_nb in [("with_neighbor", False), ("no_neighbor", True)]:
            cols = [c for c in t.columns if c not in EXCLUDE]
            if drop_nb:
                cols = [c for c in cols if not is_nb(c)]
            m = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=63,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                n_jobs=8, random_state=42, verbose=-1)
            m.fit(tr[cols], tr["y_true"])
            p = m.predict(te[cols])
            mae = float(np.abs(p - te["y_true"].to_numpy()).mean())
            rmse = float(np.sqrt(((p - te["y_true"].to_numpy()) ** 2).mean()))
            b1mae = float(np.abs(te["b1_persistence"].to_numpy() - te["y_true"].to_numpy()).mean())
            rows.append({"h": h, "variant": name, "n_features": len(cols),
                         "n_train": len(tr), "MAE": round(mae, 3), "RMSE": round(rmse, 3),
                         "B1_MAE": round(b1mae, 3),
                         "vs_B1_pct": round((b1mae - mae) / b1mae * 100, 2)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(DATA, "experiments", "ablation_m8.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    # 打印增益
    print("\n=== 邻域特征增益 (with - no, MAE 降幅) ===")
    for h in (1, 3, 6):
        w = df[(df.h == h) & (df.variant == "with_neighbor")].MAE.values[0]
        n = df[(df.h == h) & (df.variant == "no_neighbor")].MAE.values[0]
        print(f"h={h}: with={w:.3f} no={n:.3f} 邻域使 MAE {'降' if w<n else '升'} "
              f"{abs(n-w)/n*100:.2f}%")
    print("saved ->", out)


if __name__ == "__main__":
    run()
