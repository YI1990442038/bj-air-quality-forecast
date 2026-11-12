#!/usr/bin/env python3
"""京津冀邻域图特征（无风向版）。

做法：以 35 个北京站为受体，用北京以外的京津冀国控站作为邻域源，
按「K 近邻均值」和「距离倒数加权」两种方式取 T-1 时刻的 PM2.5。

与已有 bj_bg_loo 的区别：
- bj_bg_loo 只用北京市内其余站，反映本市背景
- 本特征只用北京之外的站，反映**区域输送**，两者互补

局限（必须写明）：当前没有风向，边权只能是各向同性距离。
接入 ERA5 风向后再改为上风向加权，才是真正有方向的图。

输出：把 nb_* 列并入 data/features/protocol=p1/h={h}/part.parquet（幂等）
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
NB_COLS = ["nb5_lag1", "nbinv_lag1", "nb5_lag24", "nbinv_lag24", "nb5_minus_bg"]


def haversine_matrix(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.radians(lat1)[:, None]
    p2 = np.radians(lat2)[None, :]
    dphi = p2 - p1
    dlmb = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_sites() -> pd.DataFrame:
    fs = sorted((REPO / "data" / "warehouse" / "fact_obs" / "source=china_sites").glob("year=*/part.parquet"))
    rows = []
    for f in fs:
        d = pd.read_parquet(f, columns=["station_code", "hour_start_local", "pm25", "network_era"])
        d = d[d["network_era"] == P.ERA]
        if not d.empty:
            rows.append(d)
    if not rows:
        raise SystemExit("china_sites 无纪元 B 数据")
    df = pd.concat(rows, ignore_index=True)
    df = df.drop_duplicates(["station_code", "hour_start_local"], keep="first")
    return df.pivot(index="hour_start_local", columns="station_code", values="pm25")


def main():
    pm = load_sites()
    print(f"国控站矩阵 {pm.shape[0]:,} 小时 × {pm.shape[1]} 站")

    coords = pd.read_csv(REPO / "data" / "meta" / "stations_jingjinji.csv")
    coords = coords.dropna(subset=["lon", "lat"])
    beh = pd.read_csv(REPO / "data" / "meta" / "stations_beijing.csv")
    beh = beh[(beh["network_era"] == P.ERA)].dropna(subset=["lon", "lat"])

    # 邻域源：北京以外、且数据里确实存在的站
    src = coords[(coords["city"] != "北京") & (coords["code"].isin(pm.columns))]
    print(f"邻域源 {len(src)} 站（北京外）｜受体 {len(beh)} 站（北京内）")
    if len(src) < 10:
        raise SystemExit("邻域源过少")

    dist = haversine_matrix(beh["lat"].to_numpy(), beh["lon"].to_numpy(),
                            src["lat"].to_numpy(), src["lon"].to_numpy())
    W_k = np.zeros_like(dist)
    K = 5
    for i in range(dist.shape[0]):
        idx = np.argsort(dist[i])[:K]
        W_k[i, idx] = 1.0
    W_inv = 1.0 / np.maximum(dist, 1.0) ** 2
    W_inv[dist > 150] = 0.0  # 150 km 外不参与，避免远距离噪声
    print(f"最近邻距离：中位 {np.median(np.sort(dist, axis=1)[:, 0]):.1f} km，"
          f"第 {K} 近邻中位 {np.median(np.sort(dist, axis=1)[:, K-1]):.1f} km")

    pm_src = pm[src["code"].to_numpy()]  # 只保留邻域源列，与 dist 第二维严格对齐
    S = pm_src.to_numpy(dtype="float32")
    mask = (~np.isnan(S)).astype("float32")
    S0 = np.nan_to_num(S, nan=0.0)

    def weighted(Wmat):
        num = S0 @ Wmat.T                       # (T, 35)
        den = mask @ Wmat.T
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / den, np.nan)
        return out.astype("float32")

    f5 = weighted(W_k)
    finv = weighted(W_inv)
    t_index = pm_src.index

    # 构造到发布时点的特征：T-1 与 T-24
    def at_lag(arr, lag):
        s = pd.DataFrame(arr, index=t_index)
        return s.shift(lag)

    nb5_1 = at_lag(f5, 1)
    ninv_1 = at_lag(finv, 1)
    nb5_24 = at_lag(f5, 24)
    ninv_24 = at_lag(finv, 24)
    nb5_1.columns = beh["name_cn"].to_numpy()
    ninv_1.columns = beh["name_cn"].to_numpy()
    nb5_24.columns = beh["name_cn"].to_numpy()
    ninv_24.columns = beh["name_cn"].to_numpy()

    stations = beh["name_cn"].to_numpy()
    for h in P.HORIZONS:
        f = FEATDIR / f"h={h}" / "part.parquet"
        df = pd.read_parquet(f)
        times = pd.DatetimeIndex(sorted(df["issue_time"].unique()))
        n_t = len(times)
        assert len(df) % n_t == 0, "行数不能被时点数整除"
        n_s = len(df) // n_t
        # 站点次序以特征文件自身的铺排为准，不能假设与元数据表行序一致
        stations = df["station_name"].to_numpy()[:n_s]
        assert set(stations) == set(beh["name_cn"]), "站点集合与元数据不符"
        assert (df["station_name"].to_numpy() == np.tile(stations, n_t)).all(), "站点铺排不符"

        def flat(mat):
            return mat.reindex(index=times, columns=stations).to_numpy(dtype="float32").ravel(order="C")

        df = df.drop(columns=[c for c in NB_COLS if c in df.columns])
        df["nb5_lag1"] = flat(nb5_1)
        df["nbinv_lag1"] = flat(ninv_1)
        df["nb5_lag24"] = flat(nb5_24)
        df["nbinv_lag24"] = flat(ninv_24)
        df["nb5_minus_bg"] = df["nb5_lag1"] - df["bj_bg_loo"]  # 区域减去本市，突出输送梯度
        df.to_parquet(f, index=False, compression="zstd")
        cov = float(np.isfinite(df["nb5_lag1"]).mean())
        print(f"  h={h} 已并入 5 个邻域特征，可用率 {cov:.1%}")

    print("完成。注意：当前为无风向的距离加权，接 ERA5 风向后再升级为有向图。")


if __name__ == "__main__":
    main()
