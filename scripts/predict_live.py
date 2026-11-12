#!/usr/bin/env python3
"""实时预测：不依赖冻结索引，直接由仓库算特征，可对最新时点发布预报。

为什么要单独一条路径：评价索引 protocol=p1 已冻结（截止 2026-10-30），
若为了预测新时点去重建索引，会改变指纹、使既有实验全部失效。
所以推理走独立通道，但**必须与冻结特征表在重叠时点上逐值一致**，
否则说明两条路径的特征计算已经分叉，预测不可信。

自检不通过会直接终止。

用法
  python scripts/predict_live.py                    # 用最新可用发布时点
  python scripts/predict_live.py --time "2026-10-31 20:00"
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
sys.path.insert(0, str(REPO / "scripts"))

from aircast import protocol as P  # noqa: E402
from build_features import build_era_b, make_features  # noqa: E402
from build_nb_features import haversine_matrix  # noqa: E402

FEATDIR = REPO / "data" / "features" / f"protocol={P.PROTOCOL_ID}"
DEFAULT_RUN = REPO / "data" / "runs" / "20261031T225104_lgbm_v3_graph"
CHECK_TIME = pd.Timestamp("2026-10-20 18:00")  # 选在年度包覆盖范围内，两侧都有真实邻域值


def load_booster(run: Path, h: int) -> lgb.Booster:
    f = run / f"model_h{h}.txt"
    if not f.exists():
        raise SystemExit(f"缺少模型 {f}")
    return lgb.Booster(model_str=f.read_text(encoding="utf-8"))


def load_recent_sites(codes: np.ndarray, days: int = 14) -> pd.DataFrame:
    """从原始日文件读最近若干天的国控站 PM2.5。

    年度包每周才更新一次（最近一版到 2026-10-24），而逐日文件每天更新。
    不做这一步，最近一周的邻域特征会全是 NaN——模型能跑但少了一组关键输入。
    """
    raw = REPO / "data" / "raw" / "china_sites"
    files = sorted(raw.glob("*.csv"))[-days:]
    frames = []
    for f in files:
        head = pd.read_csv(f, nrows=0, encoding="utf-8-sig").columns
        usecols = [c for c in ["date", "hour", "type"] + list(codes) if c in head]
        if len(usecols) < 4:
            continue
        d = pd.read_csv(f, dtype=str, encoding="utf-8-sig", usecols=usecols)
        d["type"] = d["type"].astype(str).str.strip()
        d = d[d["type"] == "PM2.5"]
        if d.empty:
            continue
        long = d.melt(id_vars=["date", "hour", "type"], var_name="station_code", value_name="v")
        long["hour"] = pd.to_numeric(long["hour"], errors="coerce").astype("Int8")
        long = long[long["hour"].between(0, 23)]
        long["value"] = pd.to_numeric(long["v"], errors="coerce")
        long["ts"] = pd.to_datetime(long["date"].str.strip(), format="%Y%m%d") + pd.to_timedelta(
            long["hour"].astype(int), unit="h")
        frames.append(long.pivot_table(index="ts", columns="station_code", values="value", aggfunc="first"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def neighbour_matrices(pm: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """重算京津冀邻域特征，逻辑与 build_nb_features.py 保持一致。"""
    coords = pd.read_csv(REPO / "data" / "meta" / "stations_jingjinji.csv").dropna(subset=["lon", "lat"])
    beh = pd.read_csv(REPO / "data" / "meta" / "stations_beijing.csv")
    beh = beh[(beh["network_era"] == P.ERA)].dropna(subset=["lon", "lat"])

    fs = sorted((REPO / "data" / "warehouse" / "fact_obs" / "source=china_sites").glob("year=*/part.parquet"))
    parts = []
    for f in fs:
        d = pd.read_parquet(f, columns=["station_code", "hour_start_local", "pm25", "network_era"])
        d = d[d["network_era"] == P.ERA]
        if not d.empty:
            parts.append(d)
    sites = pd.concat(parts, ignore_index=True).drop_duplicates(["station_code", "hour_start_local"])
    sites = sites.pivot(index="hour_start_local", columns="station_code", values="pm25")

    # 用逐日文件补齐年度包之后的日子
    recent = load_recent_sites(coords["code"].to_numpy())
    if not recent.empty:
        recent = recent.reindex(columns=sites.columns.union(recent.columns))
        sites = sites.reindex(columns=recent.columns)
        sites = pd.concat([sites[~sites.index.isin(recent.index)], recent]).sort_index()
        print(f"  邻域源已用逐日文件补齐至 {recent.index.max():%Y-%m-%d %H:%M}")

    sites = sites.reindex(pd.date_range(pm.index.min(), pm.index.max(), freq="h"))

    src = coords[(coords["city"] != "北京") & (coords["code"].isin(sites.columns))]
    dist = haversine_matrix(beh["lat"].to_numpy(), beh["lon"].to_numpy(),
                            src["lat"].to_numpy(), src["lon"].to_numpy())
    W_k = np.zeros_like(dist)
    for i in range(dist.shape[0]):
        W_k[i, np.argsort(dist[i])[:5]] = 1.0
    W_inv = 1.0 / np.maximum(dist, 1.0) ** 2
    W_inv[dist > 150] = 0.0

    S = sites[src["code"].to_numpy()].to_numpy(dtype="float32")
    mask = (~np.isnan(S)).astype("float32")
    S0 = np.nan_to_num(S, nan=0.0)

    def weighted(W):
        num = S0 @ W.T
        den = mask @ W.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan).astype("float32")

    out = {}
    for name, arr, lag in (("nb5_lag1", weighted(W_k), 1), ("nbinv_lag1", weighted(W_inv), 1),
                           ("nb5_lag24", weighted(W_k), 24), ("nbinv_lag24", weighted(W_inv), 24)):
        out[name] = pd.DataFrame(arr, index=sites.index).shift(lag)
        out[name].columns = beh["name_cn"].to_numpy()
    return out


def features_at(feats: dict, nb: dict, T: pd.Timestamp, h: int, stations: np.ndarray) -> pd.DataFrame:
    """把某一发布时点的特征装配成 (station × feature) 表。"""
    tgt = T + pd.Timedelta(hours=h - 1)
    cols = {}
    for name, mat in {**feats, **nb}.items():
        if isinstance(mat, pd.DataFrame):
            if T not in mat.index:
                return pd.DataFrame()
            cols[name] = mat.loc[T].reindex(stations).to_numpy(dtype="float64")
        elif isinstance(mat, pd.Series):
            if T not in mat.index:
                return pd.DataFrame()
            cols[name] = np.full(len(stations), float(mat.loc[T]))
        else:
            arr = np.asarray(mat)
            if arr.size != len(stations):
                continue
            cols[name] = arr.astype("float64")

    df = pd.DataFrame(cols, index=stations)
    df["nb5_minus_bg"] = df["nb5_lag1"] - df["bj_bg_loo"]
    hour = tgt.hour
    df["tgt_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["tgt_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["tgt_dow"] = tgt.dayofweek
    df["tgt_is_weekend"] = int(tgt.dayofweek >= 5)
    df["tgt_month_sin"] = np.sin(2 * np.pi * tgt.month / 12)
    df["tgt_month_cos"] = np.cos(2 * np.pi * tgt.month / 12)
    df["issue_hour"] = T.hour
    return df


def build_all(run: Path):
    pm, pm10, extra, cities = build_era_b(REPO)
    feats = make_features(pm, pm10, extra, cities)
    # 一维区域特征转成随时间索引的 Series，便于按 T 取值
    for k in ("region_city_mean", "region_city_mean_lag6", "region_city_mean_lag24"):
        arr = np.asarray(feats[k]).ravel()
        feats[k] = pd.Series(arr, index=pm.index)
    nb = neighbour_matrices(pm)
    return pm, feats, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--time", default=None)
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)

    pm, feats, nb = build_all(run)
    print(f"特征矩阵重建完成：{pm.shape[0]:,} 小时 × {pm.shape[1]} 站，"
          f"覆盖到 {pm.index.max():%Y-%m-%d %H:%M}")

    stations = pm.columns.to_numpy()
    boosters = {h: load_booster(run, h) for h in P.HORIZONS}
    model_feats = boosters[P.HORIZONS[0]].feature_name()

    # ---------- 自检：与冻结特征表在重叠时点逐值比对 ----------
    if not args.skip_check:
        ok = True
        for h in P.HORIZONS:
            live = features_at(feats, nb, CHECK_TIME, h, stations)
            frozen = pd.read_parquet(FEATDIR / f"h={h}" / "part.parquet")
            frozen = frozen[frozen["issue_time"] == CHECK_TIME].set_index("station_name")
            if live.empty or frozen.empty:
                print(f"  ! h={h} 校对时点无数据，跳过")
                ok = False
                continue
            bad = []
            for c in model_feats:
                if c not in live.columns or c not in frozen.columns:
                    bad.append((c, "缺列"))
                    continue
                a = live[c].reindex(frozen.index).to_numpy(dtype="float64")
                b = frozen[c].to_numpy(dtype="float64")
                if not np.allclose(a, b, rtol=1e-4, atol=1e-3, equal_nan=True):
                    bad.append((c, float(np.nanmax(np.abs(a - b)))))
            if bad:
                print(f"  ✗ h={h} 实时特征与冻结表不一致：{bad[:5]}")
                ok = False
            else:
                print(f"  ✓ h={h} 实时特征与冻结表全部一致（{len(model_feats)} 个特征）")
        if not ok:
            raise SystemExit("自检失败：实时特征路径与冻结特征表分叉，拒绝出预测。")

    # ---------- 发布 ----------
    T = pd.Timestamp(args.time) if args.time else pm.index.max() - pd.Timedelta(hours=0)
    rows = []
    for h in P.HORIZONS:
        live = features_at(feats, nb, T, h, stations)
        if live.empty:
            raise SystemExit(f"发布时点 {T} 缺少特征（可能 T-1 数据未到）")
        X = live.reindex(columns=boosters[h].feature_name())
        rows.append(pd.DataFrame({
            "issue_time": T, "h": h, "station_name": stations,
            "target_start": T + pd.Timedelta(hours=h - 1),
            "y_pred": np.round(np.clip(boosters[h].predict(X), 0, None), 1),
        }))
    out = pd.concat(rows, ignore_index=True)

    print(f"\n发布时点 T = {T:%Y-%m-%d %H:%M}（北京时间）")
    print(f"可用观测截至 {T - pd.Timedelta(hours=1):%Y-%m-%d %H:%M}\n")
    piv = out.pivot_table(index="station_name", columns="h", values="y_pred")
    piv.columns = [f"h={c}" for c in piv.columns]
    for h in P.HORIZONS:
        print(f"  h={h} 目标 {T + pd.Timedelta(hours=h-1):%m-%d %H:%M}  "
              f"均值 {piv[f'h={h}'].mean():.1f}  最高 {piv[f'h={h}'].max():.1f}  "
              f"（{piv[f'h={h}'].idxmax()}）")
    print()
    print(piv.sort_values("h=1", ascending=False).to_string())

    d = REPO / "data" / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"live_{T:%Y%m%d_%H%M}.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    print(f"\n明细已存 {path}")


if __name__ == "__main__":
    main()
