"""数据访问层：懒加载 + 进程内缓存，供 FastAPI 端点调用。

所有读取都基于已落盘的 Parquet / CSV，不依赖任何上游服务。
双时间列（hour_start_local / hour_start_utc）保持北京时间口径。
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
META = os.path.join(DATA, "meta")
RUN_V4 = os.path.join(DATA, "runs", "20261101T170802_lgbm_v4_meteo")

# 进程级缓存
_cache: dict = {}


def _feature_file(h: int) -> str:
    fs = glob.glob(os.path.join(DATA, "features/protocol=p1", f"h={h}", "*.parquet"))
    if not fs:
        raise FileNotFoundError(f"no feature file for h={h}")
    return fs[0]


def get_stations() -> list[dict]:
    """35 个 era-B 站点（建模目标口径），含经纬度。"""
    if "stations" in _cache:
        return _cache["stations"]
    f = _feature_file(1)
    names = list(dict.fromkeys(pq.read_table(f, columns=["station_name"]).column("station_name").to_pylist()))
    meta = pd.read_csv(os.path.join(META, "stations_beijing.csv"))
    coord = {r["name_cn"]: (float(r["lon"]), float(r["lat"])) for _, r in meta.iterrows()}
    stations = [{"name": n, "lon": coord.get(n, (None, None))[0], "lat": coord.get(n, (None, None))[1]} for n in names]
    _cache["stations"] = stations
    return stations


def _fact_obs_era_b() -> pd.DataFrame:
    """加载 fact_obs（era B）。

    注意：warehouse 按 source 分区，beijing_all 含 pm25/pm10/aqi，beijing_extra
    只有气体（无 pm25），二者 schema 不同且在相同 (站,时) 各有一行。
    因此全量读取、补列统一，再按 pm25 非空行筛选以避免重复计数。
    """
    if "fact_b" in _cache:
        return _cache["fact_b"]
    pats = glob.glob(os.path.join(DATA, "warehouse/fact_obs/**/*.parquet"), recursive=True)
    need = ["name_cn", "hour_start_local", "pm25", "pm10", "aqi", "value_status", "network_era"]
    frames = []
    for p in pats:
        df = pq.read_table(p).to_pandas(types_mapper=None)
        for c in need:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[need]
        df = df[df["network_era"] == "B"]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=need)
    out["hour_start_local"] = pd.to_datetime(out["hour_start_local"])
    out["pm25"] = pd.to_numeric(out["pm25"], errors="coerce")
    # fact_obs 是跨源统一表（含全国/京津冀站点），仅保留 35 个北京建模站
    bj_names = {s["name"] for s in get_stations()}
    out = out[out["name_cn"].isin(bj_names)]
    # 仓库存在 (站,时) 重复行（部分组 pm25 取值冲突），按 (站,时) 确定性去重，
    # 优先保留 observed 状态，保证演示时间序列一一对应。
    _rank = {"observed": 0, "missing_value": 1, "out_of_range": 2}
    out = (
        out.assign(_r=out["value_status"].map(_rank).fillna(3))
        .sort_values(["name_cn", "hour_start_local", "_r"])
        .drop_duplicates(["name_cn", "hour_start_local"], keep="first")
        .drop(columns="_r")
    )
    _cache["fact_b"] = out
    return out


def get_overview() -> dict:
    """数据概览：覆盖区间、站点观测率、时间轴完整度。"""
    if "overview" in _cache:
        return _cache["overview"]
    df = _fact_obs_era_b()
    df = df[df["pm25"].notna()]  # 仅 PM 分区，避免与气体分区重复计数
    grp = df.groupby("name_cn")
    per = grp.agg(n_rows=("pm25", "size"), n_obs=("value_status", lambda s: (s == "observed").sum())).reset_index()
    per["obs_rate"] = (per["n_obs"] / per["n_rows"]).round(4)
    per["missing_rate"] = (1 - per["obs_rate"]).round(4)
    # 时间轴完整度（基于有观测的整点占理论整点）
    tmin, tmax = df["hour_start_local"].min(), df["hour_start_local"].max()
    n_stations = per.shape[0]
    total_hours = int((tmax - tmin).total_seconds() // 3600) + 1
    theoretical = total_hours * n_stations
    actual_obs = int((df["value_status"] == "observed").sum())
    overview = {
        "date_min": str(tmin),
        "date_max": str(tmax),
        "n_stations": n_stations,
        "theoretical_hours": theoretical,
        "observed_hours": actual_obs,
        "time_completeness": round(actual_obs / theoretical, 4) if theoretical else None,
        "per_station": per.rename(columns={"name_cn": "station"})[
            ["station", "n_rows", "n_obs", "obs_rate", "missing_rate"]
        ].to_dict("records"),
    }
    _cache["overview"] = overview
    return overview


def get_trend(station: str, start: str | None = None, end: str | None = None, limit: int = 2000) -> dict:
    """单站趋势：pm25 / pm10 / aqi 时间序列 + 质量状态标记。"""
    df = _fact_obs_era_b()
    sub = df[(df["name_cn"] == station) & df["pm25"].notna()].sort_values("hour_start_local")
    if start:
        sub = sub[sub["hour_start_local"] >= pd.Timestamp(start)]
    if end:
        sub = sub[sub["hour_start_local"] <= pd.Timestamp(end)]
    if len(sub) > limit:
        sub = sub.tail(limit)
    return {
        "station": station,
        "times": [str(t) for t in sub["hour_start_local"]],
        "pm25": [None if pd.isna(v) else float(v) for v in sub["pm25"]],
        "pm10": [None if pd.isna(v) else float(v) for v in sub["pm10"]],
        "aqi": [None if pd.isna(v) else float(v) for v in sub["aqi"]],
        "status": [str(s) for s in sub["value_status"]],
    }


def _pred_table(h: int) -> pd.DataFrame:
    key = f"pred_{h}"
    if key not in _cache:
        _cache[key] = pq.read_table(os.path.join(RUN_V4, f"pred_h{h}_test.parquet")).to_pandas()
    return _cache[key]


def get_replay(station: str, h: int = 6, mode: str = "review", replay_time: str | None = None, limit: int = 1500) -> dict:
    """预测回放：某站某跨度的 预测 vs 真值 vs 基线B1。

    双模式信息拦截（来自方案 §7）：
    - retrospective_review：返回全部历史成绩
    - historical_replay：只返回 target_end <= replay_time 的样本（不得泄露未来）
    """
    df = _pred_table(h)
    sub = df[df["station_name"] == station].sort_values("issue_time")
    if mode == "historical_replay":
        rt = pd.Timestamp(replay_time) if replay_time else sub["target_start"].max()
        sub = sub[sub["target_start"] <= rt]
    if len(sub) > limit:
        sub = sub.tail(limit)
    return {
        "station": station,
        "h": h,
        "mode": mode,
        "replay_time": str(replay_time) if replay_time else None,
        "issue_times": [str(t) for t in sub["issue_time"]],
        "y_true": [None if pd.isna(v) else float(v) for v in sub["y_true"]],
        "y_pred": [None if pd.isna(v) else float(v) for v in sub["y_pred"]],
        "b1": [None if pd.isna(v) else float(v) for v in sub["b1_persistence"]],
    }


def get_comparison() -> dict:
    """模型对比：v4(LightGBM) 主指标、峰值 F1、特征重要度；若有 DL 结果一并纳入。"""
    if "comparison" in _cache:
        return _cache["comparison"]

    metrics = pd.read_csv(os.path.join(RUN_V4, "metrics.csv"))
    peak = pd.read_csv(os.path.join(RUN_V4, "eval_peak.csv"))
    importance = {}
    for h in (1, 3, 6):
        p = os.path.join(RUN_V4, f"importance_h{h}.csv")
        if os.path.exists(p):
            imp = pd.read_csv(p).sort_values("gain", ascending=False).head(12)
            importance[str(h)] = imp.rename(columns={"feature": "name", "gain": "value"})[
                ["name", "value"]
            ].to_dict("records")

    comp = {
        "lightgbm_v4": {
            "metrics": metrics.to_dict("records"),
            "peak": peak.to_dict("records"),
            "importance": importance,
            "run_id": "20261101T170802_lgbm_v4_meteo",
        }
    }

    # 若存在 DL 训练结果，自动并入
    dl_dir = os.path.join(DATA, "runs")
    for d in sorted(glob.glob(os.path.join(dl_dir, "*dl*"))):
        rid = os.path.basename(d)
        mp = os.path.join(d, "metrics.csv")
        if os.path.exists(mp):
            dm = pd.read_csv(mp)
            comp[f"dl_{rid}"] = {"metrics": dm.to_dict("records"), "run_id": rid}

    # 扁平化 DL 结果列表，供前端「深度学习对比」表直接渲染
    # （每个 run 的 metric 行展开，附 run_id；前端表格按 run/h/split 展示）
    dl_rows = []
    for key, val in list(comp.items()):
        if key.startswith("dl_"):
            rid = val.get("run_id", key[3:])
            for m in val.get("metrics", []):
                row = dict(m)
                row["run_id"] = rid
                # 友好标签：模型 + 版本（v1 单通道 / v2 多通道时空）
                ver = "v2" if "_dl2_" in rid else "v1"
                mdl = "LSTM" if rid.endswith("lstm") else ("TCN" if rid.endswith("tcn") else rid)
                row["label"] = f"{mdl} {ver}"
                vs = row.get("vs_B1")
                if isinstance(vs, (int, float)):
                    row["vs_B1"] = f"{vs:+.1f}%"
                dl_rows.append(row)
    comp["dl"] = dl_rows

    _cache["comparison"] = comp
    return comp


def get_live() -> dict:
    """实时预报：读取最新 data/predictions/live_*.parquet（predict_live 产出）。"""
    d = os.path.join(DATA, "predictions")
    fs = sorted(glob.glob(os.path.join(d, "live_*.parquet")))
    if not fs:
        return {"available": False}
    f = fs[-1]
    df = pd.read_parquet(f)
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    issue = df["issue_time"].min()
    piv = df.pivot_table(index="station_name", columns="h", values="y_pred")
    rows = []
    for name, r in piv.iterrows():
        def gv(h):
            return None if h not in r.index or pd.isna(r[h]) else round(float(r[h]), 1)
        rows.append({"station": name, "h1": gv(1), "h3": gv(3), "h6": gv(6)})
    rows.sort(key=lambda x: (x["h1"] is None, -(x["h1"] or 0)))
    targets = {}
    for h in (1, 3, 6):
        sub = df[df["h"] == h]
        if len(sub):
            targets[str(h)] = str(pd.to_datetime(sub["target_start"]).iloc[0])
    return {"available": True, "issue_time": str(issue), "file": os.path.basename(f),
            "targets": targets, "rows": rows}
