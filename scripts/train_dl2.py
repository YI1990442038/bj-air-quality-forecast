#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M5 深度学习 v2：多通道时空序列 LSTM / TCN。

相比 v1（只 5 维：pm25 + 时间周期），v2 引入：
  - 站点自有污染物 pm25 / pm10 / aqi（beijing_all）
  - 京津冀邻域均值 region_city_mean（M8 时空信号，data/meta/region_hourly.parquet）
  - 气象 met_temp/rh/wspd/blh/pres/ssrd（data/meta/meteo_hourly.parquet）
  - 时间周期 hour/dow 正余弦
共 10 个基础通道 + 4 个时间通道 = 14 维/时刻，输入窗口 24 小时。

协议与 LightGBM v4 一致：切分按发布时间，train<2025-01-01, val[2025-01-01,2025-07-01), test>=2025-07-01；
B1 基线 = 发布时点 t 的观测。无泄漏：归一化统计量仅在 train 拟合。

产出：data/runs/<ts>_dl2_<model>/ 下 pred_h{1,3,6}_test.parquet + metrics.csv + 模型权重
用法（a40）：
    ~/miniconda3/envs/biomatrix/bin/python scripts/train_dl2.py --model lstm
    ~/miniconda3/envs/biomatrix/bin/python scripts/train_dl2.py --model tcn
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SEQ = 24
TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-07-01")

BASE_CH = ["pm25", "pm10", "aqi", "region_city_mean",
           "met_temp_tgt", "met_rh_tgt", "met_wspd_tgt", "met_blh_tgt",
           "met_pres_tgt", "met_ssrd_tgt"]
TIME_CH = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
CHANNELS = BASE_CH + TIME_CH
N_BASE = len(BASE_CH)
N_FEATURES = len(CHANNELS)


def _bj_names() -> list[str]:
    fs = glob.glob(os.path.join(DATA, "features/protocol=p1/h=1/*.parquet"))
    if fs:
        return list(dict.fromkeys(
            pq.read_table(fs[0], columns=["station_name"]).column("station_name").to_pylist()))
    txt = os.path.join(DATA, "meta/station_names_35.txt")
    with open(txt, encoding="utf-8") as fp:
        return [ln.strip() for ln in fp if ln.strip()]


def load_frames() -> dict[str, pd.DataFrame]:
    """每站一个连续小时网格的 DataFrame，含 10 个基础通道（未归一化）。"""
    names = set(_bj_names())
    ps = glob.glob(os.path.join(DATA, "warehouse/fact_obs/source=beijing_all/**/*.parquet"),
                   recursive=True)
    need = ["name_cn", "hour_start_local", "pm25", "pm10", "aqi", "value_status", "network_era"]
    frames = []
    for p in ps:
        df = pq.read_table(p).to_pandas(types_mapper=None)
        for c in need:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[need])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["network_era"] == "B"]
    df = df[df["name_cn"].isin(names)]
    _rank = {"observed": 0, "missing_value": 1, "out_of_range": 2}
    df = (df.assign(_r=df["value_status"].map(_rank).fillna(3))
            .sort_values(["name_cn", "hour_start_local", "_r"])
            .drop_duplicates(["name_cn", "hour_start_local"], keep="first")
            .drop(columns="_r"))
    for c in ("pm25", "pm10", "aqi"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["hour_start_local"] = pd.to_datetime(df["hour_start_local"])

    reg = pd.read_parquet(os.path.join(DATA, "meta/region_hourly.parquet"))
    reg["hour_start_local"] = pd.to_datetime(reg["hour_start_local"])
    met = pd.read_parquet(os.path.join(DATA, "meta/meteo_hourly.parquet"))
    met["hour_start_local"] = pd.to_datetime(met["hour_start_local"])

    out = {}
    for name, g in df.groupby("name_cn"):
        g = g.sort_values("hour_start_local")[["hour_start_local", "pm25", "pm10", "aqi"]]
        full = pd.date_range(g["hour_start_local"].min(), g["hour_start_local"].max(), freq="h")
        base = g.set_index("hour_start_local").reindex(full).rename_axis("hour_start_local").reset_index()
        # 邻域均值（按小时 join）
        base = base.merge(reg[["hour_start_local", "region_city_mean"]], on="hour_start_local", how="left")
        # 气象（按 站+小时 join）
        mm = met[met["station_name"] == name][["hour_start_local"] + [c for c in BASE_CH if c.startswith("met_")]]
        base = base.merge(mm, on="hour_start_local", how="left")
        # 时间周期
        h = base["hour_start_local"]
        base["hour_sin"] = np.sin(2 * np.pi * h.dt.hour / 24)
        base["hour_cos"] = np.cos(2 * np.pi * h.dt.hour / 24)
        base["dow_sin"] = np.sin(2 * np.pi * h.dt.dayofweek / 7)
        base["dow_cos"] = np.cos(2 * np.pi * h.dt.dayofweek / 7)
        # 前向填充基础通道（pm 与气象/邻域），保证窗口连续
        for c in BASE_CH:
            base[c] = base[c].ffill()
        out[name] = base
    return out


def build_dataset(frames: dict[str, pd.DataFrame], horizons=(1, 3, 6), max_train=None):
    """构建多通道滑动窗口样本，返回归一化后的 X 及标签/切分等。"""
    names = list(frames.keys())
    name2id = {n: i for i, n in enumerate(names)}

    # 1) 先算 train 期各通道归一化统计量（原始小时值，pm25 与各基础通道）
    sums = np.zeros(N_BASE); sqs = np.zeros(N_BASE); cnt = np.zeros(N_BASE)
    for name in names:
        b = frames[name]
        m = b["hour_start_local"] < TRAIN_END
        for ci, c in enumerate(BASE_CH):
            v = b.loc[m, c].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if v.size:
                sums[ci] += v.sum(); sqs[ci] += (v ** 2).sum(); cnt[ci] += v.size
    mu = sums / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(sqs / np.maximum(cnt, 1) - mu ** 2, 1e-12))
    y_mu, y_sd = mu[0], sd[0]  # pm25 即通道 0

    Xs, SIDs, Ys, B1s, SPs, ITs, TSt, NMs = [], [], [], [], [], [], [], []
    for name in names:
        b = frames[name]
        n = len(b)
        if n < SEQ + max(horizons) + 1:
            continue
        feat = b[BASE_CH].to_numpy(dtype=np.float32)  # (n, N_BASE)
        # 归一化基础通道
        for ci in range(N_BASE):
            feat[:, ci] = (feat[:, ci] - mu[ci]) / sd[ci]
        tarr = b["hour_start_local"].to_numpy()
        feat = np.concatenate([feat, b[TIME_CH].to_numpy(dtype=np.float32)], axis=1)  # (n, 14)
        nanflag = np.isnan(feat).any(axis=1).astype(np.int8)  # 任一通道 NaN 即视为无效时刻

        _win = np.lib.stride_tricks.sliding_window_view(feat, SEQ, axis=0)   # (m, 14, SEQ)
        win = np.ascontiguousarray(_win.transpose(0, 2, 1))                   # (m, SEQ, 14)
        issue = np.arange(SEQ - 1, n)
        winnan = np.lib.stride_tricks.sliding_window_view(nanflag, SEQ).sum(axis=1)
        tidx = pd.DatetimeIndex(tarr)
        pm_raw = b["pm25"].to_numpy(dtype=float)  # 原始 pm25 供标签/B1

        for h in horizons:
            tgt = issue + h
            ok = (winnan == 0) & (nanflag[issue] == 0) & (tgt < n)
            idx = np.where(ok)[0]
            if len(idx) == 0:
                continue
            abs_i = issue[idx]; tgt_i = abs_i + h
            y = pm_raw[tgt_i]
            b1 = pm_raw[abs_i]
            good = ~(np.isnan(y) | np.isnan(b1))
            idx = idx[good]; abs_i = abs_i[good]; tgt_i = tgt_i[good]
            if len(idx) == 0:
                continue
            Xs.append(win[idx])
            SIDs.append(np.full(len(idx), name2id[name], dtype=np.int64))
            Ys.append(pm_raw[tgt_i].astype(np.float32))
            B1s.append(pm_raw[abs_i].astype(np.float32))
            ti = tidx[abs_i]
            SPs.append(np.where(ti < TRAIN_END, "train", np.where(ti < VAL_END, "val", "test")))
            ITs.append(tarr[abs_i])
            TSt.append((tidx[abs_i] + pd.Timedelta(hours=h)).to_numpy())
            NMs.append(np.full(len(idx), name))

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    sid = np.concatenate(SIDs); yorig = np.concatenate(Ys)
    b1 = np.concatenate(B1s); split = np.concatenate(SPs)
    itime = np.concatenate(ITs); tstart = np.concatenate(TSt); nname = np.concatenate(NMs)
    ynorm = ((yorig - y_mu) / y_sd).astype(np.float32)

    tr = split == "train"
    if max_train and int(tr.sum()) > max_train:
        idx = np.where(tr)[0]
        keep = np.random.RandomState(0).choice(idx, max_train, replace=False)
        mask = np.zeros(len(split), dtype=bool); mask[keep] = True; mask[~tr] = True
        X, sid, ynorm, yorig, b1, split, itime, tstart, nname = (
            X[mask], sid[mask], ynorm[mask], yorig[mask], b1[mask],
            split[mask], itime[mask], tstart[mask], nname[mask])

    return {"X": X, "sid": sid, "ynorm": ynorm, "yorig": yorig, "b1": b1,
            "split": split, "itime": itime, "tstart": tstart, "nname": nname,
            "y_mu": float(y_mu), "y_sd": float(y_sd), "n_stations": len(names)}


def make_loaders(d, h, batch=2048):
    m = (d["tstart"] - d["itime"]).astype("timedelta64[h]").astype(int) == h
    parts = {}
    for sp in ("train", "val", "test"):
        sel = m & (d["split"] == sp)
        parts[sp] = TensorDataset(
            torch.from_numpy(d["X"][sel]), torch.from_numpy(d["sid"][sel]),
            torch.from_numpy(d["ynorm"][sel]), torch.from_numpy(d["yorig"][sel]),
            torch.from_numpy(d["b1"][sel]))
    return {sp: DataLoader(ds, batch_size=batch, shuffle=(sp == "train")) for sp, ds in parts.items()}, m


class LSTMModel(nn.Module):
    def __init__(self, n_stations, hidden=96, emb=8, layers=2):
        super().__init__()
        self.emb = nn.Embedding(n_stations, emb)
        self.lstm = nn.LSTM(N_FEATURES, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + emb, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, sid):
        out, _ = self.lstm(x)
        e = self.emb(sid)
        return self.head(torch.cat([out[:, -1, :], e], dim=1)).squeeze(-1)


class TCNModel(nn.Module):
    def __init__(self, n_stations, emb=8, channels=(64, 64, 64), kernel=3):
        super().__init__()
        self.emb = nn.Embedding(n_stations, emb)
        layers = []
        in_ch = N_FEATURES
        for c in channels:
            layers += [nn.Conv1d(in_ch, c, kernel, padding=kernel // 2), nn.ReLU()]
            in_ch = c
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(channels[-1] + emb, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, sid):
        y = self.net(x.transpose(1, 2)).mean(dim=2)
        e = self.emb(sid)
        return self.head(torch.cat([y, e], dim=1)).squeeze(-1)


def train_one(model, loaders, epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.L1Loss()
    for ep in range(epochs):
        model.train()
        for x, s, y, _, _ in loaders["train"]:
            x, s, y = x.to(device), s.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x, s), y); loss.backward(); opt.step()
        sched.step()
        model.eval(); errs = []
        with torch.no_grad():
            for x, s, y, _, _ in loaders["val"]:
                x, s, y = x.to(device), s.to(device), y.to(device)
                errs.append((model(x, s) - y).abs().mean().item())
        print(f"   epoch {ep+1}/{epochs}  val MAE(norm)={np.mean(errs):.4f}", flush=True)


@torch.no_grad()
def evaluate(model, loader_test, y_mu, y_sd, device):
    model.eval(); preds, yt, b1 = [], [], []
    for x, s, _, yo, bb in loader_test:
        x, s = x.to(device), s.to(device)
        p = model(x, s).cpu().numpy() * y_sd + y_mu
        preds.append(p); yt.append(yo.numpy()); b1.append(bb.numpy())
    preds = np.concatenate(preds); yt = np.concatenate(yt); b1 = np.concatenate(b1)
    mae = float(np.abs(preds - yt).mean()); rmse = float(np.sqrt(((preds - yt) ** 2).mean()))
    b1mae = float(np.abs(b1 - yt).mean()); win = float((np.abs(preds - yt) < np.abs(b1 - yt)).mean())
    vs = (b1mae - mae) / b1mae * 100 if b1mae > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "B1_MAE": b1mae, "vs_B1": vs, "win_rate": win}, preds, yt, b1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "tcn"], default="lstm")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--max-train", type=int, default=800000)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M5v2] model={args.model} device={device} channels={N_FEATURES}", flush=True)
    frames = load_frames()
    print(f"[M5v2] stations={len(frames)}", flush=True)
    ds = build_dataset(frames, horizons=tuple(args.horizons), max_train=args.max_train)
    print(f"      X={ds['X'].shape} train/val/test="
          f"{(ds['split']=='train').sum()}/{(ds['split']=='val').sum()}/{(ds['split']=='test').sum()}",
          flush=True)

    run_id = f"{pd.Timestamp.now():%Y%m%dT%H%M%S}_dl2_{args.model}"
    out_dir = os.path.join(DATA, "runs", run_id); os.makedirs(out_dir, exist_ok=True)
    metrics_rows = []
    for h in args.horizons:
        print(f"[M5v2] horizon h={h}", flush=True)
        loaders, mask = make_loaders(ds, h, batch=args.batch)
        model = LSTMModel(ds["n_stations"]) if args.model == "lstm" else TCNModel(ds["n_stations"])
        model.to(device)
        train_one(model, loaders, args.epochs, device)
        res, preds, yt, b1 = evaluate(model, loaders["test"], ds["y_mu"], ds["y_sd"], device)
        metrics_rows.append({"h": h, "split": "test", "n": int(mask.sum()), **res})
        print(f"      test MAE={res['MAE']:.3f} B1={res['B1_MAE']:.3f} "
              f"vs_B1={res['vs_B1']:+.1f}% win={res['win_rate']:.3f}", flush=True)
        sel = mask & (ds["split"] == "test")
        tbl = pa.table({
            "station_name": ds["nname"][sel],
            "issue_time": ds["itime"][sel].astype("datetime64[us]"),
            "target_start": ds["tstart"][sel].astype("datetime64[us]"),
            "y_true": yt, "y_pred": preds, "b1_persistence": b1})
        pq.write_table(tbl, os.path.join(out_dir, f"pred_h{h}_test.parquet"))
        torch.save(model.state_dict(), os.path.join(out_dir, f"model_h{h}.pt"))

    pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as fp:
        json.dump({"run_id": run_id, "model": args.model, "version": "dl2",
                   "channels": CHANNELS, "epochs": args.epochs, "horizons": args.horizons,
                   "y_mu": ds["y_mu"], "y_sd": ds["y_sd"], "device": str(device)},
                  fp, ensure_ascii=False, indent=2)
    print(f"[M5v2] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
