#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M5 深度学习：LSTM / TCN 多站点 PM2.5 短期预测。

数据：warehouse/fact_obs（北京 35 站 era-B，按 (站,时) 去重）
协议：切分与 LightGBM v4 一致 —— train < 2025-01-01, val [2025-01-01,2025-07-01), test >= 2025-07-01
目标：h=1/3/6 单目标小时；B1 基线 = 发布时点 t 的观测(persistence)
无泄漏：归一化统计量仅在 train 拟合；输入窗口只用 t 及之前时刻。

产出：data/runs/<ts>_dl_<model>/ 下 pred_h{1,3,6}_test.parquet + metrics.csv + 模型权重
      （命名含 `dl`，可被 app 的 /api/comparison 自动纳入模型对比页）

用法：
    python scripts/train_dl.py --model lstm
    python scripts/train_dl.py --model tcn --epochs 15 --max-train 600000
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SEQ = 24  # 输入窗口：过去 24 小时
FEAT_PM = 1
# 特征：pm25 + hour_sin + hour_cos + dow_sin + dow_cos
N_FEATURES = 5

TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-07-01")


def _bj_names() -> list[str]:
    """35 个建模站名。

    优先读已构建的 feature 文件；若服务器上未上传大体积 feature 文件，
    回退到 data/meta/station_names_35.txt（同一份 35 站清单）。
    """
    fs = glob.glob(os.path.join(DATA, "features/protocol=p1/h=1/*.parquet"))
    if fs:
        return list(dict.fromkeys(
            pq.read_table(fs[0], columns=["station_name"]).column("station_name").to_pylist()))
    txt = os.path.join(DATA, "meta/station_names_35.txt")
    with open(txt, encoding="utf-8") as fp:
        return [ln.strip() for ln in fp if ln.strip()]


def load_series() -> dict[str, tuple[list, np.ndarray]]:
    """返回 {站名: (时间列表, pm25 数组)}，已去重 + 前向填充。"""
    pats = glob.glob(os.path.join(DATA, "warehouse/fact_obs/**/*.parquet"), recursive=True)
    need = ["name_cn", "hour_start_local", "pm25", "value_status", "network_era"]
    frames = []
    for p in pats:
        df = pq.read_table(p).to_pandas(types_mapper=None)
        for c in need:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[need])
    df = pd.concat(frames, ignore_index=True)
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df[df["network_era"] == "B"]
    df = df[df["name_cn"].isin(set(_bj_names()))]
    _rank = {"observed": 0, "missing_value": 1, "out_of_range": 2}
    df = (
        df.assign(_r=df["value_status"].map(_rank).fillna(3))
        .sort_values(["name_cn", "hour_start_local", "_r"])
        .drop_duplicates(["name_cn", "hour_start_local"], keep="first")
        .drop(columns="_r")
    )
    out = {}
    for name, g in df.groupby("name_cn"):
        g = g.sort_values("hour_start_local")
        pm = g["pm25"].to_numpy(dtype=float)
        # 前向填充，保证窗口内无 NaN（尾部的 NaN 截断）
        pm = pd.Series(pm).ffill().to_numpy(dtype=float)
        out[name] = (list(g["hour_start_local"]), pm)
    return out


def _cal(t: pd.Timestamp) -> tuple[float, float, float, float]:
    h = t.hour + t.minute / 60.0
    d = t.dayofweek
    return (math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
            math.sin(2 * math.pi * d / 7), math.cos(2 * math.pi * d / 7))


def build_dataset(series: dict[str, tuple], horizons=(1, 3, 6), max_train=None):
    """向量化构建滑动窗口样本 (X, station_id, y_norm, y_orig, b1_orig, split, ...)。

    逐站用 sliding_window_view 生成窗口，再按 horizon 取目标；纯 NumPy，无逐样本 Python 循环。
    """
    names = list(series.keys())
    name2id = {n: i for i, n in enumerate(names)}

    Xs, SIDs, Ys, B1s, SPs, ITs, TSt, NMs = [], [], [], [], [], [], [], []
    for name in names:
        times, pm = series[name]
        n = len(pm)
        if n < SEQ + max(horizons) + 1:
            continue
        tarr = np.array(times, dtype="datetime64[ns]")
        tidx = pd.DatetimeIndex(tarr)
        hrs = tarr.astype("datetime64[h]").astype(float) % 24 + (tarr.astype("datetime64[m]").astype(float) % 60) / 60.0
        dows = tidx.dayofweek.to_numpy()
        hs = np.sin(2 * np.pi * hrs / 24); hc = np.cos(2 * np.pi * hrs / 24)
        ds = np.sin(2 * np.pi * dows / 7); dc = np.cos(2 * np.pi * dows / 7)
        feat = np.empty((n, N_FEATURES), dtype=np.float32)
        feat[:, 0] = pm
        feat[:, 1] = hs; feat[:, 2] = hc; feat[:, 3] = ds; feat[:, 4] = dc

        # sliding_window_view(feat, SEQ, axis=0) 得到 (n-SEQ+1, 5, SEQ)：
        # 窗口维被追加到最后。需转置为 (n-SEQ+1, SEQ, 5) 以匹配 LSTM/TCN 的 (batch, seq, feat)。
        _win = np.lib.stride_tricks.sliding_window_view(feat, SEQ, axis=0)  # (m, 5, SEQ)
        win = np.ascontiguousarray(_win.transpose(0, 2, 1))                # (m, SEQ, 5)
        issue = np.arange(SEQ - 1, n)  # 对应 win 的第 0..(n-SEQ) 行
        nanflag = np.isnan(feat[:, 0]).astype(np.int8)
        winnan = np.lib.stride_tricks.sliding_window_view(nanflag, SEQ)
        winnan_sum = winnan.sum(axis=1)  # 与 issue 对齐

        for h in horizons:
            tgt = issue + h
            valid = tgt < n
            ok = (winnan_sum == 0) & (~nanflag[issue]) & valid
            idx = np.where(ok)[0]
            if len(idx) == 0:
                continue
            abs_i = issue[idx]
            tgt_i = abs_i + h
            Xs.append(win[idx])
            SIDs.append(np.full(len(idx), name2id[name], dtype=np.int64))
            Ys.append(pm[tgt_i].astype(np.float32))
            B1s.append(pm[abs_i].astype(np.float32))
            ti = tidx[abs_i]
            sp = np.where(ti < TRAIN_END, "train", np.where(ti < VAL_END, "val", "test"))
            SPs.append(sp)
            ITs.append(tarr[abs_i])
            TSt.append((tidx[abs_i] + pd.Timedelta(hours=h)).to_numpy())
            NMs.append(np.full(len(idx), name))

    X = np.concatenate(Xs, axis=0)
    sid = np.concatenate(SIDs)
    yorig = np.concatenate(Ys)
    b1 = np.concatenate(B1s)
    split = np.concatenate(SPs)
    itime = np.concatenate(ITs)
    tstart = np.concatenate(TSt)
    nname = np.concatenate(NMs)
    ynorm = yorig.copy()

    mean = float(yorig[split == "train"].mean())
    std = float(yorig[split == "train"].std()) + 1e-6
    X[..., 0] = (X[..., 0] - mean) / std
    ynorm = (ynorm - mean) / std

    tr = split == "train"
    if max_train and tr.sum() > max_train:
        idx = np.where(tr)[0]
        keep = np.random.RandomState(0).choice(idx, max_train, replace=False)
        mask = np.zeros(len(split), dtype=bool)
        mask[keep] = True
        mask[~tr] = True
        X, sid, ynorm, yorig, b1, split, itime, tstart, nname = (
            X[mask], sid[mask], ynorm[mask], yorig[mask], b1[mask],
            split[mask], itime[mask], tstart[mask], nname[mask])

    return {
        "X": X, "sid": sid, "ynorm": ynorm, "yorig": yorig, "b1": b1,
        "split": split, "itime": itime, "tstart": tstart, "nname": nname,
        "mean": mean, "std": std, "n_stations": len(names),
    }


def make_loaders(d, h, batch=2048):
    m = (d["tstart"] - d["itime"]).astype("timedelta64[h]").astype(int) == h
    parts = {}
    for sp in ("train", "val", "test"):
        sel = m & (d["split"] == sp)
        parts[sp] = TensorDataset(
            torch.from_numpy(d["X"][sel]), torch.from_numpy(d["sid"][sel]),
            torch.from_numpy(d["ynorm"][sel]), torch.from_numpy(d["yorig"][sel]),
            torch.from_numpy(d["b1"][sel]),
        )
    return {sp: DataLoader(ds, batch_size=batch, shuffle=(sp == "train")) for sp, ds in parts.items()}, m


class LSTMModel(nn.Module):
    def __init__(self, n_stations, hidden=64, emb=8, layers=2):
        super().__init__()
        self.emb = nn.Embedding(n_stations, emb)
        self.lstm = nn.LSTM(N_FEATURES, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + emb, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, sid):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        e = self.emb(sid)
        return self.head(torch.cat([h, e], dim=1)).squeeze(-1)


class TCNModel(nn.Module):
    def __init__(self, n_stations, emb=8, channels=(32, 32, 32), kernel=3):
        super().__init__()
        self.emb = nn.Embedding(n_stations, emb)
        layers = []
        in_ch = N_FEATURES
        for c in channels:
            layers += [nn.Conv1d(in_ch, c, kernel, padding=kernel // 2), nn.ReLU()]
            in_ch = c
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(channels[-1] + emb, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, sid):
        y = self.net(x.transpose(1, 2))
        y = y.mean(dim=2)
        e = self.emb(sid)
        return self.head(torch.cat([y, e], dim=1)).squeeze(-1)


def train_one(model, loaders, mean, std, epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    for ep in range(epochs):
        model.train()
        for x, s, y, _, _ in loaders["train"]:
            x, s, y = x.to(device), s.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, s), y)
            loss.backward()
            opt.step()
        # val
        model.eval()
        errs = []
        with torch.no_grad():
            for x, s, y, _, _ in loaders["val"]:
                x, s, y = x.to(device), s.to(device), y.to(device)
                errs.append((model(x, s) - y).abs().mean().item())
        print(f"   epoch {ep+1}/{epochs}  val MAE(norm)={np.mean(errs):.4f}")


@torch.no_grad()
def evaluate(model, loader_test, mean, std, device):
    model.eval()
    preds, yt, b1 = [], [], []
    for x, s, _, yo, b in loader_test:
        x, s = x.to(device), s.to(device)
        p = model(x, s).cpu().numpy() * std + mean
        preds.append(p)
        yt.append(yo.numpy())
        b1.append(b.numpy())
    preds = np.concatenate(preds); yt = np.concatenate(yt); b1 = np.concatenate(b1)
    mae = float(np.abs(preds - yt).mean())
    rmse = float(np.sqrt(((preds - yt) ** 2).mean()))
    b1mae = float(np.abs(b1 - yt).mean())
    win = float((np.abs(preds - yt) < np.abs(b1 - yt)).mean())
    vs = (b1mae - mae) / b1mae * 100 if b1mae > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "B1_MAE": b1mae, "vs_B1": vs, "win_rate": win}, preds, yt, b1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "tcn"], default="lstm")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--max-train", type=int, default=500000)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M5] model={args.model} device={device}")

    print("[M5] loading + building dataset ...")
    ds = build_dataset(load_series(), horizons=tuple(args.horizons), max_train=args.max_train)
    print(f"      samples={len(ds['X'])}  stations={ds['n_stations']}  "
          f"train/val/test={ (ds['split']=='train').sum() }/{ (ds['split']=='val').sum() }/{ (ds['split']=='test').sum() }")

    run_id = f"{pd.Timestamp.now():%Y%m%dT%H%M%S}_dl_{args.model}"
    out_dir = os.path.join(DATA, "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    metrics_rows = []
    import pyarrow as pa

    for h in args.horizons:
        print(f"[M5] horizon h={h}")
        loaders, mask = make_loaders(ds, h, batch=args.batch)
        n_st = ds["n_stations"]
        model = (LSTMModel(n_st, hidden=args.hidden) if args.model == "lstm"
                 else TCNModel(n_st))
        model.to(device)
        train_one(model, loaders, ds["mean"], ds["std"], args.epochs, device)
        res, preds, yt, b1 = evaluate(model, loaders["test"], ds["mean"], ds["std"], device)
        n = int(mask.sum())
        metrics_rows.append({"h": h, "split": "test", "n": n, **res})
        print(f"      test MAE={res['MAE']:.3f}  B1={res['B1_MAE']:.3f}  vs_B1={res['vs_B1']:+.1f}%  win={res['win_rate']:.3f}")

        # 落盘预测（仅 test），供演示对比页
        sel = mask & (ds["split"] == "test")
        tbl = pa.table({
            "station_name": ds["nname"][sel],
            "issue_time": ds["itime"][sel].astype("datetime64[us]"),
            "target_start": ds["tstart"][sel].astype("datetime64[us]"),
            "y_true": yt, "y_pred": preds, "b1_persistence": b1,
        })
        pq.write_table(tbl, os.path.join(out_dir, f"pred_h{h}_test.parquet"))

        torch.save(model.state_dict(), os.path.join(out_dir, f"model_h{h}.pt"))

    pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "model": args.model, "epochs": args.epochs,
                   "horizons": args.horizons, "mean": ds["mean"], "std": ds["std"],
                   "device": str(device)}, f, ensure_ascii=False, indent=2)
    print(f"[M5] done -> {out_dir}")


if __name__ == "__main__":
    main()
