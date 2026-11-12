"""评价指标：日块 bootstrap 置信区间、峰值检出、分组失败模式。

设计要点
--------
bootstrap 的采样单位是**北京自然日整块**，不是行。小时样本自相关极强（1 小时自相关 0.974），
按行重采样会把有效样本量放大几十倍，置信区间假性收窄。
实现上不重采样行，而是按天聚合「绝对误差和」与「计数」——MAE 是可加量，
这样 2000 次重采样既与逐行重采样完全等价，又只需毫秒级计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POLLUTION_BINS = [0, 35, 75, 115, 150, 250, np.inf]
POLLUTION_LABELS = ["优(<35)", "良(35-75)", "轻度(75-115)", "中度(115-150)", "重度(150-250)", "严重(>250)"]
SEASON_MAP = {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
              6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}


def _prep(df: pd.DataFrame, pred_col: str, base_col: str, true_col: str = "y_true") -> pd.DataFrame:
    d = df.dropna(subset=[true_col, pred_col, base_col]).copy()
    d["err_m"] = (d[pred_col] - d[true_col]).abs()
    d["err_b"] = (d[base_col] - d[true_col]).abs()
    d["day"] = d["target_start"].dt.normalize()
    d["season"] = d["target_start"].dt.month.map(SEASON_MAP)
    h = d["target_start"].dt.hour
    d["daypart"] = np.where((h >= 6) & (h <= 18), "昼(06-18)", "夜(19-05)")
    d["level"] = pd.cut(d[true_col], bins=POLLUTION_BINS, labels=POLLUTION_LABELS, right=False)
    return d


def bootstrap_gain(df: pd.DataFrame, pred_col="y_pred", base_col="b1_persistence",
                   n_boot: int = 2000, seed: int = 42) -> dict:
    """按自然日整块重采样的置信区间。返回 MAE、B1 MAE、提升幅度及其 95% CI。"""
    d = _prep(df, pred_col, base_col)
    if d.empty:
        return {}
    g = d.groupby("day", observed=True)
    sm = g["err_m"].sum().to_numpy()
    sb = g["err_b"].sum().to_numpy()
    cnt = g.size().to_numpy()
    n = len(cnt)

    mae_m = sm.sum() / cnt.sum()
    mae_b = sb.sum() / cnt.sum()
    gain = (1 - mae_m / mae_b) * 100

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ma = sm[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    mb = sb[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    gs = (1 - ma / mb) * 100

    return {
        "n": int(cnt.sum()),
        "n_days": int(n),
        "MAE": round(float(mae_m), 3),
        "MAE_ci": (round(float(np.percentile(ma, 2.5)), 3), round(float(np.percentile(ma, 97.5)), 3)),
        "B1_MAE": round(float(mae_b), 3),
        "gain_pct": round(float(gain), 2),
        "gain_ci": (round(float(np.percentile(gs, 2.5)), 2), round(float(np.percentile(gs, 97.5)), 2)),
        "p_gain_le_0": round(float((gs <= 0).mean()), 4),
        "significant": bool(np.percentile(gs, 2.5) > 0),
    }


def station_thresholds(df: pd.DataFrame, q: float = 0.95, true_col="y_true") -> pd.Series:
    """每站阈值取**训练期**真值的 q 分位。必须只在 train 上拟合，否则阈值会看到未来。"""
    return df.dropna(subset=[true_col]).groupby("station_name", observed=True)[true_col].quantile(q)


def peak_prf(df: pd.DataFrame, thr: pd.Series, pred_col="y_pred",
             base_col="b1_persistence", true_col="y_true") -> pd.DataFrame:
    """按站阈值判峰值，返回微平均与宏平均的 Precision / Recall / F1。"""
    d = df.dropna(subset=[true_col, pred_col]).copy()
    d["thr"] = d["station_name"].map(thr)
    d = d.dropna(subset=["thr"])
    rows = []
    for name, col in (("LGBM", pred_col), ("B1", base_col)):
        sub = d.dropna(subset=[col])
        act = sub[true_col] >= sub["thr"]
        pre = sub[col] >= sub["thr"]
        tp = int((act & pre).sum())
        fp = int((~act & pre).sum())
        fn = int((act & ~pre).sum())
        p = tp / (tp + fp) if tp + fp else np.nan
        r = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * p * r / (p + r) if p and r and not np.isnan(p) and not np.isnan(r) else np.nan

        # 宏平均：先按站算再平均，避免大站主导
        per = []
        for st, gg in sub.groupby("station_name", observed=True):
            a = gg[true_col] >= gg["thr"]
            pr = gg[col] >= gg["thr"]
            t = int((a & pr).sum())
            fpn = int((~a & pr).sum())
            fnn = int((a & ~pr).sum())
            pp = t / (t + fpn) if t + fpn else np.nan
            rr = t / (t + fnn) if t + fnn else np.nan
            per.append((pp, rr))
        arr = np.array(per, dtype=float)
        mp = np.nanmean(arr[:, 0]) if len(arr) else np.nan
        mr = np.nanmean(arr[:, 1]) if len(arr) else np.nan
        mf = 2 * mp * mr / (mp + mr) if mp and mr else np.nan

        rows.append({
            "model": name, "n": len(sub), "n_event": int(act.sum()),
            "TP": tp, "FP": fp, "FN": fn,
            "P_micro": round(p, 4), "R_micro": round(r, 4), "F1_micro": round(f1, 4),
            "P_macro": round(float(mp), 4), "R_macro": round(float(mr), 4), "F1_macro": round(float(mf), 4),
        })
    return pd.DataFrame(rows)


def group_analysis(df: pd.DataFrame, by: str, pred_col="y_pred", base_col="b1_persistence") -> pd.DataFrame:
    """按指定维度拆开看模型与基线的表现，找出模型输给基线的场景。"""
    d = _prep(df, pred_col, base_col)
    rows = []
    for key, g in d.groupby(by, observed=True):
        mae_m = g["err_m"].mean()
        mae_b = g["err_b"].mean()
        rows.append({
            by: str(key),
            "n": len(g),
            "MAE": round(float(mae_m), 3),
            "B1_MAE": round(float(mae_b), 3),
            "gain_pct": round(float((1 - mae_m / mae_b) * 100), 2),
            "win_rate": round(float((g["err_m"] < g["err_b"]).mean()), 4),
        })
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


def failure_summary(df: pd.DataFrame, pred_col="y_pred", base_col="b1_persistence") -> dict:
    """整体失败模式：模型在哪些样本上比基线更差。"""
    d = _prep(df, pred_col, base_col)
    worse = d[d["err_m"] > d["err_b"]]
    worse_heavy = worse[worse["y_true"] >= 75]
    return {
        "n": len(d),
        "n_worse_than_base": len(worse),
        "share_worse": round(len(worse) / len(d), 4),
        "err_share_from_worse": round(float(worse["err_m"].sum() / d["err_m"].sum()), 4),
        "n_worse_heavy": len(worse_heavy),
        "mean_true_when_worse": round(float(worse["y_true"].mean()), 1) if len(worse) else None,
        "mean_true_overall": round(float(d["y_true"].mean()), 1),
    }
