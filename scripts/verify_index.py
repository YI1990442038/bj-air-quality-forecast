#!/usr/bin/env python3
"""索引手算对账：完全绕过索引构建代码，从原始日文件独立复算。

做法：直接读 data/raw/beijing_all/YYYYMMDD.csv 的单元格，
按契约手算 y_true / B1 / B2，再与 data/index 里的值逐项比对。
若两边一致，说明 [T+h-1, T+h) 的定义、shift 方向、昨日同期偏移都正确。

任何一个不一致都会以非零退出码结束。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aircast import protocol as P  # noqa: E402

RAW = REPO / "data" / "raw" / "beijing_all"
IDX = REPO / "data" / "index" / f"protocol={P.PROTOCOL_ID}"


def raw_value(station: str, ts: pd.Timestamp, want_type: str = "PM2.5"):
    """从原始日文件读某站某小时的 PM2.5 原始单元格。"""
    path = RAW / f"{ts:%Y%m%d}.csv"
    if not path.exists():
        return None, "file_missing"
    with path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["type"] != want_type:
                continue
            if int(row["hour"]) != ts.hour:
                continue
            cell = (row.get(station) or "").strip()
            if cell in ("", "-", "NA", "N/A"):
                return None, "empty_cell"
            return float(cell), "ok"
    return None, "hour_absent"


def index_row(station: str, ts: pd.Timestamp, h: int):
    f = IDX / f"h={h}" / "part.parquet"
    df = pd.read_parquet(f, columns=["station_name", "issue_time", "h", "y_true", "b1_persistence", "b2_yesterday", "split"])
    m = df[(df["station_name"] == station) & (df["issue_time"] == ts)]
    if m.empty:
        return None
    return m.iloc[0]


def check_case(station: str, T: pd.Timestamp):
    print(f"\n{'='*78}\n对账：站点 {station}　发布时点 T = {T:%Y-%m-%d %H:%M}（北京时间）\n{'='*78}")
    print(f"{'h':>2} {'目标小时':<17}{'手算 y_true':>12}{'索引 y_true':>12}{'手算 B1':>10}{'索引 B1':>10}{'手算 B2':>10}{'索引 B2':>10}  判定")

    failures = 0
    for h in P.HORIZONS:
        row = index_row(station, T, h)
        if row is None:
            print(f"{h:>2} 索引中缺少该行")
            failures += 1
            continue

        t_start, t_end = P.target_window(T, h)
        assert t_end - t_start == pd.Timedelta(hours=1)

        v_true, s_true = raw_value(station, t_start)
        v_b1, s_b1 = raw_value(station, P.persistence_hour(T))
        v_b2, s_b2 = raw_value(station, P.yesterday_hour(T, h))

        def cmp(a, b):
            if a is None and pd.isna(b):
                return True
            if a is None or pd.isna(b):
                return False
            return abs(a - b) < 1e-4

        ok = cmp(v_true, row["y_true"]) and cmp(v_b1, row["b1_persistence"]) and cmp(v_b2, row["b2_yesterday"])
        failures += 0 if ok else 1

        fmt = lambda v: "缺失" if v is None else f"{v:.0f}"
        fmti = lambda v: "缺失" if pd.isna(v) else f"{v:.0f}"
        print(
            f"{h:>2} {t_start:%m-%d %H:%M}—{t_end:%H:%M}  "
            f"{fmt(v_true):>10}{fmti(row['y_true']):>13}{fmt(v_b1):>11}{fmti(row['b1_persistence']):>11}"
            f"{fmt(v_b2):>11}{fmti(row['b2_yesterday']):>11}  {'✓' if ok else '✗ 不一致'}"
        )
        if not ok:
            print(f"     原始状态: y_true={s_true} b1={s_b1} b2={s_b2}")
    return failures


def check_no_leakage(station: str, T: pd.Timestamp):
    """确认 B1 取数时刻恰为 T-1，且不晚于允许的观测边界。"""
    print("\n泄漏检查：")
    ok = True
    for h in P.HORIZONS:
        row = index_row(station, T, h)
        if row is None:
            print(f"  h={h} 缺行，跳过")
            ok = False
            continue
        limit = P.last_observed_hour(T)
        v_b1, _ = raw_value(station, limit)
        same = (pd.isna(row["b1_persistence"]) and v_b1 is None) or (
            v_b1 is not None and abs(row["b1_persistence"] - v_b1) < 1e-4
        )
        assert_limit = limit == T - pd.Timedelta(hours=1)
        print(
            f"  h={h}: B1 应取 T-1 = {limit:%m-%d %H:%M}，索引一致={same}，T-1 定义正确={assert_limit}"
        )
        ok = ok and same and assert_limit
    return 0 if ok else 1


def main():
    cases = [
        ("东城东四", pd.Timestamp("2026-06-15 10:00")),
        ("朝阳奥体中心", pd.Timestamp("2025-11-03 07:00")),
        ("海淀万柳", pd.Timestamp("2022-03-18 22:00")),
    ]
    total_fail = 0
    for station, T in cases:
        total_fail += check_case(station, T)
    total_fail += check_no_leakage(*cases[0])

    print(f"\n{'='*78}")
    if total_fail:
        print(f"对账失败：{total_fail} 项不一致")
        raise SystemExit(1)
    print("对账通过：全部手算值与索引一致")


if __name__ == "__main__":
    main()
