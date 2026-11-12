#!/usr/bin/env python3
"""数据获取质量核验：产出可交差的数据清单与缺口台账。

输出：
  data/reports/qc_summary.md   人类可读汇总
  data/reports/gaps_<ds>.csv   每日缺口明细
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATASETS = ["beijing_all", "beijing_extra", "china_cities", "china_sites"]


def daterange(s: str, e: str):
    d = datetime.strptime(s, "%Y%m%d").date()
    end = datetime.strptime(e, "%Y%m%d").date()
    while d <= end:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


def load(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["date"]] = r
    return out


def main():
    repo = Path(__file__).resolve().parent.parent
    man = repo / "data" / "manifests"
    rep = repo / "data" / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    lines = ["# 数据获取质量核验", "", f"生成时间：{datetime.now():%Y-%m-%d %H:%M}", ""]
    lines.append("| 数据集 | 覆盖区间 | 应有天数 | 已获取 | 完整 | 部分缺失 | 源端无文件 | 失败 | 完整率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for ds in DATASETS:
        recs = load(man / f"{ds}.jsonl")
        if not recs:
            continue
        dates = sorted(recs)
        start, end = dates[0], dates[-1]
        n_expect = len(list(daterange(start, end)))
        c = Counter(r.get("status") for r in recs.values())
        ok = c.get("ok", 0)
        partial = c.get("partial", 0)
        miss = c.get("missing_source", 0)
        fail = c.get("failed", 0)
        rate = ok / n_expect if n_expect else 0
        lines.append(
            f"| {ds} | {start}—{end} | {n_expect} | {len(recs)} | {ok} | {partial} | {miss} | {fail} | {rate:.2%} |"
        )

        rows = []
        for d in daterange(start, end):
            r = recs.get(d)
            if r is None:
                rows.append({"date": d, "issue": "not_downloaded", "missing_hours": "", "detail": ""})
                continue
            st = r.get("status")
            if st in ("partial", "failed", "missing_source"):
                rows.append(
                    {
                        "date": d,
                        "issue": st,
                        "missing_hours": ",".join(map(str, r.get("missing_hours") or [])),
                        "detail": r.get("reason") or ",".join(r.get("missing_types") or []),
                    }
                )
        if rows:
            pd.DataFrame(rows).to_csv(rep / f"gaps_{ds}.csv", index=False, encoding="utf-8")

    # 站点小时覆盖（仅北京两类）
    lines += ["", "## 站点小时覆盖", ""]
    for ds, nst in (("beijing_all", 35), ("beijing_extra", 35)):
        recs = load(man / f"{ds}.jsonl")
        if not recs:
            continue
        site_hours = 0
        for r in recs.values():
            if r.get("status") in ("ok", "partial"):
                nh = r.get("n_hours")
                if nh is None:
                    nh = 24 if r.get("rows") == 120 else round((r.get("rows") or 0) / 5)
                site_hours += int(nh) * nst
        full = len(list(daterange(min(recs), max(recs)))) * 24 * nst
        lines.append(f"- `{ds}`：约 **{site_hours:,}** 站点小时 / 理论上限 {full:,}，实得 {site_hours/full:.2%}")

    # 仓库层总账（含无清单的 china_sites，它来自官方年度包）
    wh = repo / "data" / "warehouse" / "fact_obs"
    if wh.exists():
        import pyarrow.parquet as pq

        lines += ["", "## 仓库总账（Parquet）", ""]
        lines.append("| 数据集 | 年分区 | 行数 | 体积 |")
        lines.append("|---|---|---|---|")
        grand = 0
        for d in sorted(wh.glob("source=*")):
            files = sorted(d.glob("year=*/part.parquet"))
            if not files:
                continue
            n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
            sz = sum(f.stat().st_size for f in files)
            grand += n
            lines.append(f"| {d.name.replace('source=', '')} | {len(files)} | {n:,} | {sz/1e6:.1f} MB |")
        lines.append(f"| **合计** | | **{grand:,}** | |")

    # 变量覆盖率（北京）
    cov_path = repo / "data" / "meta" / "meteo_coverage.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        vcols = [c for c in cov.columns if c not in ("point_id", "year")]
        lines += ["", "## 气象变量覆盖率（按点年最低值）", ""]
        lines.append("| 变量 | 最低覆盖率 | 平均 |")
        lines.append("|---|---|---|")
        for c in vcols:
            lines.append(f"| {c} | {cov[c].min():.1%} | {cov[c].mean():.1%} |")

    out = rep / "qc_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
