#!/usr/bin/env python3
"""评价：日块 bootstrap 置信区间、q95 峰值检出、分组失败模式。

用法
  python scripts/evaluate.py --run data/runs/20261031T190310_lgbm_v1
产物
  {run}/eval_report.md          人读报告
  {run}/eval_bootstrap.csv      置信区间
  {run}/eval_peak.csv           峰值 P/R/F1
  {run}/eval_group_*.csv        分组结果
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aircast import metrics as M  # noqa: E402
from aircast import protocol as P  # noqa: E402

GROUPS = ["station_name", "season", "daypart", "level"]


def train_truth(h: int) -> pd.DataFrame:
    f = REPO / "data" / "index" / f"protocol={P.PROTOCOL_ID}" / f"h={h}" / "part.parquet"
    df = pd.read_parquet(f, columns=["station_name", "split", "y_true"])
    return df[df["split"] == "train"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = REPO / run
    md = [f"# 评价报告：{run.name}", ""]

    boot_rows, peak_rows, group_rows = [], [], []
    for h in P.HORIZONS:
        pf = run / f"pred_h{h}_{args.split}.parquet"
        if not pf.exists():
            print(f"缺少 {pf}")
            continue
        df = pd.read_parquet(pf)

        b = M.bootstrap_gain(df)
        b.update({"h": h, "split": args.split})
        boot_rows.append(b)

        thr = M.station_thresholds(train_truth(h))
        pk = M.peak_prf(df, thr)
        pk["h"], pk["split"] = h, args.split
        peak_rows.append(pk)

        for g in GROUPS:
            ga = M.group_analysis(df, g)
            ga["h"], ga["split"], ga["group_by"] = h, args.split, g
            group_rows.append(ga)

        fs = M.failure_summary(df)
        md.append(f"## h = {h}")
        md.append("")
        md.append(f"日块 bootstrap（{b['n_days']} 个自然日，2000 次重采样）：")
        md.append("")
        md.append("| 指标 | 值 |")
        md.append("|---|---|")
        md.append(f"| LGBM MAE | **{b['MAE']}** (95% CI {b['MAE_ci'][0]}–{b['MAE_ci'][1]}) |")
        md.append(f"| B1 MAE | {b['B1_MAE']} |")
        md.append(f"| 提升 | **{b['gain_pct']:+.2f}%** (95% CI {b['gain_ci'][0]:+.2f}% – {b['gain_ci'][1]:+.2f}%) |")
        md.append(f"| P(提升≤0) | {b['p_gain_le_0']:.4f} |")
        md.append(f"| 显著（CI 下界>0） | {'是' if b['significant'] else '否'} |")
        md.append("")
        md.append("峰值检出（阈值 = 各站训练期 q95）：")
        md.append("")
        md.append(pk[["model", "n_event", "P_micro", "R_micro", "F1_micro", "P_macro", "R_macro", "F1_macro"]]
                  .to_markdown(index=False))
        md.append("")
        md.append(
            f"失败模式：模型比基线差的样本占 {fs['share_worse']:.1%}（{fs['n_worse_than_base']:,} 条），"
            f"却贡献了 {fs['err_share_from_worse']:.1%} 的总绝对误差；"
            f"这些样本真值均值 {fs['mean_true_when_worse']}，全体均值 {fs['mean_true_overall']} μg/m³。"
        )
        md.append("")

    if not boot_rows:
        raise SystemExit("没有可评价的预测文件")

    pd.DataFrame(boot_rows).to_csv(run / "eval_bootstrap.csv", index=False, encoding="utf-8")
    pd.concat(peak_rows, ignore_index=True).to_csv(run / "eval_peak.csv", index=False, encoding="utf-8")
    gr = pd.concat(group_rows, ignore_index=True)
    for g in GROUPS:
        gr[gr["group_by"] == g].to_csv(run / f"eval_group_{g}.csv", index=False, encoding="utf-8")

    # 汇总表放最前
    bt = pd.DataFrame(boot_rows)
    head = ["| h | LGBM MAE | B1 MAE | 提升 | 95% CI | P(≤0) | 显著 |", "|---|---|---|---|---|---|---|"]
    for _, r in bt.iterrows():
        head.append(
            f"| {r['h']} | {r['MAE']} | {r['B1_MAE']} | {r['gain_pct']:+.2f}% | "
            f"{r['gain_ci'][0]:+.2f}% – {r['gain_ci'][1]:+.2f}% | {r['p_gain_le_0']:.4f} | "
            f"{'✅' if r['significant'] else '❌'} |"
        )
    md = md[:2] + ["## 汇总（test 段）", ""] + head + [""] + md[2:]
    (run / "eval_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(head[2:]))
    print(f"\n报告 → {run / 'eval_report.md'}")


if __name__ == "__main__":
    main()
