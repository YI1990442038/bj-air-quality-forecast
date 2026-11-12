"""时间与切分契约（protocol v1）。

全项目唯一的时间定义来源。任何模型、特征、评价都必须从这里取定义，禁止各自实现。

契约要点
--------
1. 时间轴用北京时间的「小时起点」表示，区间左闭右开：hour = [t, t+1)。
2. 发布时点 T 为整点。可用观测只到 T-1 这一小时（即 [T-1, T)）。
3. 提前 h 小时的目标是**单个小时** [T+h-1, T+h)，即 hour_start = T + (h-1)。
   三个跨度是三个独立目标，不是一条连续 h 小时曲线。
4. 基线 B1 用 T-1 的观测；B2 用目标小时减 24 小时（昨日同期）的观测。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

PROTOCOL_ID = "p1"
HORIZONS = (1, 3, 6)

# 切分点一律为北京时间零点，左闭右开 [start, end)。
# warmup 预留 15 天，用于 lag-24 基线与 7 日滚动特征的预热，不参与训练与评价。
SPLITS = {
    "warmup": ("2021-01-24", "2021-02-08"),
    "train": ("2021-02-08", "2025-01-01"),
    "val": ("2025-01-01", "2025-07-01"),
    "test": ("2025-07-01", "2026-10-30"),
}

# 建模只用纪元 B：2021-01-24 前后是两套站网，属性不可直接拼接。
ERA = "B"

# 目标变量可用性下限。设 0.75 是为了保留全部 35 站：实测最低两站为
# 门头沟三家店 80.2%、昌平南邵 81.6%，其余均 ≥95.5%。整站剔除会让时空图
# 少一个节点，代价大于该站 20% 样本缺失；缺失行在索引里已标 NaN，评价时按可用行计。
MIN_STATION_COVERAGE = 0.75


def target_hour(issue_time, h: int):
    """返回目标小时的起点，即 T + (h-1)。"""
    _check_h(h)
    return issue_time + timedelta(hours=h - 1)


def target_window(issue_time, h: int):
    """返回目标窗口 (start, end)，左闭右开，长度恒为 1 小时。"""
    s = target_hour(issue_time, h)
    return s, s + timedelta(hours=1)


def last_observed_hour(issue_time):
    """发布时点可用的最后一个完整观测小时，即 T-1。"""
    return issue_time - timedelta(hours=1)


def persistence_hour(issue_time):
    """B1 持久性取数时刻：T-1。"""
    return last_observed_hour(issue_time)


def yesterday_hour(issue_time, h: int):
    """B2 昨日同期取数时刻：目标小时减 24 小时。"""
    return target_hour(issue_time, h) - timedelta(hours=24)


def split_of(ts) -> str | None:
    """判断某个时刻属于哪个切分；不在任何区间返回 None。"""
    t = pd.Timestamp(ts)
    for name, (a, b) in SPLITS.items():
        if pd.Timestamp(a) <= t < pd.Timestamp(b):
            return name
    return None


def assert_no_leakage(feature_max_time, issue_time):
    """特征能用到的最晚时刻必须早于发布时点。

    允许等于 T-1 的整点（即 [T-1, T) 这一小时），不允许触及 T 及以后。
    """
    if feature_max_time is None:
        return
    if pd.Timestamp(feature_max_time) > last_observed_hour(pd.Timestamp(issue_time)):
        raise AssertionError(
            f"特征泄漏：特征最晚时刻 {feature_max_time} 晚于允许的 {last_observed_hour(pd.Timestamp(issue_time))}"
        )


def _check_h(h: int):
    if h not in HORIZONS:
        raise ValueError(f"h 必须是 {HORIZONS} 之一，收到 {h}")


def describe() -> str:
    lines = [f"protocol {PROTOCOL_ID} | 纪元 {ERA} | 跨度 {HORIZONS}", ""]
    lines.append(f"{'阶段':<8}{'起':<14}{'止':<14}{'天数':>6}")
    for name, (a, b) in SPLITS.items():
        d = (pd.Timestamp(b) - pd.Timestamp(a)).days
        lines.append(f"{name:<8}{a:<14}{b:<14}{d:>6}")
    lines += [
        "",
        "目标定义：h=1 → T+0，h=3 → T+2，h=6 → T+5（均为单个小时，左闭右开）",
        "基线：B1 = y(T-1)；B2 = y(目标小时 - 24h)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    t = pd.Timestamp("2026-06-15 10:00")
    for h in HORIZONS:
        s, e = target_window(t, h)
        print(f"  T={t:%H:%M} h={h} → 目标 [{s:%m-%d %H:%M}, {e:%H:%M})  B1={persistence_hour(t):%H:%M}  B2={yesterday_hour(t, h):%m-%d %H:%M}")
