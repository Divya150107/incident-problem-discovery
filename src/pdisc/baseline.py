"""Phase 1 baseline: recurrence, reopen, resolution and SLA levels before any grouping.

These numbers are the reference that candidate signals are compared against
("elevated vs baseline"), and they feed the dashboard's trend pages.
"""
from __future__ import annotations

import pandas as pd

SIGNALS = ["is_reopened", "is_sla_missed", "is_bounced", "is_high_priority"]


def _sum(s: pd.Series):
    # a field the source lacks stays blank (NaN), not 0
    return s.astype(float).sum(min_count=1)


def global_rates(inc: pd.DataFrame) -> dict[str, float]:
    rates = {s: float(inc[s].astype(float).mean()) for s in SIGNALS}
    rates["median_resolution_hours"] = float(inc["resolution_hours"].median())
    rates["p90_resolution_hours"] = float(inc["resolution_hours"].quantile(0.9))
    return rates


def summary_table(inc: pd.DataFrame) -> pd.DataFrame:
    r = global_rates(inc)
    rows = [
        ("Incidents in window", len(inc)),
        ("Distinct service signatures (category/subcategory/symptom)", inc["signature"].nunique()),
        ("Signatures with >= 15 incidents", int((inc["signature"].value_counts() >= 15).sum())),
        ("Reopened incidents %", round(100 * r["is_reopened"], 2)),
        ("SLA missed %", round(100 * r["is_sla_missed"], 2)),
        ("Reassigned 3+ times %", round(100 * r["is_bounced"], 2)),
        ("High/critical priority %", round(100 * r["is_high_priority"], 2)),
        ("Median resolution hours", round(r["median_resolution_hours"], 1)),
        ("P90 resolution hours", round(r["p90_resolution_hours"], 1)),
        ("Incidents linked to a known Problem", int(inc["has_known_problem"].sum())),
        ("Known Problems referenced", int(inc["problem_id"].nunique())),
        ("Incidents linked to a change (RFC)", int(inc["has_change"].sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def weekly_by_category(inc: pd.DataFrame) -> pd.DataFrame:
    g = inc.groupby(["opened_week", "category"])
    return g.agg(
        incidents=("number", "size"),
        reopened=("is_reopened", _sum),
        sla_missed=("is_sla_missed", _sum),
        bounced=("is_bounced", _sum),
        high_priority=("is_high_priority", _sum),
        median_resolution_hours=("resolution_hours", "median"),
    ).reset_index()
