"""Extraction and cleansing: raw ServiceNow event log -> one row per incident.

The UCI log stores every update of an incident as a separate row (141k rows for
~25k incidents). We keep the latest update per incident as its final state,
parse timestamps, drop personal identifiers, and derive the fields the later
stages need. Nothing here makes analytical judgements.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TS_FORMAT = "%d/%m/%Y %H:%M"
TS_COLS = ["opened_at", "sys_created_at", "sys_updated_at", "resolved_at", "closed_at"]

PRIORITY_RANK = {"1 - Critical": 1, "2 - High": 2, "3 - Moderate": 3, "4 - Low": 4}
IMPACT_RANK = {"1 - High": 1, "2 - Medium": 2, "3 - Low": 3}


def load_event_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?"], low_memory=False, dtype=str)
    for c in TS_COLS:
        df[c] = pd.to_datetime(df[c], format=TS_FORMAT, errors="coerce")
    for c in ["reassignment_count", "reopen_count", "sys_mod_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["active", "made_sla", "knowledge", "u_priority_confirmation"]:
        df[c] = df[c].str.lower().eq("true")
    return df


def build_incidents(events: pd.DataFrame, personal_columns: list[str]) -> pd.DataFrame:
    """Collapse the event log to the latest state of each incident."""
    ev = events.sort_values(["number", "sys_mod_count", "sys_updated_at"])
    # count distinct callers per signature later, so keep a pseudonymous caller key
    # (a hash), then drop all named-person columns.
    last = ev.groupby("number", as_index=False).tail(1).copy()
    n_events = ev.groupby("number").size().rename("event_count")
    ever_awaiting_problem = (
        ev.assign(ap=ev["incident_state"].eq("Awaiting Problem"))
        .groupby("number")["ap"].any().rename("ever_awaiting_problem")
    )
    last = last.join(n_events, on="number").join(ever_awaiting_problem, on="number")

    last["caller_key"] = pd.util.hash_pandas_object(last["caller_id"].fillna(""), index=False).astype(str)
    last = last.drop(columns=[c for c in personal_columns if c in last.columns])

    last["priority_rank"] = last["priority"].map(PRIORITY_RANK)
    last["impact_rank"] = last["impact"].map(IMPACT_RANK)
    last["is_high_priority"] = last["priority_rank"].le(2)
    last["is_high_impact"] = last["impact_rank"].eq(1)
    last["is_reopened"] = last["reopen_count"].gt(0)
    last["is_sla_missed"] = ~last["made_sla"]
    last["is_bounced"] = last["reassignment_count"].ge(3)   # ping-pong between groups
    last["is_resolved"] = last["resolved_at"].notna()
    hrs = (last["resolved_at"] - last["opened_at"]).dt.total_seconds() / 3600
    # negative durations exist in the source (clock / data entry errors) -> null, reported in quality
    last["resolution_hours"] = hrs.where(hrs >= 0)
    last["has_known_problem"] = last["problem_id"].notna()
    last["has_change"] = last["rfc"].notna()
    # service signature used throughout: category / subcategory / symptom
    last["u_symptom"] = last["u_symptom"].fillna("Symptom (blank)")
    last["category"] = last["category"].fillna("Category (blank)")
    last["subcategory"] = last["subcategory"].fillna("Subcategory (blank)")
    last["location"] = last["location"].fillna("Location (blank)")
    last["assignment_group"] = last["assignment_group"].fillna("Group (blank)")
    last["signature"] = last["category"] + " | " + last["subcategory"] + " | " + last["u_symptom"]
    last["opened_week"] = last["opened_at"].dt.to_period("W-SUN").dt.start_time
    last["opened_date"] = last["opened_at"].dt.normalize()
    return last.reset_index(drop=True)


def apply_window(inc: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into in-window incidents and those outside (reported, not silently dropped)."""
    mask = inc["opened_at"].between(pd.Timestamp(start), pd.Timestamp(end))
    return inc[mask].reset_index(drop=True), inc[~mask].reset_index(drop=True)


def quality_report(events: pd.DataFrame, inc: pd.DataFrame, out_of_window: pd.DataFrame) -> pd.DataFrame:
    """Documented data-quality findings: one row per check, with counts."""
    rows = []

    def add(area, check, count, total, note=""):
        rows.append({"area": area, "check": check, "count": int(count), "total": int(total),
                     "pct": round(100 * count / total, 2) if total else np.nan, "note": note})

    n = len(inc)
    # raw (pre-cleansing) final state of each in-window incident, for missing-value counts
    raw_last = (events[events["number"].isin(inc["number"])]
                .sort_values(["number", "sys_mod_count", "sys_updated_at"])
                .groupby("number").tail(1))
    add("coverage", "event rows in source", len(events), len(events))
    add("coverage", "incidents in source", len(inc) + len(out_of_window), len(inc) + len(out_of_window))
    add("coverage", "incidents outside analysis window", len(out_of_window), len(inc) + len(out_of_window),
        "sparse tail after May 2016; excluded from trend analysis")
    add("duplicates", "exact duplicate event rows", events.duplicated().sum(), len(events))
    add("integrity", "state value '-100' (invalid)", events["incident_state"].eq("-100").sum(), len(events))
    for col, note in [
        ("cmdb_ci", "configuration item almost never recorded; signature/location used as proxy"),
        ("u_symptom", "blank symptom kept as its own value"),
        ("assignment_group", ""),
        ("problem_id", "only these incidents can be matched to known Problems"),
        ("rfc", "only these incidents can be linked to a change"),
        ("caused_by", "change causation essentially unrecorded"),
        ("resolved_at", ""),
    ]:
        add("missing", f"{col} missing", raw_last[col].isna().sum(), n, note)
    add("integrity", "negative resolution time", ((inc["resolved_at"] - inc["opened_at"]).dt.total_seconds() < 0).sum(), n,
        "resolution_hours set to null")
    add("impact", "business-impact / service-criticality data", n, n,
        "not in source: impact uses ticket priority/impact fields only - flagged, not invented")
    return pd.DataFrame(rows)
