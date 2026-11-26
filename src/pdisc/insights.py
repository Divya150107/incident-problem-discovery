"""Source-specific operational insights, written as extra dashboard tables (insight_*.csv).

These describe the estate rather than detect candidates:
  UCI      - SLA breakdown, known-Problem links, change links
  Mendeley - resolution patterns (outcome mix, time per workflow step), handoffs

Computed on full (non-replay) runs only: some inputs, such as Mendeley's per-step
durations, are final-state totals and would leak future information into an as-of run.

Handoff analysis uses counts of distinct ticket holders only. Who the holders were is
never read into an output, and nothing is aggregated per person (the brief forbids
ranking individual employees).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rate(s: pd.Series) -> float:
    return float(s.astype(float).mean()) if len(s) else np.nan


def compare_groups(inc: pd.DataFrame, flag: str, labels=("linked", "not linked")) -> pd.DataFrame:
    """Outcome profile of incidents with vs without a flag."""
    rows = []
    for val, name in [(True, labels[0]), (False, labels[1])]:
        g = inc[inc[flag].astype(bool) == val]
        rows.append({"group": name, "incidents": len(g), "share_of_all": round(len(g) / len(inc), 4) if len(inc) else np.nan,
                     "reopen_rate": _rate(g["is_reopened"]), "sla_miss_rate": _rate(g["is_sla_missed"]),
                     "reassigned_3plus_rate": _rate(g["is_bounced"]), "high_priority_rate": _rate(g["is_high_priority"]),
                     "median_resolution_hours": g["resolution_hours"].median()})
    return pd.DataFrame(rows)


# ======================================================================== UCI
def sla_breakdown(inc: pd.DataFrame, raw, cfg) -> dict[str, pd.DataFrame]:
    inc = inc.assign(family=inc["category"] + " | " + inc["subcategory"])
    parts = []
    for dim in ["priority", "contact_type", "family", "assignment_group", "opened_week"]:
        g = inc.groupby(dim, dropna=False)
        t = g.agg(incidents=("number", "size"), sla_missed=("is_sla_missed", "sum"),
                  median_resolution_hours=("resolution_hours", "median"),
                  p90_resolution_hours=("resolution_hours", lambda s: s.quantile(0.9))).reset_index()
        t = t.rename(columns={dim: "value"}).assign(dimension=dim)
        t["value"] = t["value"].astype(str)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out["sla_miss_rate"] = out["sla_missed"] / out["incidents"]
    out["note"] = "source records SLA met/missed only - no targets or clock pauses"
    cols = ["dimension", "value", "incidents", "sla_missed", "sla_miss_rate",
            "median_resolution_hours", "p90_resolution_hours", "note"]
    return {"insight_sla_breakdown": out[cols]}


def problem_links(inc: pd.DataFrame, events: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    inc = inc.assign(family=inc["category"] + " | " + inc["subcategory"])
    ev = events[events["number"].isin(inc["number"]) & events["problem_id"].notna()]
    first_link = ev.groupby("number")["sys_updated_at"].min().rename("first_linked_at")
    linked = inc[inc["has_known_problem"]].join(first_link, on="number")
    linked["first_linked_at"] = pd.to_datetime(linked["first_linked_at"])   # stays datetime even with no events
    linked["hours_to_link"] = ((linked["first_linked_at"] - linked["opened_at"]).dt.total_seconds() / 3600).clip(lower=0)
    per_problem = linked.groupby("problem_id").agg(
        incidents=("number", "size"), families=("family", "nunique"), locations=("location", "nunique"),
        assignment_groups=("assignment_group", "nunique"), first_opened=("opened_at", "min"),
        last_opened=("opened_at", "max"), median_hours_to_link=("hours_to_link", "median"),
        reopen_rate=("is_reopened", "mean"), sla_miss_rate=("is_sla_missed", "mean"),
        example_incidents=("number", lambda s: ",".join(s.head(10)))).reset_index()
    per_problem = per_problem.sort_values("incidents", ascending=False)
    size_dist = per_problem["incidents"].clip(upper=5).value_counts().sort_index()
    dist = pd.DataFrame({"incidents_per_problem": [str(i) if i < 5 else "5+" for i in size_dist.index],
                         "problems": size_dist.values})
    return {"insight_problem_links": per_problem,
            "insight_problem_link_comparison": compare_groups(inc, "has_known_problem"),
            "insight_problem_link_size": dist}


def change_links(inc: pd.DataFrame, raw, cfg) -> dict[str, pd.DataFrame]:
    comp = compare_groups(inc, "has_change")
    per_change = inc[inc["has_change"]].groupby("rfc").size()
    comp["note"] = (f"{inc['has_change'].sum()} incidents reference {per_change.size} changes "
                    f"({(per_change == 1).mean():.0%} of changes have a single incident). Change records "
                    "themselves (dates, type, outcome) are not in the source, so change-induced "
                    "incident analysis is not possible.")
    return {"insight_change_link_comparison": comp}


# ======================================================================== Mendeley
def _mendeley_join(inc: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    iss = issues.assign(number="MHD-" + issues["id"].astype("int64").astype(str))
    wf_cols = [c for c in issues.columns if c.startswith("wf_") and c != "wf_total_time"]
    return inc[["number", "category", "subcategory", "closed_code", "resolution_hours", "is_reopened"]].merge(
        iss[["number"] + wf_cols], on="number", how="left"), wf_cols


def resolution_patterns(inc: pd.DataFrame, raw, cfg) -> dict[str, pd.DataFrame]:
    issues, _ = raw
    m, wf_cols = _mendeley_join(inc, issues)
    m["product"] = m["category"]
    m["outcome"] = m["closed_code"].fillna("Unresolved")
    mix = (m.groupby(["product", "outcome"]).size().rename("tickets").reset_index())
    mix["share_of_product"] = mix["tickets"] / mix.groupby("product")["tickets"].transform("sum")
    overall = m["outcome"].value_counts(normalize=True)
    mix["overall_share"] = mix["outcome"].map(overall)

    long = m.melt(id_vars=["number", "product"], value_vars=wf_cols, var_name="step", value_name="seconds")
    long["step"] = long["step"].str.replace("wf_", "", regex=False)
    entered = long[long["seconds"].fillna(0) > 0]          # steps a ticket never entered are excluded
    def step_table(df, keys):
        t = df.groupby(keys).agg(tickets_entering=("number", "nunique"),
                                 median_hours=("seconds", lambda s: s.median() / 3600),
                                 total_hours=("seconds", lambda s: s.sum() / 3600)).reset_index()
        return t
    steps_all = step_table(entered, ["step"])
    steps_all["share_of_tickets"] = steps_all["tickets_entering"] / m["number"].nunique()
    steps_all["share_of_total_time"] = steps_all["total_hours"] / steps_all["total_hours"].sum()
    steps_prod = step_table(entered, ["product", "step"])
    steps_prod["share_of_product_time"] = steps_prod["total_hours"] / steps_prod.groupby("product")["total_hours"].transform("sum")
    return {"insight_resolution_mix": mix,
            "insight_workflow_steps": steps_all.sort_values("total_hours", ascending=False),
            "insight_workflow_steps_by_product": steps_prod}


def handoffs(inc: pd.DataFrame, raw, cfg) -> dict[str, pd.DataFrame]:
    _, hist = raw
    nums = inc["number"].str[4:].astype(float)
    a = hist[(hist["field"] == "assignee") & hist["issueid"].isin(nums)]
    first_assign = a.groupby("issueid")["created"].min()
    inc = inc.assign(holders=inc["reassignment_count"] + 1,
                     first_assigned_at=nums.map(first_assign).values)
    inc["hours_to_first_assignment"] = ((inc["first_assigned_at"] - inc["opened_at"]).dt.total_seconds() / 3600).clip(lower=0)
    inc["holders_bucket"] = inc["holders"].clip(upper=4).map({1: "1", 2: "2", 3: "3", 4: "4+"})
    dist = inc.groupby("holders_bucket").agg(
        tickets=("number", "size"), median_resolution_hours=("resolution_hours", "median"),
        reopen_rate=("is_reopened", "mean"), cannot_reproduce_rate=("is_cannot_reproduce", "mean")).reset_index()
    dist["share"] = dist["tickets"] / dist["tickets"].sum()
    by_prod = inc.groupby("category").agg(
        tickets=("number", "size"), median_holders=("holders", "median"),
        share_3plus_holders=("is_bounced", "mean"),
        median_hours_to_first_assignment=("hours_to_first_assignment", "median"),
        median_resolution_hours=("resolution_hours", "median")).reset_index().rename(columns={"category": "product"})
    by_prod = by_prod.sort_values("tickets", ascending=False)
    return {"insight_handoff_distribution": dist, "insight_handoffs_by_product": by_prod}


INSIGHTS = {
    "sla": sla_breakdown,
    "problem_links": problem_links,
    "change_links": change_links,
    "resolution_patterns": resolution_patterns,
    "handoffs": handoffs,
}


def run(names: list[str], inc: pd.DataFrame, raw, cfg: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for n in names:
        out.update(INSIGHTS[n](inc, raw, cfg))
    return out
