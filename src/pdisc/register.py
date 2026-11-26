"""Candidate register: evidence, known-Problem comparison, explainable priority, confidence.

Keeps three things deliberately separate (as the brief requires):
  * incident priority   - the ticket field, untouched
  * investigation priority - our proposed rubric score (0-1) with per-factor breakdown
  * evidence confidence - High / Medium / Low, from how strong the evidence is
Text fields are generated from the numbers so each claim can be traced to records.
Observed patterns, possible explanations and confirmed causes are kept apart:
nothing here is a confirmed cause.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(x):
    return float(np.clip(x, 0.0, 1.0))


def candidate_stats(inc: pd.DataFrame, bridge: pd.DataFrame, recent_weeks: int) -> pd.DataFrame:
    p90 = inc["resolution_hours"].quantile(0.9)
    m = bridge.merge(inc.assign(is_slow=inc["resolution_hours"].gt(p90)), on="number", how="left")
    weeks = sorted(inc["opened_week"].unique())
    recent = set(weeks[-recent_weeks:])
    tot_recent = inc["opened_week"].isin(recent).sum()
    tot_earlier = len(inc) - tot_recent

    def per(g: pd.DataFrame) -> pd.Series:
        grp_share = g["assignment_group"].value_counts(normalize=True)
        loc_share = g["location"].value_counts(normalize=True)
        problems = g["problem_id"].dropna()
        rfcs = g["rfc"].dropna()
        shared_rfc = rfcs.value_counts()
        shared_rfc = shared_rfc[shared_rfc >= 2]
        is_recent = g["opened_week"].isin(recent)
        r, e = is_recent.sum(), (~is_recent).sum()
        trend_ratio = (r / tot_recent) / max(e / tot_earlier, 0.5 / tot_earlier) if tot_earlier else np.nan
        return pd.Series({
            "incidents": len(g),
            "distinct_callers": g["caller_key"].nunique(),
            "locations": g["location"].nunique(),
            "first_seen": g["opened_at"].min(),
            "last_seen": g["opened_at"].max(),
            "weeks_active": g["opened_week"].nunique(),
            "dominant_group": grp_share.index[0], "dominant_group_share": grp_share.iloc[0],
            "dominant_location": loc_share.index[0], "dominant_location_share": loc_share.iloc[0],
            # astype(float): a field the source lacks is NaN -> rate NaN ("n/a"), not 0
            "high_priority_rate": g["is_high_priority"].astype(float).mean(),
            "high_impact_rate": g["is_high_impact"].astype(float).mean(),
            "reopen_rate": g["is_reopened"].astype(float).mean(),
            "sla_miss_rate": g["is_sla_missed"].astype(float).mean(),
            "bounce_rate": g["is_bounced"].astype(float).mean(),
            "slow_rate": g["is_slow"].astype(float).mean(),
            "cannot_reproduce_rate": g["is_cannot_reproduce"].astype(float).mean() if "is_cannot_reproduce" in g else np.nan,
            "validation_label_rate": g["validation_label"].astype(float).mean() if "validation_label" in g else np.nan,
            "median_resolution_hours": g["resolution_hours"].median(),
            "total_resolution_hours": g["resolution_hours"].sum(),
            "total_reassignments": g["reassignment_count"].sum(),
            "known_problem_incidents": len(problems),
            "known_problem_ids": ",".join(problems.value_counts().index[:5]),
            "top_known_problem_share": problems.value_counts().iloc[0] / len(g) if len(problems) else 0.0,
            "shared_change_ids": ",".join(shared_rfc.index[:5]),
            "change_linked_incidents": len(rfcs),
            "recent_incidents": int(r),
            "trend_ratio": trend_ratio,
            "example_incidents": ",".join(g.sort_values("opened_at")["number"].head(10)),
        })

    return m.groupby("candidate_id").apply(per, include_groups=False).reset_index()


def known_problem_status(row, base_rate: float, available: bool = True) -> str:
    if not available:
        return "Not available - no Problem records in source"
    if row["known_problem_incidents"] == 0:
        return "New - no known Problem linked"
    if row["top_known_problem_share"] >= 0.3:
        return "Matches known Problem"
    if row["known_problem_incidents"] / row["incidents"] >= 3 * base_rate:
        return "Partially linked - review overlap"
    return "Incidental links only"


def score(row, g: dict, cfg: dict) -> dict:
    """Factor scores in [0, 1]. Rates are scored as lift over the global baseline,
    saturating at 3x, so 'normal for this estate' scores ~0.33, not high."""
    def lift(rate, base):
        return _clip(rate / (3 * base))

    def has(base):   # signal recorded by this source (NaN = field absent, 0 = never occurs)
        return pd.notna(base) and base > 0

    sat = cfg["impact_saturation"]
    if has(g["high_impact"]):
        impact = 0.6 * _clip(row["high_priority_rate"] / sat) + 0.4 * _clip(row["high_impact_rate"] / sat)
    else:   # no impact field: priority alone carries the impact factor
        impact = _clip(row["high_priority_rate"] / sat)
    recurrence = 0.7 * _clip(np.log10(max(row["incidents"], 1)) / np.log10(500)) \
        + 0.3 * _clip(np.log10(max(row["distinct_callers"], 1)) / np.log10(300))
    deterioration = _clip((row["trend_ratio"] - 1) / 2) if pd.notna(row["trend_ratio"]) else 0.0
    # burden averages only the signals the source records, so a missing SLA field
    # doesn't silently drag every candidate's burden down
    parts = [(row[r], g[k]) for r, k in [("reopen_rate", "reopen"), ("sla_miss_rate", "sla_miss"),
                                         ("bounce_rate", "bounce"), ("slow_rate", "slow"),
                                         ("cannot_reproduce_rate", "cannot_reproduce")] if has(g[k])]
    burden = float(np.mean([lift(r, b) for r, b in parts])) if parts else 0.0
    f = {"impact": impact, "recurrence": recurrence, "deterioration": deterioration, "burden": burden}
    w = cfg["priority_weights"]
    f["priority_score"] = sum(w[k] * f[k] for k in w)
    return f


def confidence(row, cfg: dict) -> tuple[str, str]:
    c = cfg["confidence"]
    pts, why = 0, []
    if row["incidents"] >= c["min_incidents_high"]:
        pts += 1; why.append(f"{row['incidents']} incidents")
    elif row["incidents"] < c["min_incidents_medium"]:
        pts -= 1; why.append(f"only {row['incidents']} incidents")
    if row["min_p_value"] < 1e-4:
        pts += 1; why.append("statistically strong signal")
    if "(blank)" not in row["signature"]:
        pts += 1; why.append("specific symptom recorded")
    else:
        why.append("symptom/category blank - grouping is coarser")
    if row["weeks_active"] >= 3 or row["type"] == "Burst":
        pts += 1; why.append("persistent across weeks" if row["type"] != "Burst" else "tight time cluster")
    level = "High" if pts >= 3 else "Medium" if pts == 2 else "Low"
    return level, "; ".join(why)


EXPERTISE = {
    "Recurring": "Service owner + resolver group for the signature",
    "Localised": "Site / local infrastructure support",
    "Burst": "Major-incident / infrastructure team for the site",
    "Emerging": "Service owner; check recent changes",
}


def narrative(row, source_gaps: list[str]) -> dict:
    """Plain-language hypothesis, alternatives, gaps and next step, generated from evidence."""
    where = f" at {row['location']}" if row["location"] != "(all)" else ""
    hypo = {
        "Recurring": f"Incidents under '{row['signature']}' may share an underlying weakness: they show rework/failure well above baseline.",
        "Localised": f"'{row['signature']}' is concentrated{where}; a shared local cause (site infrastructure, local config, local practice) may exist.",
        "Burst": f"A cluster of '{row['signature']}' incidents{where} in a short window suggests a single triggering event.",
        "Emerging": f"'{row['signature']}' is rising faster than overall demand; a new or worsening issue may be developing.",
    }[row["type"]]
    alts = []
    if row["dominant_group_share"] >= 0.8 and row["dominant_group"] != "(not recorded)":
        alts.append(f"{row['dominant_group_share']:.0%} handled by {row['dominant_group']}: may reflect that group's process/routing rather than a technical fault")
    if row["type"] in ("Recurring", "Emerging") and row["locations"] >= 20:
        alts.append(f"spread over {row['locations']} locations: could be routine demand or a categorisation habit, not one cause")
    if row["type"] == "Emerging":
        alts.append("could be a categorisation change or a planned rollout generating expected tickets")
    if row["type"] == "Burst":
        alts.append("could be one event logged as many tickets (duplicates) rather than a recurring problem")
        if pd.notna(row.get("burst_episodes")) and row["burst_episodes"] >= 5:
            alts.append(f"{int(row['burst_episodes'])} separate bursts: may be a batch-reporting habit or a recurring scheduled-job failure")
        if row.get("co_occurring_bursts"):
            alts.append(f"co-occurs with other bursts at the same site ({row['co_occurring_bursts']}): likely one shared event")
    if row["sla_miss_rate"] >= 0.8 and row["bounce_rate"] < 0.15:
        alts.append("high SLA misses without reassignment churn: SLA target may be mis-set for this ticket type")
    if row.get("shared_change_ids"):
        alts.append(f"incidents share change(s) {row['shared_change_ids']}: possibly change-induced")
    if "is_high_priority" in str(row.get("elevated_signals", "")):
        alts.append("elevated share of top-priority tickets: may reflect how the requester/customer sets priority, not severity")
    if pd.notna(row.get("cannot_reproduce_rate")) and row["cannot_reproduce_rate"] >= 0.1:
        alts.append("many 'Cannot Reproduce' closures: could be an intermittent fault, or tickets lacking detail")
    gaps = list(source_gaps)
    if row["change_linked_incidents"] == 0 and row["known_problem_status"] != "Not available - no Problem records in source":
        gaps.append("no linked changes recorded")
    if "(blank)" in row["signature"]:
        gaps.append("symptom or category blank on these tickets")
    step = {
        "Recurring": "Sample 10 linked incidents with the resolver group; check whether resolutions point to one fix",
        "Localised": "Ask site support whether the location shares infrastructure/config for this service",
        "Burst": "Check monitoring and change records for the burst window; confirm whether tickets are duplicates of one event",
        "Emerging": "Check changes/releases just before the rise; compare the recent and earlier tickets",
    }[row["type"]]
    if row["known_problem_status"] == "Matches known Problem":
        step = f"Review against known Problem(s) {row['known_problem_ids']} before opening anything new"
    return {"hypothesis": hypo, "alternative_explanations": " | ".join(alts) or "none generated - challenge manually",
            "evidence_gaps": " | ".join(gaps), "proposed_next_step": step, "required_expertise": EXPERTISE[row["type"]]}


REGISTER_EXTRA_COLS = [
    "family", "incidents", "distinct_callers", "locations", "first_seen", "last_seen", "weeks_active",
    "dominant_group", "dominant_group_share", "dominant_location", "dominant_location_share",
    "high_priority_rate", "high_impact_rate", "reopen_rate", "sla_miss_rate", "bounce_rate", "slow_rate",
    "cannot_reproduce_rate", "validation_label_rate",
    "median_resolution_hours", "total_resolution_hours", "total_reassignments", "known_problem_incidents",
    "known_problem_ids", "top_known_problem_share", "shared_change_ids", "change_linked_incidents",
    "recent_incidents", "trend_ratio", "example_incidents", "known_problem_status", "impact", "recurrence",
    "deterioration", "burden", "priority_score", "confidence", "confidence_reason", "hypothesis",
    "alternative_explanations", "evidence_gaps", "proposed_next_step", "required_expertise",
    "priority_rationale", "priority_rank", "related_candidates", "in_backlog", "backlog_rank"]


def build_register(inc: pd.DataFrame, cand: pd.DataFrame, bridge: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cand.empty:   # quiet period: well-formed empty register so later stages still run
        reg = cand.reindex(columns=list(cand.columns) + REGISTER_EXTRA_COLS)
        return reg, pd.DataFrame(columns=["candidate_id", "factor", "factor_score", "weight", "contribution"])
    stats_ = candidate_stats(inc, bridge, cfg["candidates"]["emerging"]["recent_weeks"])
    reg = cand.merge(stats_, on="candidate_id", how="left")
    reg["family"] = reg["signature"].str.split(" | ", regex=False).str[:2].str.join(" | ")
    p90 = inc["resolution_hours"].quantile(0.9)

    def rate(col):
        return float(inc[col].astype(float).mean()) if col in inc else np.nan

    g = {"high_priority": rate("is_high_priority"), "high_impact": rate("is_high_impact"),
         "reopen": rate("is_reopened"), "sla_miss": rate("is_sla_missed"), "bounce": rate("is_bounced"),
         "slow": float(inc["resolution_hours"].gt(p90).mean()), "cannot_reproduce": rate("is_cannot_reproduce")}
    base_kp = inc["has_known_problem"].mean()

    reg["known_problem_status"] = reg.apply(known_problem_status, axis=1, base_rate=base_kp,
                                            available=cfg.get("has_known_problems", True))
    scores = reg.apply(lambda r: pd.Series(score(r, g, cfg)), axis=1)
    reg = pd.concat([reg, scores], axis=1)
    conf = reg.apply(lambda r: pd.Series(confidence(r, cfg), index=["confidence", "confidence_reason"]), axis=1)
    reg = pd.concat([reg, conf], axis=1)
    narr = reg.apply(lambda r: pd.Series(narrative(r, cfg.get("evidence_gaps", []))), axis=1)
    reg = pd.concat([reg, narr], axis=1)

    w = cfg["priority_weights"]
    def top_drivers(r):
        contrib = sorted(((w[k] * r[k], k) for k in w), reverse=True)
        return ", ".join(f"{k} ({c:.2f})" for c, k in contrib[:2])
    reg["priority_rationale"] = reg.apply(top_drivers, axis=1)
    reg = reg.sort_values("priority_score", ascending=False).reset_index(drop=True)
    reg["priority_rank"] = np.arange(1, len(reg) + 1)

    # backlog: best candidate per family; the rest are listed as related
    reg["related_candidates"] = reg.groupby("family")["candidate_id"].transform(lambda s: ",".join(s))
    reg["related_candidates"] = [",".join(x for x in rel.split(",") if x != cid)
                                 for cid, rel in zip(reg["candidate_id"], reg["related_candidates"])]
    reg["in_backlog"] = ~reg.duplicated("family", keep="first")
    reg["backlog_rank"] = np.where(reg["in_backlog"], reg["in_backlog"].cumsum(), np.nan)

    factors = reg.melt(id_vars=["candidate_id"], value_vars=list(w), var_name="factor", value_name="factor_score")
    factors["weight"] = factors["factor"].map(w)
    factors["contribution"] = factors["factor_score"] * factors["weight"]
    return reg, factors
