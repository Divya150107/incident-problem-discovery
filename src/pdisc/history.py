"""Continuous operation: candidate history across runs, reviewer feedback, escalation proposals.

Each run is tagged with an as-of date. Comparing a run's register to the previous
run's gives each candidate a lifecycle status (New / Persistent / Closed). Reviewer
decisions are read from config/reviews.csv (owned by Problem Management) and fed
back into the next run:
  * Rejected  -> suppressed from escalation unless its priority has risen materially
  * Accepted  -> kept visible, marked as under investigation
  * Deferred  -> kept, re-proposed when priority rises
Nothing here creates Problems, changes incident priorities or triggers remediation:
the escalation queue is a list of PROPOSALS awaiting human review.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_COLS = ["run_id", "as_of", "candidate_id", "type", "family", "priority_score", "confidence",
                "incidents", "lifecycle_status"]
REVIEW_COLS = ["candidate_id", "decision", "reason", "reviewer_role", "reviewed_on", "priority_at_review"]


def load_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)


def lifecycle(reg: pd.DataFrame, history: pd.DataFrame, run_id: str, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (register with lifecycle columns, rows to append to history incl. closed)."""
    prev_runs = history[history["lifecycle_status"] != "Closed"]
    last_run = prev_runs["run_id"].iloc[-1] if len(prev_runs) else None
    prev_ids = set(prev_runs.loc[prev_runs["run_id"] == last_run, "candidate_id"]) if last_run else set()
    ever = history.groupby("candidate_id")["as_of"].min() if len(history) else pd.Series(dtype=object)

    reg = reg.copy()
    reg["lifecycle_status"] = ["Persistent" if c in prev_ids else "New" for c in reg["candidate_id"]]
    reg["first_detected_as_of"] = reg["candidate_id"].map(ever).fillna(str(as_of.date()))
    prev_score = history[history["run_id"] == last_run].set_index("candidate_id")["priority_score"] if last_run else pd.Series(dtype=float)
    reg["priority_change"] = reg["priority_score"] - reg["candidate_id"].map(prev_score)

    rows = reg.assign(run_id=run_id, as_of=str(as_of.date()))[HISTORY_COLS]
    closed_ids = prev_ids - set(reg["candidate_id"])
    if closed_ids:
        closed = history[(history["run_id"] == last_run) & history["candidate_id"].isin(closed_ids)].copy()
        closed["run_id"], closed["as_of"], closed["lifecycle_status"] = run_id, str(as_of.date()), "Closed"
        rows = pd.concat([rows, closed[HISTORY_COLS]], ignore_index=True)
    return reg, rows


def apply_reviews(reg: pd.DataFrame, reviews: pd.DataFrame, rise_threshold: float = 0.10) -> pd.DataFrame:
    reg = reg.copy()
    last = reviews.sort_values("reviewed_on").groupby("candidate_id").tail(1).set_index("candidate_id")
    reg["review_decision"] = reg["candidate_id"].map(last["decision"]).fillna("Awaiting review")
    reg["review_reason"] = reg["candidate_id"].map(last["reason"]).fillna("")
    at_review = reg["candidate_id"].map(last["priority_at_review"]).astype(float)
    reg["risen_since_review"] = (reg["priority_score"] - at_review).ge(rise_threshold)
    return reg


def escalation_queue(reg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    e = cfg["escalation"]
    eligible = reg["in_backlog"] & reg["priority_score"].ge(e["min_priority_score"])
    eligible &= ~reg["known_problem_status"].eq("Matches known Problem")
    blocked = reg["review_decision"].isin(["Rejected", "Accepted"]) & ~reg["risen_since_review"]
    q = reg[eligible & ~blocked].copy()
    # route to the resolver group that handles most linked incidents; if the source
    # doesn't record groups, route to the service owner rather than invent a team
    no_group = q["dominant_group"].eq("(not recorded)")
    q["route_to"] = np.where(no_group, "Service owner: " + q["family"].astype(str), q["dominant_group"])
    q["reason_for_routing"] = [
        "resolver group not recorded in source - routed to the owner of this product/service" if ng
        else f"{s:.0%} of linked incidents handled by this group"
        for ng, s in zip(no_group, q["dominant_group_share"])]
    q["escalation_status"] = "Proposed - awaiting Problem Management review"
    q["re_proposed"] = q["risen_since_review"]
    return q[["candidate_id", "backlog_rank", "type", "family", "signature", "location", "priority_score",
              "confidence", "route_to", "reason_for_routing", "proposed_next_step", "required_expertise",
              "escalation_status", "re_proposed", "lifecycle_status"]]
