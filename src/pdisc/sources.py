"""Data sources: each maps its raw data onto the shared incident table.

Every source provides
    load(cfg)                       -> raw data
    incidents(raw, cfg, as_of=None) -> (in-window incidents, out-of-window incidents)
    quality(raw, inc, out, cfg)     -> data-quality table

With `as_of`, a source rebuilds incidents as they stood on that date (point-in-time
replay): only updates recorded by then count, so no future outcomes leak in.

Signals a source does not have are set to NaN (unavailable), never to False: a
dashboard must show "n/a", not "0% SLA missed".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pdisc import ingest

ROOT = Path(__file__).resolve().parents[2]


def _finish_common(inc: pd.DataFrame) -> pd.DataFrame:
    """Derived columns shared by all sources."""
    inc["signature"] = inc["category"] + " | " + inc["subcategory"] + " | " + inc["u_symptom"]
    inc["opened_week"] = inc["opened_at"].dt.to_period("W-SUN").dt.start_time
    inc["opened_date"] = inc["opened_at"].dt.normalize()
    hrs = (inc["resolved_at"] - inc["opened_at"]).dt.total_seconds() / 3600
    inc["resolution_hours"] = hrs.where(hrs >= 0)
    inc["is_resolved"] = inc["resolved_at"].notna()
    inc["has_known_problem"] = inc["problem_id"].notna()
    inc["has_change"] = inc["rfc"].notna()
    return inc


# ======================================================================== UCI
class UCISource:
    name = "uci"

    def load(self, cfg: dict):
        return ingest.load_event_log(ROOT / cfg["raw"])

    def incidents(self, events: pd.DataFrame, cfg: dict, as_of: pd.Timestamp | None = None):
        if as_of is not None:
            events = events[events["sys_updated_at"].le(as_of) & events["opened_at"].le(as_of)]
        inc = ingest.build_incidents(events, cfg["personal_columns"])
        if as_of is not None:
            # outcomes not yet known at the as-of date must not count as known
            late = inc["resolved_at"].gt(as_of)
            inc.loc[late, "resolved_at"] = pd.NaT
            inc.loc[late, "resolution_hours"] = np.nan
            inc.loc[late, "is_resolved"] = False
        w = cfg["analysis_window"]
        end = min(as_of, pd.Timestamp(w["end"])) if as_of is not None else pd.Timestamp(w["end"])
        return ingest.apply_window(inc, w["start"], str(end))

    def quality(self, events, inc, out, cfg):
        return ingest.quality_report(events, inc, out)


# ======================================================================== Mendeley
PROJECT_RE = r"^(C\d\d)(..)(.+)$"     # country code (masked C00-C17) + 2-char product + client


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="mixed", utc=True, errors="coerce").dt.tz_convert(None)


class MendeleySource:
    name = "mendeley"

    def load(self, cfg: dict):
        issues = pd.read_csv(ROOT / cfg["raw_issues"], low_memory=False)
        hist = pd.read_csv(ROOT / cfg["raw_history"], low_memory=False)
        for c in ["issue_created", "issue_resolution_date"]:
            issues[c] = _ts(issues[c])
        hist["created"] = _ts(hist["created"])
        return issues, hist

    def incidents(self, raw, cfg: dict, as_of: pd.Timestamp | None = None):
        issues, hist = raw
        t = issues[issues["issue_type"].isin(cfg["issue_types"])].copy()
        if as_of is not None:
            t = t[t["issue_created"].le(as_of)]
            hist = hist[hist["created"].le(as_of)]
        hist = hist[hist["issueid"].isin(t["id"])]

        status = hist[hist["field"] == "status"].sort_values("created")
        reopens = status[status["value"].astype(str).str.lower().eq("reopened")].groupby("issueid").size()
        last_status = status.groupby("issueid")["value"].last()
        # number of distinct people who held the ticket (a count only; who they were is never used)
        holders = hist[hist["field"] == "assignee"].groupby("issueid")["value"].nunique()

        proj = t["issue_proj"].astype(str)
        parts = proj.str.extract(PROJECT_RE)
        external = parts[0].notna()
        product = np.where(external, "Product " + parts[1].fillna(""), "Internal")
        country = np.where(external, "Country " + parts[0].fillna(""), "Internal project")
        client = np.where(external, "Client " + parts[2].fillna(""), "Project " + proj)

        resolved_at = t["issue_resolution_date"]
        if as_of is not None:
            resolved_at = resolved_at.where(resolved_at.le(as_of))
        known = resolved_at.notna()
        resolution = t["issue_resolution"].where(known)

        inc = pd.DataFrame({
            "number": "MHD-" + t["id"].astype("int64").astype(str),
            "opened_at": t["issue_created"].values,
            "resolved_at": resolved_at.values,
            "project": proj.values,
            # service hierarchy: product (service) > country > client deployment
            "category": product, "subcategory": country, "u_symptom": client,
            "location": country,
            "assignment_group": "(not recorded)",
            "contact_type": None, "urgency": None, "impact": None,
            "priority": t["issue_priority"].values,
            "issue_type": t["issue_type"].values,
            "incident_state": t["id"].map(last_status).fillna("open").values,
            "closed_code": resolution.values,
            "caller_key": pd.util.hash_pandas_object(t["issue_reporter"].fillna(""), index=False).astype(str).values,
            "reopen_count": t["id"].map(reopens).fillna(0).astype(int).values,
            "reassignment_count": (t["id"].map(holders).fillna(0) - 1).clip(lower=0).astype(int).values,
            "problem_id": None, "rfc": None,
        })
        inc["is_high_priority"] = inc["priority"].isin(["Highest", "Blocker"])
        inc["is_high_impact"] = np.nan          # no impact field
        inc["is_reopened"] = inc["reopen_count"].gt(0)
        inc["is_sla_missed"] = np.nan           # no SLA field
        inc["is_bounced"] = inc["reassignment_count"].ge(2)   # 3+ people held the ticket
        inc["is_cannot_reproduce"] = (resolution == "Cannot Reproduce").values
        inc["is_duplicate"] = (resolution == "Duplicate").values
        inc = _finish_common(inc)

        w = cfg["analysis_window"]
        end = min(as_of, pd.Timestamp(w["end"])) if as_of is not None else pd.Timestamp(w["end"])
        return ingest.apply_window(inc, w["start"], str(end))

    def quality(self, raw, inc, out, cfg):
        issues, hist = raw
        n = len(inc)
        tot = len(issues)
        in_win_all = issues[issues["issue_created"].between(pd.Timestamp(cfg["analysis_window"]["start"]),
                                                            pd.Timestamp(cfg["analysis_window"]["end"]))]
        raw_inc = (issues.assign(number="MHD-" + issues["id"].astype("int64").astype(str))
                   .set_index("number").reindex(inc["number"]))
        rows = [
            ("coverage", "issues in source (all types, 2007-2023)", tot, tot, ""),
            ("coverage", "non-incident types excluded in window (stories, tasks, service requests...)",
             (~in_win_all["issue_type"].isin(cfg["issue_types"])).sum(), len(in_win_all),
             "only customer 'Ticket' issues analysed"),
            ("coverage", "tickets outside analysis window", len(out), len(out) + n,
             "priority is 'unknown' before 2019; window starts 2021"),
            ("duplicates", "duplicate issue ids", issues["id"].duplicated().sum(), tot, ""),
            ("missing", "priority 'unknown'", raw_inc["issue_priority"].eq("unknown").sum(), n, ""),
            ("missing", "not resolved by end of window", (~inc["is_resolved"]).sum(), n, ""),
            ("missing", "tickets without change history", (~inc["number"].str[4:].astype(float).isin(hist["issueid"])).sum(), n,
             "reopen / reassignment counts come from the history"),
            ("structure", "internal projects (no product/country/client pattern)",
             inc["category"].eq("Internal").sum(), n, "grouped as product 'Internal'"),
            ("missing", "SLA field", n, n, "not in source: SLA signal unavailable (shown as n/a, not 0%)"),
            ("missing", "category / configuration item", n, n, "service = project code (product / country / client)"),
            ("missing", "known Problem / change links", n, n, "known-Problem comparison not possible; Duplicate used for validation"),
            ("missing", "impact field", n, n, "impact uses priority (Highest/Blocker) only"),
        ]
        df = pd.DataFrame(rows, columns=["area", "check", "count", "total", "note"])
        df["pct"] = (100 * df["count"] / df["total"]).round(2)
        return df[["area", "check", "count", "total", "pct", "note"]]


SOURCES = {"uci": UCISource(), "mendeley": MendeleySource()}
