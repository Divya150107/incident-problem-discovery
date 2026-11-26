"""Mendeley adapter: mapping, point-in-time replay, unavailable signals, privacy."""
import numpy as np
import pandas as pd

from pdisc import candidates, pipeline, register
from pdisc.sources import MendeleySource

SRC = MendeleySource()


def fake_raw(n=400, seed=0):
    rng = np.random.default_rng(seed)
    created = pd.Timestamp("2021-01-04") + pd.to_timedelta(rng.uniform(0, 30 * 7 * 24, n), unit="h")
    projects = rng.choice(["C06gg__z0", "C06gg2xcu", "C12hhx18a", "t96"], n)
    issues = pd.DataFrame({
        "id": np.arange(1, n + 1, dtype=float),
        "issue_proj": projects,
        "issue_reporter": [f"rep{r}" for r in rng.integers(0, 50, n)],
        "issue_assignee": [f"agent{a}" for a in rng.integers(0, 9, n)],
        "issue_type": rng.choice(["Ticket"] * 9 + ["Story"], n),
        "issue_priority": rng.choice(["Medium", "High", "Highest"], n, p=[0.7, 0.2, 0.1]),
        "issue_created": created,
        "issue_resolution_date": created + pd.to_timedelta(rng.gamma(2, 50, n), unit="h"),
        "issue_resolution": rng.choice(["Done", "Duplicate", "Cannot Reproduce"], n, p=[0.9, 0.05, 0.05]),
    })
    hist = []
    for i, row in issues.iterrows():
        hist.append({"issueid": row["id"], "field": "assignee", "value": "agentA", "created": row["issue_created"]})
        hist.append({"issueid": row["id"], "field": "status", "value": "resolved",
                     "created": row["issue_resolution_date"]})
    hist = pd.DataFrame(hist)
    return issues, hist


def test_project_code_maps_to_service_hierarchy(mendeley_cfg):
    inc, _ = SRC.incidents(fake_raw(), mendeley_cfg)
    row = inc[inc["project"] == "C12hhx18a"].iloc[0]
    assert (row["category"], row["subcategory"], row["u_symptom"]) == ("Product hh", "Country C12", "Client x18a")
    internal = inc[inc["project"] == "t96"].iloc[0]
    assert internal["category"] == "Internal"


def test_only_customer_tickets_are_analysed(mendeley_cfg):
    issues, hist = fake_raw()
    inc, _ = SRC.incidents((issues, hist), mendeley_cfg)
    assert set(inc["issue_type"]) == {"Ticket"}
    assert len(inc) == (issues["issue_type"] == "Ticket").sum()


def test_replay_ignores_later_reopens_resolutions_and_labels(mendeley_cfg):
    issues, hist = fake_raw()
    iid = issues.loc[issues["issue_type"] == "Ticket", "id"].iloc[0]
    t0 = issues.set_index("id").loc[iid, "issue_created"]
    as_of = t0 + pd.Timedelta(hours=1)
    issues.loc[issues["id"] == iid, ["issue_resolution_date", "issue_resolution"]] = [t0 + pd.Timedelta(days=5), "Duplicate"]
    hist = pd.concat([hist, pd.DataFrame([{"issueid": iid, "field": "status", "value": "reopened",
                                           "created": t0 + pd.Timedelta(days=6)}])], ignore_index=True)
    num = f"MHD-{int(iid)}"
    final, _ = SRC.incidents((issues, hist), mendeley_cfg)
    then, _ = SRC.incidents((issues, hist), mendeley_cfg, as_of)
    f, t = final.set_index("number").loc[num], then.set_index("number").loc[num]
    assert f["is_reopened"] and f["is_duplicate"] and f["is_resolved"]
    assert not t["is_reopened"] and not t["is_duplicate"] and not t["is_resolved"]


def test_unavailable_fields_are_nan_not_zero(mendeley_cfg):
    inc, _ = SRC.incidents(fake_raw(), mendeley_cfg)
    assert inc["is_sla_missed"].isna().all() and inc["is_high_impact"].isna().all()
    assert "is_sla_missed" not in candidates.available_signals(inc)


def test_localised_detector_is_off_and_duplicate_is_never_a_signal(mendeley_cfg):
    assert "localised" not in mendeley_cfg["detectors"]
    assert "is_duplicate" not in candidates.SIGNAL_LABELS      # validation label, not a detection signal


def test_burden_ignores_missing_sla(mendeley_cfg):
    g = {"high_priority": 0.1, "high_impact": np.nan, "reopen": 0.04, "sla_miss": np.nan,
         "bounce": 0.12, "slow": 0.1, "cannot_reproduce": 0.02}
    row = {"high_priority_rate": 0.1, "high_impact_rate": np.nan, "incidents": 50, "distinct_callers": 20,
           "trend_ratio": 1.0, "reopen_rate": 0.12, "sla_miss_rate": np.nan, "bounce_rate": 0.36,
           "slow_rate": 0.3, "cannot_reproduce_rate": 0.06}
    s = register.score(row, g, mendeley_cfg)
    assert abs(s["burden"] - 1.0) < 1e-9          # every available signal is at 3x baseline
    assert abs(s["impact"] - 0.4) < 1e-9          # priority-only impact: 0.10 / 0.25


def test_end_to_end_routes_to_service_owner_and_hides_people(mendeley_cfg, tmp_path):
    issues, hist = fake_raw(1500)
    # plant a reopen-heavy client so at least one candidate is escalated
    bad = issues.index[(issues["issue_proj"] == "C12hhx18a") & (issues["issue_type"] == "Ticket")][:60]
    reopen = pd.DataFrame({"issueid": issues.loc[bad, "id"].values, "field": "status", "value": "reopened",
                           "created": issues.loc[bad, "issue_created"].values + pd.Timedelta(hours=2)})
    issues.loc[bad, "issue_priority"] = "Highest"
    cfg = {**mendeley_cfg, "escalation": {**mendeley_cfg["escalation"], "min_priority_score": 0.0},
           "analysis_window": {"start": "2021-01-04", "end": "2021-08-01 23:59"}}
    e = pipeline.run(None, cfg, (issues, pd.concat([hist, reopen], ignore_index=True)),
                     out_dir=tmp_path / "o", hist_dir=tmp_path / "h", reviews_path=tmp_path / "x.csv")
    assert e["status"] == "success", e
    q = pd.read_csv(tmp_path / "o" / "escalation_queue.csv")
    assert len(q) and q["route_to"].str.startswith("Service owner:").all()
    for f in (tmp_path / "o").glob("*.csv"):
        text = f.read_text()
        assert "agent" not in text and "rep1" not in text, f.name
