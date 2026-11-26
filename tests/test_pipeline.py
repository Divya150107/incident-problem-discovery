"""End-to-end pipeline on a tiny fabricated event log (no real data needed)."""
import numpy as np
import pandas as pd

from pdisc import ingest, pipeline

PERSONAL = ["caller_id", "opened_by", "sys_created_by", "sys_updated_by", "assigned_to", "resolved_by"]


def fake_events(n_inc=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    opened = pd.Timestamp("2016-03-01") + pd.to_timedelta(rng.uniform(0, 85 * 24, n_inc), unit="h")
    rows = []
    for i in range(n_inc):
        base = {
            "number": f"INC{i:07d}", "opened_at": opened[i], "caller_id": f"Caller {rng.integers(1, 900)}",
            "opened_by": "Opened by 1", "sys_created_by": "Created by 1", "assigned_to": "Person 3",
            "resolved_by": "Resolved by 4", "category": f"Category {rng.integers(1, 6)}",
            "subcategory": f"Subcategory {rng.integers(1, 4)}", "u_symptom": "Symptom 1",
            "location": f"Location {rng.integers(1, 30)}", "assignment_group": f"Group {rng.integers(1, 8)}",
            "cmdb_ci": None, "impact": "2 - Medium", "urgency": "2 - Medium", "priority": "3 - Moderate",
            "contact_type": "Phone", "problem_id": None, "rfc": None, "caused_by": None, "closed_code": "code 6",
            "reopen_count": 0, "reassignment_count": int(rng.integers(0, 3)), "made_sla": rng.random() > 0.37,
            "active": False, "knowledge": False, "u_priority_confirmation": False, "notify": "Do Not Notify",
            "vendor": None, "sys_created_at": opened[i],
        }
        resolved = opened[i] + pd.Timedelta(hours=float(rng.gamma(1.5, 20)))
        rows.append({**base, "incident_state": "New", "sys_mod_count": 0, "sys_updated_at": opened[i],
                     "sys_updated_by": "Updated by 1", "resolved_at": resolved, "closed_at": resolved})
        rows.append({**base, "incident_state": "Resolved", "sys_mod_count": 1, "sys_updated_at": resolved,
                     "sys_updated_by": "Updated by 2", "resolved_at": resolved, "closed_at": resolved})
    return pd.DataFrame(rows)


def test_build_incidents_keeps_latest_state_and_drops_people():
    inc = ingest.build_incidents(fake_events(50), PERSONAL)
    assert len(inc) == 50
    assert (inc["incident_state"] == "Resolved").all()
    assert not set(PERSONAL) & set(inc.columns)


def test_pipeline_writes_outputs_without_personal_data(cfg, tmp_path):
    e = pipeline.run(None, cfg, fake_events(), out_dir=tmp_path / "out", hist_dir=tmp_path / "hist",
                     reviews_path=tmp_path / "none.csv")
    assert e["status"] == "success", e
    out = tmp_path / "out"
    for name in ["fact_incident", "candidate_register", "bridge_candidate_incident", "refresh_log",
                 "data_quality", "escalation_queue", "candidate_history"]:
        assert (out / f"{name}.csv").exists()
    for f in out.glob("*.csv"):
        text = f.read_text()
        assert "Caller " not in text and "Person 3" not in text and "Resolved by" not in text, f.name


def test_failed_run_is_logged_and_keeps_last_good_outputs(cfg, tmp_path):
    out, hist = tmp_path / "out", tmp_path / "hist"
    ok = pipeline.run(None, cfg, fake_events(), out_dir=out, hist_dir=hist, reviews_path=tmp_path / "x.csv")
    assert ok["status"] == "success"
    before = (out / "fact_incident.csv").read_text()
    broken = fake_events().drop(columns=["category"])            # simulate a bad extract
    bad = pipeline.run(None, cfg, broken, out_dir=out, hist_dir=hist, reviews_path=tmp_path / "x.csv")
    assert bad["status"] == "failed" and "category" in bad["error"]
    assert (out / "fact_incident.csv").read_text() == before
    log = pd.read_csv(out / "refresh_log.csv")
    assert list(log["status"]) == ["success", "failed"]


def test_as_of_replay_does_not_see_future_outcomes(cfg, tmp_path):
    ev = fake_events()
    as_of = "2016-04-10"
    pipeline.run(as_of, cfg, ev, out_dir=tmp_path / "o", hist_dir=tmp_path / "h", reviews_path=tmp_path / "x.csv")
    fact = pd.read_csv(tmp_path / "o" / "fact_incident.csv", parse_dates=["opened_at", "resolved_at"])
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)
    assert fact["opened_at"].max() <= cutoff
    assert not (fact["resolved_at"] > cutoff).any()
