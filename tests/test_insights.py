import numpy as np
import pandas as pd

from conftest import make_incidents
from pdisc import insights
from pdisc.sources import MendeleySource
from test_mendeley import fake_raw


def _uci_inc():
    inc = make_incidents(500)
    inc["priority"] = np.where(np.arange(500) < 50, "2 - High", "3 - Moderate")
    inc["contact_type"] = "Phone"
    inc["location"] = "Location 1"
    inc.loc[:9, "problem_id"] = "Problem ID 1"
    inc.loc[10:11, "problem_id"] = "Problem ID 2"
    inc.loc[:4, "rfc"] = ["CHG1", "CHG1", "CHG2", "CHG3", "CHG4"]
    inc["has_known_problem"] = inc["problem_id"].notna()
    inc["has_change"] = inc["rfc"].notna()
    return inc


def test_sla_breakdown_matches_fact_table(cfg):
    inc = _uci_inc()
    t = insights.sla_breakdown(inc, None, cfg)["insight_sla_breakdown"]
    for dim in ["priority", "contact_type", "family", "assignment_group", "opened_week"]:
        d = t[t["dimension"] == dim]
        assert d["incidents"].sum() == len(inc)
        assert d["sla_missed"].sum() == inc["is_sla_missed"].sum()


def test_problem_link_timing_uses_first_update_carrying_the_id(cfg):
    inc = _uci_inc()
    num = inc.loc[0, "number"]
    t0 = inc.loc[0, "opened_at"]
    events = pd.DataFrame({
        "number": [num, num, num],
        "sys_updated_at": [t0, t0 + pd.Timedelta(hours=5), t0 + pd.Timedelta(hours=9)],
        "problem_id": [None, "Problem ID 1", "Problem ID 1"],
    })
    out = insights.problem_links(inc.iloc[:1].assign(has_known_problem=True), events, cfg)
    p = out["insight_problem_links"]
    assert abs(p.loc[p["problem_id"] == "Problem ID 1", "median_hours_to_link"].iloc[0] - 5.0) < 1e-9


def test_problem_link_comparison_and_sizes(cfg):
    inc = _uci_inc()
    out = insights.problem_links(inc, pd.DataFrame(columns=["number", "sys_updated_at", "problem_id"]), cfg)
    comp = out["insight_problem_link_comparison"].set_index("group")
    assert comp.loc["linked", "incidents"] == 12 and comp.loc["not linked", "incidents"] == 488
    sizes = out["insight_problem_link_size"].set_index("incidents_per_problem")["problems"]
    assert sizes["2"] == 1 and sizes["5+"] == 1


def test_change_links_states_the_coverage_limit(cfg):
    out = insights.change_links(_uci_inc(), None, cfg)["insight_change_link_comparison"]
    assert "not possible" in out["note"].iloc[0]
    assert out.set_index("group").loc["linked", "incidents"] == 5


def test_workflow_step_medians_ignore_steps_never_entered(mendeley_cfg):
    issues, hist = fake_raw()
    issues["wf_waiting"] = np.nan
    issues.loc[issues.index[:3], "wf_waiting"] = [3600, 7200, 10800]   # only 3 tickets ever waited
    issues["wf_open"] = 60.0
    inc, _ = MendeleySource().incidents((issues, hist), mendeley_cfg)
    steps = insights.resolution_patterns(inc, (issues, hist), mendeley_cfg)["insight_workflow_steps"].set_index("step")
    waited = inc["number"].isin("MHD-" + issues.loc[issues.index[:3], "id"].astype(int).astype(str)).sum()
    assert steps.loc["waiting", "tickets_entering"] == waited
    if waited == 3:
        assert abs(steps.loc["waiting", "median_hours"] - 2.0) < 1e-9


def test_handoff_tables_hold_no_person_ids(mendeley_cfg):
    issues, hist = fake_raw()
    hist.loc[hist["field"] == "assignee", "value"] = "agentSECRET"
    raw = (issues, hist)
    inc, _ = MendeleySource().incidents(raw, mendeley_cfg)
    out = insights.handoffs(inc, raw, mendeley_cfg)
    assert out["insight_handoff_distribution"]["tickets"].sum() == len(inc)
    import re
    for df in out.values():
        text = df.to_csv()
        assert "agentSECRET" not in text and not re.search(r"\brep\d+\b", text)
