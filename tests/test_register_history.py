import numpy as np
import pandas as pd

from conftest import make_incidents
from pdisc import candidates, history, register


def _scenario(cfg, known_problem=False):
    bg = make_incidents(6000)
    bad = make_incidents(40, seed=1, category="Category 9", subcategory="Subcategory 99")
    bad["is_reopened"] = np.arange(40) < 12
    bad["is_high_priority"] = np.arange(40) < 20
    if known_problem:
        bad["problem_id"] = ["Problem ID 7"] * 20 + [None] * 20
        bad["has_known_problem"] = bad["problem_id"].notna()
    slow = make_incidents(40, seed=2, category="Category 10", subcategory="Subcategory 100")
    slow["is_bounced"] = np.arange(40) < 20
    inc = pd.concat([bg, bad, slow], ignore_index=True)
    cand, bridge = candidates.generate(inc, cfg)
    reg, factors = register.build_register(inc, cand, bridge, cfg)
    return inc, reg, factors


def test_priority_and_confidence_are_separate_and_explained(cfg):
    _, reg, factors = _scenario(cfg)
    row = reg[reg["signature"] == "Category 9 | Subcategory 99 | Symptom 1"].iloc[0]
    assert 0 <= row["priority_score"] <= 1
    assert row["confidence"] in {"High", "Medium", "Low"}
    assert row["priority_rationale"] and row["confidence_reason"]
    # priority = sum of weighted factor contributions (fully explainable)
    contrib = factors[factors["candidate_id"] == row["candidate_id"]]["contribution"].sum()
    assert abs(contrib - row["priority_score"]) < 1e-9


def test_hypothesis_never_claims_confirmed_cause(cfg):
    _, reg, _ = _scenario(cfg)
    text = " ".join(reg["hypothesis"]).lower()
    assert "may" in text and "confirmed" not in text and "root cause is" not in text


def test_known_problem_match_is_detected(cfg):
    _, reg, _ = _scenario(cfg, known_problem=True)
    row = reg[reg["signature"] == "Category 9 | Subcategory 99 | Symptom 1"].iloc[0]
    assert row["known_problem_status"] == "Matches known Problem"
    assert "Problem ID 7" in row["proposed_next_step"]


def test_backlog_has_one_entry_per_family(cfg):
    _, reg, _ = _scenario(cfg)
    bl = reg[reg["in_backlog"]]
    assert bl["family"].is_unique


def _hist_run(reg, hist, run_id, as_of):
    reg2, rows = history.lifecycle(reg, hist, run_id, pd.Timestamp(as_of))
    return reg2, pd.concat([hist, rows], ignore_index=True)


def test_lifecycle_new_persistent_closed(cfg):
    _, reg, _ = _scenario(cfg)
    hist = pd.DataFrame(columns=history.HISTORY_COLS)
    r1, hist = _hist_run(reg, hist, "r1", "2016-05-01")
    assert set(r1["lifecycle_status"]) == {"New"}
    dropped = reg["candidate_id"].iloc[0]
    r2, hist = _hist_run(reg.iloc[1:], hist, "r2", "2016-05-08")
    assert set(r2["lifecycle_status"]) == {"Persistent"}
    closed = hist[(hist["run_id"] == "r2") & (hist["lifecycle_status"] == "Closed")]
    assert list(closed["candidate_id"]) == [dropped]


def test_rejected_candidate_suppressed_unless_priority_rises(cfg):
    cfg = {**cfg, "escalation": {**cfg["escalation"], "min_priority_score": 0.0}}
    _, reg, _ = _scenario(cfg)
    reg = reg.assign(lifecycle_status="New")
    top = reg[reg["in_backlog"]].iloc[0]
    reviews = pd.DataFrame([{"candidate_id": top["candidate_id"], "decision": "Rejected",
                             "reason": "normal demand", "reviewer_role": "Problem Manager",
                             "reviewed_on": "2016-05-02", "priority_at_review": top["priority_score"]}])
    q = history.escalation_queue(history.apply_reviews(reg, reviews), cfg)
    assert top["candidate_id"] not in set(q["candidate_id"])
    # if the pattern worsens materially after rejection, it is re-proposed
    reviews.loc[0, "priority_at_review"] = top["priority_score"] - 0.2
    q = history.escalation_queue(history.apply_reviews(reg, reviews), cfg)
    assert q.set_index("candidate_id").loc[top["candidate_id"], "re_proposed"]


def test_escalation_is_only_ever_a_proposal(cfg):
    cfg = {**cfg, "escalation": {**cfg["escalation"], "min_priority_score": 0.0}}
    _, reg, _ = _scenario(cfg)
    reg = history.apply_reviews(reg.assign(lifecycle_status="New"), pd.DataFrame(columns=history.REVIEW_COLS))
    q = history.escalation_queue(reg, cfg)
    assert len(q) and q["escalation_status"].str.startswith("Proposed").all()
