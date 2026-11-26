import numpy as np
import pandas as pd

from conftest import finish, make_incidents
from pdisc import candidates


def _types(cand, sig, loc=None):
    m = cand["signature"].eq(sig)
    if loc:
        m &= cand["location"].eq(loc)
    return set(cand.loc[m, "type"])


def test_background_noise_produces_no_statistical_candidates(cfg):
    # pure noise must not yield candidates: without multiple-testing correction,
    # ~150 signature x signal tests at p<0.01 produced chance "findings" here
    for seed in range(5):
        cand, _ = candidates.generate(make_incidents(6000, seed=100 + seed), cfg)
        stat = cand[cand["type"].isin(["Recurring", "Localised", "Emerging"])] if len(cand) else cand
        assert len(stat) == 0, stat[["type", "signature", "evidence"]].to_string()


def test_bh_adjustment_matches_known_values():
    q = candidates.bh_adjust(np.array([0.01, 0.04, 0.03, 0.005]))
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_recurring_flags_signature_with_elevated_reopens(background, cfg):
    bad = make_incidents(40, seed=1, category="Category 9", subcategory="Subcategory 99")
    bad["is_reopened"] = np.arange(40) < 12          # 30% reopened vs ~1% baseline
    inc = pd.concat([background, bad], ignore_index=True)
    cand, bridge = candidates.generate(inc, cfg)
    sig = "Category 9 | Subcategory 99 | Symptom 1"
    assert "Recurring" in _types(cand, sig)
    row = cand[(cand["signature"] == sig) & (cand["type"] == "Recurring")].iloc[0]
    assert "reopened" in row["evidence"]
    # evidence is traceable: all 40 incidents are linked to the candidate
    assert bridge[bridge["candidate_id"] == row["candidate_id"]]["number"].nunique() == 40


def test_high_volume_alone_is_not_a_candidate(background, cfg):
    busy = make_incidents(800, seed=2, category="Category 8", subcategory="Subcategory 88")
    inc = pd.concat([background, busy], ignore_index=True)
    cand, _ = candidates.generate(inc, cfg)
    assert "Recurring" not in _types(cand, "Category 8 | Subcategory 88 | Symptom 1")


def test_localised_flags_concentration_at_one_site(background, cfg):
    sig_rows = make_incidents(60, seed=3, category="Category 7", subcategory="Subcategory 77")
    sig_rows.loc[:29, "location"] = "Location 99"      # half at a site that is otherwise unused
    inc = pd.concat([background, sig_rows], ignore_index=True)
    cand, _ = candidates.generate(inc, cfg)
    assert "Localised" in _types(cand, "Category 7 | Subcategory 77 | Symptom 1", "Location 99")


def test_burst_detected_and_spread_out_incidents_are_not(background, cfg):
    t0 = pd.Timestamp("2016-04-12 09:00")
    burst = make_incidents(12, seed=4, category="Category 6", subcategory="Subcategory 66", location="Location 5")
    burst["opened_at"] = [t0 + pd.Timedelta(minutes=15 * i) for i in range(12)]
    spread = make_incidents(12, seed=5, category="Category 6", subcategory="Subcategory 67", location="Location 5")
    inc = pd.concat([background, finish(burst), spread], ignore_index=True)
    cand, bridge = candidates.generate(inc, cfg)
    assert "Burst" in _types(cand, "Category 6 | Subcategory 66 | Symptom 1", "Location 5")
    assert "Burst" not in _types(cand, "Category 6 | Subcategory 67 | Symptom 1", "Location 5")


def test_co_occurring_bursts_at_same_site_are_linked(background, cfg):
    t0 = pd.Timestamp("2016-04-12 09:00")
    parts = []
    for i, sub in enumerate(["Subcategory 61", "Subcategory 62"]):
        b = make_incidents(12, seed=10 + i, category="Category 6", subcategory=sub, location="Location 5")
        b["opened_at"] = [t0 + pd.Timedelta(minutes=10 * j + 30 * i) for j in range(12)]
        parts.append(finish(b))
    cand, _ = candidates.generate(pd.concat([background, *parts], ignore_index=True), cfg)
    bursts = cand[cand["type"] == "Burst"].set_index("signature")
    a = bursts.loc["Category 6 | Subcategory 61 | Symptom 1"]
    b = bursts.loc["Category 6 | Subcategory 62 | Symptom 1"]
    assert b["candidate_id"] in a["co_occurring_bursts"]


def test_emerging_flags_rising_signature(background, cfg):
    weeks_start = pd.Timestamp("2016-03-07")
    early = make_incidents(9, seed=6, category="Category 15", subcategory="Subcategory 155", weeks=9)
    late = make_incidents(40, seed=7, category="Category 15", subcategory="Subcategory 155",
                          start=str((weeks_start + pd.Timedelta(weeks=9)).date()), weeks=3)
    inc = pd.concat([background, early, late], ignore_index=True)
    cand, _ = candidates.generate(inc, cfg)
    assert "Emerging" in _types(cand, "Category 15 | Subcategory 155 | Symptom 1")


def test_candidate_ids_are_stable_across_runs(background, cfg):
    bad = make_incidents(40, seed=1, category="Category 9", subcategory="Subcategory 99")
    bad["is_reopened"] = np.arange(40) < 12
    inc = pd.concat([background, bad], ignore_index=True)
    a, _ = candidates.generate(inc, cfg)
    b, _ = candidates.generate(inc.sample(frac=1, random_state=3), cfg)
    assert set(a["candidate_id"]) == set(b["candidate_id"])
