"""Phase 2: group related incidents into evidence-backed problem candidates.

Four transparent detectors (the non-AI baseline). Each produces candidates with
the incidents that support them, so every candidate is reproducible from source
records:

* recurring  - a service signature with a rework/failure signal well above baseline
* localised  - a signature concentrated in one location beyond what volume predicts
* burst      - many incidents of one signature at one location within 24h
* emerging   - a signature whose weekly volume is rising significantly

A candidate is a HYPOTHESIS that a shared underlying problem exists. Similar
categorisation or timing alone is not proof of a shared cause.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import stats

SIGNAL_LABELS = {
    "is_reopened": "reopened",
    "is_sla_missed": "missed SLA",
    "is_bounced": "reassigned 3+ times",
    "is_slow": "took longer than the overall P90 to resolve",
    "is_high_priority": "high/critical priority",
    "is_cannot_reproduce": "closed as 'Cannot Reproduce' (intermittent fault?)",
}


def available_signals(inc: pd.DataFrame) -> dict[str, float]:
    """Signals this source actually records, with their global rate.
    NaN columns (field not in source) and all-zero signals are skipped."""
    out = {}
    for s in SIGNAL_LABELS:
        if s in inc and inc[s].notna().any():
            rate = float(inc[s].astype(float).mean())
            if rate > 0:
                out[s] = rate
    return out


def candidate_id(kind: str, signature: str, location: str = "") -> str:
    """Stable ID so the same pattern keeps its ID across refresh runs."""
    h = hashlib.sha1(f"{kind}|{signature}|{location}".encode()).hexdigest()[:8].upper()
    return f"{kind[:3].upper()}-{h}"


def add_slow_flag(inc: pd.DataFrame) -> pd.DataFrame:
    p90 = inc["resolution_hours"].quantile(0.9)
    return inc.assign(is_slow=inc["resolution_hours"].gt(p90))


def _members(df: pd.DataFrame, cid: str) -> pd.DataFrame:
    return pd.DataFrame({"candidate_id": cid, "number": df["number"].values})


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values. Hundreds of signature/signal tests run per refresh;
    without this, chance alone produces 'findings' (verified on pure-noise test data)."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


# ---------------------------------------------------------------- recurring
def detect_recurring(inc: pd.DataFrame, cfg: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    c = cfg["candidates"]["recurring"]
    alpha = cfg["candidates"]["fdr_alpha"]
    g_rates = available_signals(inc)
    tests, groups = [], {}
    # every signature x signal pair with enough volume is a test, whether or not it looks elevated
    for sig, grp in inc.groupby("signature"):
        n = len(grp)
        if n < c["min_incidents"]:
            continue
        groups[sig] = grp
        for s, base in g_rates.items():
            label = SIGNAL_LABELS[s]
            k = int(grp[s].astype(float).sum())
            p = stats.binomtest(k, n, base, alternative="greater").pvalue
            tests.append({"signature": sig, "signal": s, "label": label, "k": k, "n": n,
                          "rate": k / n, "base": base, "lift": (k / n) / base, "p": p})
    if not tests:
        return [], []
    t = pd.DataFrame(tests)
    t["q"] = bh_adjust(t["p"].values)
    t = t[(t["q"] < alpha) & (t["lift"] >= c["lift_threshold"]) & (t["k"] >= c["min_signal_incidents"])]
    cands, members = [], []
    for sig, el in t.sort_values("p").groupby("signature", sort=False):
        cid = candidate_id("recurring", sig)
        evidence = "; ".join(
            f"{e.k} of {e.n} {e.label} ({e.rate:.0%} vs {e.base:.0%} overall, q={e.q:.1g})" for e in el.itertuples())
        cands.append({"candidate_id": cid, "type": "Recurring", "signature": sig, "location": "(all)",
                      "evidence": evidence, "min_p_value": float(el["q"].min()),
                      "elevated_signals": ",".join(el["signal"])})
        members.append(_members(groups[sig], cid))
    return cands, members


# ---------------------------------------------------------------- localised
def detect_localised(inc: pd.DataFrame, cfg: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    c = cfg["candidates"]["localised"]
    alpha = cfg["candidates"]["fdr_alpha"]
    loc_share = inc["location"].value_counts(normalize=True)
    sig_n = inc["signature"].value_counts()
    tests = []
    for (sig, loc), grp in inc.groupby(["signature", "location"]):
        n = len(grp)
        if n < c["min_incidents"] or loc == "Location (blank)":
            continue
        # how surprising is this many at this location given the signature's volume?
        p = stats.binomtest(n, int(sig_n[sig]), loc_share[loc], alternative="greater").pvalue
        tests.append((sig, loc, grp, n, n / sig_n[sig], (n / sig_n[sig]) / loc_share[loc], p))
    if not tests:
        return [], []
    q = bh_adjust(np.array([t[-1] for t in tests]))
    cands, members = [], []
    for (sig, loc, grp, n, share, lift, p), qv in zip(tests, q):
        if qv >= alpha or share < c["min_share_of_signature"] or lift < c["min_lift_vs_location_mix"]:
            continue
        cid = candidate_id("localised", sig, loc)
        evidence = (f"{n} of {sig_n[sig]} incidents for this signature ({share:.0%}) are at {loc}, "
                    f"which handles {loc_share[loc]:.1%} of all incidents ({lift:.1f}x expected, q={qv:.1g})")
        cands.append({"candidate_id": cid, "type": "Localised", "signature": sig, "location": loc,
                      "evidence": evidence, "min_p_value": float(qv), "elevated_signals": "location_concentration"})
        members.append(_members(grp, cid))
    return cands, members


# ---------------------------------------------------------------- burst
def _burst_episodes(times: np.ndarray, window: np.timedelta64, min_n: int) -> list[tuple[int, int]]:
    """Greedy non-overlapping windows [i, j) holding >= min_n events within `window`."""
    episodes, i = [], 0
    while i < len(times):
        j = int(np.searchsorted(times, times[i] + window, side="right"))
        if j - i >= min_n:
            episodes.append((i, j))
            i = j
        else:
            i += 1
    return episodes


def detect_bursts(inc: pd.DataFrame, cfg: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    c = cfg["candidates"]["burst"]
    window = np.timedelta64(c["window_hours"], "h")
    cands, members = [], []
    for (sig, loc), grp in inc.groupby(["signature", "location"]):
        if len(grp) < c["min_incidents"] or loc == "Location (blank)":
            continue
        grp = grp.sort_values("opened_at")
        times = grp["opened_at"].values
        eps = _burst_episodes(times, window, c["min_incidents"])
        if not eps:
            continue
        idx = np.concatenate([np.arange(i, j) for i, j in eps])
        burst = grp.iloc[idx]
        biggest = max(eps, key=lambda e: e[1] - e[0])
        start = pd.Timestamp(times[biggest[0]])
        cid = candidate_id("burst", sig, loc)
        evidence = (f"{len(eps)} burst episode(s) at {loc}; largest: {biggest[1] - biggest[0]} incidents "
                    f"within {c['window_hours']}h from {start:%Y-%m-%d %H:%M}")
        # crude significance: chance of the largest burst if the signature/location's
        # incidents were spread evenly over the window (Poisson on 24h rate)
        span_h = max((inc["opened_at"].max() - inc["opened_at"].min()).total_seconds() / 3600, 1)
        lam = len(grp) * c["window_hours"] / span_h
        p = float(stats.poisson.sf(biggest[1] - biggest[0] - 1, lam))
        if p > c["max_p_value"]:
            continue
        cands.append({"candidate_id": cid, "type": "Burst", "signature": sig, "location": loc,
                      "evidence": evidence, "min_p_value": p, "elevated_signals": "temporal_burst",
                      "burst_episodes": len(eps),
                      "burst_start": start, "burst_end": pd.Timestamp(times[biggest[1] - 1])})
        members.append(_members(burst, cid))
    return cands, members


def link_co_occurring_bursts(cand: pd.DataFrame, hours: float) -> pd.Series:
    """For each burst, list other-signature bursts at the same location overlapping in time.

    Several signatures spiking together at one site points to one shared event
    (e.g. a local outage logged under different categories) rather than several problems.
    """
    out = pd.Series("", index=cand.index, dtype=object)
    bursts = cand[cand["type"] == "Burst"]
    slack = pd.Timedelta(hours=hours)
    for loc, grp in bursts.groupby("location"):
        for i, a in grp.iterrows():
            linked = [b["candidate_id"] for j, b in grp.iterrows()
                      if j != i and a["burst_start"] <= b["burst_end"] + slack and b["burst_start"] <= a["burst_end"] + slack]
            out[i] = ",".join(linked)
    return out


# ---------------------------------------------------------------- emerging
def detect_emerging(inc: pd.DataFrame, cfg: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    c = cfg["candidates"]["emerging"]
    weeks = sorted(inc["opened_week"].unique())
    if len(weeks) < c["recent_weeks"] + 4:
        return [], []
    recent_weeks = weeks[-c["recent_weeks"]:]
    earlier_weeks = weeks[:-c["recent_weeks"]]
    # normalise for overall demand: compare the signature's SHARE of incidents,
    # so a busy week for everyone doesn't look like an emerging problem
    tot_recent = inc["opened_week"].isin(recent_weeks).sum()
    tot_earlier = inc["opened_week"].isin(earlier_weeks).sum()
    tests = []
    for sig, grp in inc.groupby("signature"):
        is_recent = grp["opened_week"].isin(recent_weeks)
        r, e = int(is_recent.sum()), int((~is_recent).sum())
        if r < c["min_recent_incidents"]:
            continue
        expected = max(e, 0.5) * tot_recent / tot_earlier   # 0.5 = continuity floor for brand-new signatures
        tests.append((sig, grp[is_recent], r, expected, r / expected, float(stats.poisson.sf(r - 1, expected))))
    if not tests:
        return [], []
    q = bh_adjust(np.array([t[-1] for t in tests]))
    cands, members = [], []
    for (sig, recent_rows, r, expected, ratio, p), qv in zip(tests, q):
        if ratio < c["min_rate_ratio"] or qv > c["max_q_value"]:
            continue
        cid = candidate_id("emerging", sig)
        evidence = (f"{r} incidents in the last {c['recent_weeks']} weeks vs {expected:.1f} expected from "
                    f"the earlier {len(earlier_weeks)} weeks' share of demand ({ratio:.1f}x, q={qv:.1g})")
        cands.append({"candidate_id": cid, "type": "Emerging", "signature": sig, "location": "(all)",
                      "evidence": evidence, "min_p_value": float(qv), "elevated_signals": "rising_trend"})
        members.append(_members(recent_rows, cid))
    return cands, members


DETECTORS = {"recurring": detect_recurring, "localised": detect_localised,
             "burst": detect_bursts, "emerging": detect_emerging}
CANDIDATE_COLS = ["candidate_id", "type", "signature", "location", "evidence", "min_p_value",
                  "elevated_signals", "burst_start", "burst_end", "burst_episodes"]


def generate(inc: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all detectors. Returns (candidates, candidate->incident bridge)."""
    inc = add_slow_flag(inc)
    cands, members = [], []
    # each source enables only the detectors whose assumptions its data meets
    for name in cfg.get("detectors", list(DETECTORS)):
        c, m = DETECTORS[name](inc, cfg)
        cands += c
        members += m
    cand = pd.DataFrame(cands, columns=list(dict.fromkeys(
        [k for c in cands for k in c] + CANDIDATE_COLS)))
    bridge = pd.concat(members, ignore_index=True) if members else pd.DataFrame(columns=["candidate_id", "number"])
    if cand.empty:   # a quiet period is a valid outcome: return an empty, well-formed table
        return cand.assign(parent_candidate_id=None, co_occurring_bursts=""), bridge
    # link nested candidates (e.g. a burst inside a recurring signature) to their parent,
    # so the register shows them as related rather than as unrelated duplicates
    parents = cand[cand["type"] == "Recurring"].set_index("signature")["candidate_id"]
    cand["parent_candidate_id"] = np.where(cand["type"] != "Recurring", cand["signature"].map(parents), None)
    if "burst_start" not in cand:
        cand["burst_start"] = pd.NaT
        cand["burst_end"] = pd.NaT
    cand["co_occurring_bursts"] = link_co_occurring_bursts(cand, cfg["candidates"]["burst"]["co_occurrence_hours"])
    return cand, bridge
