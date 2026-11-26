"""Validation against a partial ground-truth label, plus a volume-only baseline.

Each source names its label in config (`validation_label`):
  * UCI      -> has_known_problem : incident linked to a known Problem record
  * Mendeley -> is_duplicate      : ticket closed as a duplicate of another report

Both are INCOMPLETE ground truth: the point is to find problems nobody has raised,
so a candidate without the label is not a false positive. The checks only test that
the method points somewhere sensible:
  * is the label over-represented inside candidates?
  * does the rubric's top of the backlog capture more labelled incidents than
    ranking by ticket volume alone (the approach the brief rules out)?
The label is never used as a detection signal, so this is not circular.
"""
from __future__ import annotations

import pandas as pd


def enrichment(inc: pd.DataFrame, bridge: pd.DataFrame, label: str = "has_known_problem",
               label_name: str = "linked to a known Problem") -> dict:
    in_cand = inc["number"].isin(bridge["number"])
    y = inc[label].astype(bool)
    rate_in, rate_out = y[in_cand].mean(), y[~in_cand].mean()
    out = {
        "validation_label": label_name,
        "incidents_in_candidates": int(in_cand.sum()),
        "share_of_incidents_in_candidates": round(float(in_cand.mean()), 3),
        "labelled_incidents": int(y.sum()),
        "label_rate_inside": round(float(rate_in), 4) if in_cand.any() else None,
        "label_rate_outside": round(float(rate_out), 4) if (~in_cand).any() else None,
        "enrichment_ratio": round(float(rate_in / rate_out), 2) if rate_out else None,
    }
    if label == "has_known_problem":
        multi = inc[y].groupby("problem_id").size()
        multi = set(multi[multi >= 2].index)
        caught = set(inc.loc[y & in_cand, "problem_id"]) & multi
        out["known_problems_with_2plus_incidents"] = len(multi)
        out["of_which_touched_by_a_candidate"] = len(caught)
    return out


def topk_vs_volume(inc: pd.DataFrame, reg: pd.DataFrame, label: str = "has_known_problem",
                   ks=(10, 20, 50)) -> pd.DataFrame:
    """Labelled incidents covered by the top-k backlog families, rubric vs volume ranking."""
    inc = inc.assign(family=inc["category"] + " | " + inc["subcategory"])
    y = inc[label].astype(bool)
    rubric_families = reg.loc[reg["in_backlog"].astype(bool)].sort_values("backlog_rank")["family"].tolist()
    volume_families = inc["family"].value_counts().index.tolist()
    total = int(y.sum())
    rows = []
    for k in ks:
        for name, fams in [("rubric", rubric_families), ("volume_only", volume_families)]:
            m = inc["family"].isin(fams[:k])
            rows.append({"k": k, "ranking": name, "families": len(fams[:k]), "incidents_in_scope": int(m.sum()),
                         "labelled_incidents": int(y[m].sum()),
                         "label_rate": round(float(y[m].mean()), 4) if m.any() else 0.0,
                         "share_of_all_labelled": round(y[m].sum() / total, 3) if total else None})
    return pd.DataFrame(rows)
