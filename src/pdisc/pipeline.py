"""Repeatable analysis run: extract -> cleanse -> baseline -> candidates -> register -> history -> export.

Usage (PYTHONPATH=src):
    python -m pdisc.pipeline --source uci                 # one run, as of the end of the window
    python -m pdisc.pipeline --source mendeley --as-of 2022-06-26
    python -m pdisc.pipeline --source uci --backfill --reset-history   # weekly replay across the window

Outputs go to outputs/powerbi/<source>/ via a staging folder that is only swapped in
when the whole run succeeds, so a failed run never leaves the dashboard half-refreshed;
the failure is still recorded in the refresh log.
"""
from __future__ import annotations

import argparse
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from pdisc import baseline, candidates, evaluate, history, insights, register
from pdisc.sources import SOURCES

ROOT = Path(__file__).resolve().parents[2]

FACT_COLS = ["number", "opened_at", "opened_date", "opened_week", "resolved_at", "signature", "category",
             "subcategory", "u_symptom", "location", "assignment_group", "contact_type", "priority", "impact",
             "urgency", "incident_state", "closed_code", "is_high_priority", "is_high_impact", "is_reopened",
             "is_sla_missed", "is_bounced", "is_resolved", "reopen_count", "reassignment_count",
             "resolution_hours", "has_known_problem", "problem_id", "has_change", "rfc"]
OPTIONAL_FACT_COLS = ["project", "issue_type", "is_cannot_reproduce", "is_duplicate"]


def load_config(source: str = "uci") -> dict:
    """Shared settings with the source's block layered on top."""
    cfg = yaml.safe_load((ROOT / "config" / "rubric.yaml").read_text())
    merged = {k: v for k, v in cfg.items() if k != "sources"}
    merged.update(cfg["sources"][source])
    merged["source"] = source
    return merged


def run(as_of: str | None, cfg: dict, raw=None, out_dir: Path | None = None,
        hist_dir: Path | None = None, reviews_path: Path | None = None) -> dict:
    src = SOURCES[cfg["source"]]
    out_dir = out_dir or ROOT / "outputs" / "powerbi" / cfg["source"]
    hist_dir = hist_dir or ROOT / "data" / "history" / cfg["source"]
    hist_dir.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now().isoformat(timespec="seconds")
    as_of_ts = (pd.Timestamp(as_of or cfg["analysis_window"]["end"]).normalize()
                + pd.Timedelta(hours=23, minutes=59))
    run_id = f"{as_of_ts:%Y%m%d}-{datetime.now():%H%M%S%f}"
    log_path = hist_dir / "refresh_log.csv"
    log = history.load_csv(log_path, [])
    entry = {"run_id": run_id, "source": cfg["source"], "as_of": str(as_of_ts.date()), "run_at": run_at,
             "method_version": cfg["method_version"], "status": "failed"}
    try:
        raw = raw if raw is not None else src.load(cfg)
        # a default run uses all data (so out-of-window data is still reported as a coverage gap);
        # an --as-of run replays the data as it stood on that date
        inc, out_win = src.incidents(raw, cfg, as_of_ts if as_of else None)
        inc = inc.assign(validation_label=inc[cfg["validation_label"]].astype(bool))

        cand, bridge = candidates.generate(inc, cfg)
        reg, factors = register.build_register(inc, cand, bridge, cfg)

        hist_path = hist_dir / "candidate_history.csv"
        hist = history.load_csv(hist_path, history.HISTORY_COLS)
        reg, new_hist_rows = history.lifecycle(reg, hist, run_id, as_of_ts)
        reviews = history.load_csv(reviews_path or ROOT / "config" / "reviews.csv", history.REVIEW_COLS)
        reg = history.apply_reviews(reg, reviews)
        queue = history.escalation_queue(reg, cfg)
        hist = pd.concat([hist, new_hist_rows], ignore_index=True)

        enr = evaluate.enrichment(inc, bridge, cfg["validation_label"], cfg["validation_label_name"])
        topk = evaluate.topk_vs_volume(inc, reg, cfg["validation_label"])

        stage = out_dir.parent / (out_dir.name + "_staging")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        fact_cols = FACT_COLS + [c for c in OPTIONAL_FACT_COLS if c in inc]
        tables = {
            "fact_incident": inc[fact_cols],
            "candidate_register": reg.drop(columns=["burst_start", "burst_end"], errors="ignore"),
            "bridge_candidate_incident": bridge,
            "candidate_factors": factors,
            "escalation_queue": queue,
            "baseline_summary": baseline.summary_table(inc),
            "baseline_weekly_category": baseline.weekly_by_category(inc),
            "data_quality": src.quality(raw, inc, out_win, cfg),
            "validation_summary": pd.DataFrame([enr]),
            "validation_topk_vs_volume": topk,
            "candidate_history": hist,
        }
        if not as_of:   # descriptive insights on full runs only (some inputs are final-state totals)
            tables.update(insights.run(cfg.get("insights", []), inc, raw, cfg))
        for name, df in tables.items():
            df.to_csv(stage / f"{name}.csv", index=False)

        counts = reg["lifecycle_status"].value_counts()
        entry.update({
            "status": "success", "incidents_analysed": len(inc), "outside_window": len(out_win),
            "data_latest_opened_at": str(inc["opened_at"].max()), "candidates": len(reg),
            "backlog_families": int(reg["in_backlog"].astype(bool).sum()), "proposed_escalations": len(queue),
            "new": int(counts.get("New", 0)), "persistent": int(counts.get("Persistent", 0)),
            "closed": int((new_hist_rows["lifecycle_status"] == "Closed").sum()),
        })
        log = pd.concat([log, pd.DataFrame([entry])], ignore_index=True)
        log.to_csv(stage / "refresh_log.csv", index=False)

        # commit: history first, then swap outputs in
        hist.to_csv(hist_path, index=False)
        log.to_csv(log_path, index=False)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        stage.rename(out_dir)
    except Exception as exc:  # record the failure, keep the last good outputs
        entry["error"] = f"{type(exc).__name__}: {exc}"
        log = pd.concat([log, pd.DataFrame([entry])], ignore_index=True)
        log.to_csv(log_path, index=False)
        if out_dir.exists():
            log.to_csv(out_dir / "refresh_log.csv", index=False)
        traceback.print_exc()
    return entry


def backfill_dates(cfg: dict, warmup_weeks: int = 7, every_weeks: int = 1) -> list[pd.Timestamp]:
    start = pd.Timestamp(cfg["analysis_window"]["start"]) + pd.Timedelta(weeks=warmup_weeks)
    end = pd.Timestamp(cfg["analysis_window"]["end"]).normalize()
    return [d for d in pd.date_range(start, end, freq=f"{every_weeks}W-SUN") if d < end]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="uci", choices=sorted(SOURCES))
    ap.add_argument("--as-of")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--every-weeks", type=int, default=1, help="backfill spacing (e.g. 4 for a 2-year window)")
    ap.add_argument("--reset-history", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.source)
    hist_dir = ROOT / "data" / "history" / a.source
    if a.reset_history and hist_dir.exists():
        shutil.rmtree(hist_dir)
    if a.backfill:
        raw = SOURCES[a.source].load(cfg)
        # point-in-time replays for past weeks, then a normal full-data run as the latest refresh
        for d in [*backfill_dates(cfg, every_weeks=a.every_weeks), None]:
            e = run(str(d.date()) if d is not None else None, cfg, raw)
            print(f"{e['as_of']}: {e['status']}  candidates={e.get('candidates')}  new={e.get('new')}  "
                  f"persistent={e.get('persistent')}  closed={e.get('closed')}  "
                  f"escalations={e.get('proposed_escalations')}  {e.get('error', '')}")
    else:
        print(run(a.as_of, cfg))


if __name__ == "__main__":
    main()
