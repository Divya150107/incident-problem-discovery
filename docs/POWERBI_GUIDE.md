# Building the Power BI report

The Python pipeline produces model-ready CSVs; Power BI is the presentation layer.
Re-running the pipeline and pressing **Refresh** in Power BI is the whole refresh cycle.

## 1. Load the data

1. Run the pipeline for one or both sources (see README), so `outputs/powerbi/uci/` and/or
   `outputs/powerbi/mendeley/` exist. Both folders hold the same tables with the same columns.
2. In Power BI Desktop: **Transform data → Manage Parameters → New**: `OutputFolder` (Text) =
   full path to `outputs\powerbi\uci\` (with trailing backslash). Changing this one parameter
   to `...\mendeley\` and refreshing switches the whole report to the other dataset.
   (Optional: make the parameter a list of the two paths so viewers can switch from a dropdown.)
3. **Get data → Text/CSV** for each file below. In Power Query, change each source line to
   `= Csv.Document(File.Contents(OutputFolder & "candidate_register.csv"), [Delimiter=",", Encoding=65001])`
   so the whole report repoints by changing one parameter.
4. Check types: `opened_at`, `resolved_at`, `first_seen`, `last_seen`, `run_at` → Date/Time;
   `opened_date`, `opened_week` → Date; `is_*`, `has_*`, `in_backlog` → True/False; scores → Decimal.

Tables: `fact_incident`, `candidate_register`, `bridge_candidate_incident`, `candidate_factors`,
`escalation_queue`, `candidate_history`, `refresh_log`, `data_quality`, `baseline_summary`,
`baseline_weekly_category`, `validation_topk_vs_volume`, plus (optional) `outputs/text_prototype/scores_vs_baseline.csv`.

## 2. Model (Model view)

| From | To | Cardinality | Cross-filter |
|---|---|---|---|
| candidate_register[candidate_id] | bridge_candidate_incident[candidate_id] | 1 : * | Single |
| fact_incident[number] | bridge_candidate_incident[number] | 1 : * | **Both** (so picking a candidate filters its incidents) |
| candidate_register[candidate_id] | candidate_factors[candidate_id] | 1 : * | Single |
| candidate_register[candidate_id] | escalation_queue[candidate_id] | 1 : 1 | Single |
| Date[Date] | fact_incident[opened_date] | 1 : * | Single |

Leave `candidate_history` unrelated (closed candidates are not in the current register) and
slice it on its own columns. Create the date table (it follows whichever dataset is loaded):

```DAX
Date = CALENDAR(MIN(fact_incident[opened_date]), MAX(fact_incident[opened_date]))
```

Fields a source doesn't record (e.g. SLA for Mendeley) arrive blank, so cards and measures show
blank rather than 0%. Leave them blank; don't replace them with zero.

## 3. Measures

```DAX
Incidents = COUNTROWS(fact_incident)
Reopen % = DIVIDE(CALCULATE([Incidents], fact_incident[is_reopened] = TRUE()), [Incidents])
SLA missed % = DIVIDE(CALCULATE([Incidents], fact_incident[is_sla_missed] = TRUE()), [Incidents])
Reassigned 3+ % = DIVIDE(CALCULATE([Incidents], fact_incident[is_bounced] = TRUE()), [Incidents])
Median resolution h = MEDIAN(fact_incident[resolution_hours])

Candidates = COUNTROWS(candidate_register)
Backlog items = CALCULATE([Candidates], candidate_register[in_backlog] = TRUE())
New this run = CALCULATE([Candidates], candidate_register[lifecycle_status] = "New")
Proposed escalations = COUNTROWS(escalation_queue)
Awaiting review = CALCULATE([Candidates], candidate_register[review_decision] = "Awaiting review")

Latest successful refresh =
    CALCULATE(MAX(refresh_log[run_at]), refresh_log[status] = "success")
Analysis as-of =
    CALCULATE(MAX(refresh_log[as_of]), refresh_log[status] = "success")
Last run failed? =
    VAR lastRun = MAX(refresh_log[run_at])
    RETURN IF(CALCULATE(MAX(refresh_log[status]), refresh_log[run_at] = lastRun) = "failed",
              "⚠ Last refresh failed - showing previous successful data", "OK")
```

## 4. Pages (mapped to the brief's "Dashboard and Solution Expectations")

**1 · Operations overview (landing view)**
- Cards: Incidents, Backlog items, New this run, Proposed escalations, Latest successful refresh, Analysis as-of, Last run failed?
- Table, top 10 backlog (filter `in_backlog = True`, sort `backlog_rank`): rank, family, type,
  priority_score, **confidence** (separate column, conditional colour), known_problem_status, lifecycle_status.
- Clustered bar: Backlog items by type and category.
- Line: Incidents and SLA missed % by Date[Date] (week).

**2 · Candidate register**
- Slicers: type, confidence, known_problem_status, lifecycle_status, review_decision.
- Table with: candidate_id, signature, location, incidents, evidence, hypothesis,
  alternative_explanations, evidence_gaps, known_problem_ids, confidence, priority_score.
- Add a **drill-through** on candidate_id to page 3.

**3 · Candidate detail (drill-through target)**
- Multi-row card: hypothesis, evidence, alternative_explanations, evidence_gaps, proposed_next_step, required_expertise.
- Bar: `candidate_factors[contribution]` by `factor`. This shows *why* it ranks where it does.
- Separate card: confidence + confidence_reason (not merged with priority).
- Table of linked incidents (from fact_incident via the bridge): number, opened_at, location,
  assignment_group, priority, is_reopened, is_sla_missed, reassignment_count, resolution_hours, problem_id, rfc.
- Column chart: linked incidents by opened_date (temporal pattern); bar: by assignment_group and by location.

**4 · Investigation backlog & escalation**
- Table from escalation_queue: backlog_rank, candidate_id, family, route_to, reason_for_routing,
  priority_score, confidence, proposed_next_step, escalation_status, re_proposed.
- Text box: "Proposals only. Problem Management decides. Nothing is created or changed automatically."

**5 · Continuous operation**
- Line/column: candidate_history count by as_of, legend lifecycle_status (New / Persistent / Closed).
- Table: refresh_log (as_of, run_at, status, incidents_analysed, candidates, new, closed, error).

**6 · Data quality & method**
- Table: data_quality (check, count, pct, note).
- Clustered column: validation_topk_vs_volume `label_rate` by k, legend ranking (rubric vs volume_only).
  Title it from `validation_summary[validation_label]` ("linked to a known Problem" / "closed as Duplicate").
- Cards: `validation_summary[enrichment_ratio]`, `label_rate_inside`, `label_rate_outside`.
- Table: rubric weights (enter manually or from config) and text prototype scores_vs_baseline (F1 by method).

**7 · SLA & resolution (UCI)**: from `insight_sla_breakdown`
- Slicer on `dimension`; bar chart of `sla_miss_rate` by `value`; table with incidents and median/P90 resolution hours.
- Text box: "Source records SLA met/missed only - no targets or clock pauses."

**8 · Problem & change links (UCI)**: from `insight_problem_links`, `insight_problem_link_comparison`,
`insight_problem_link_size`, `insight_change_link_comparison`
- Clustered bars, linked vs not linked: SLA miss, P1/P2 share, median resolution.
- Column chart: Problems by incidents per Problem (shows that most Problems have one incident).
- Table: per-Problem links with median hours to link. Card: the change-coverage note.

**9 · Resolution & workflow (Mendeley)**: from `insight_resolution_mix`, `insight_workflow_steps(_by_product)`
- 100% stacked bar: outcome mix by product.
- Bar: `share_of_total_time` by step (where ticket time goes); matrix of product × step median hours.

**10 · Handoffs (Mendeley)**: from `insight_handoff_distribution`, `insight_handoffs_by_product`
- Column: tickets and median resolution hours by holders bucket (1 / 2 / 3 / 4+).
- Table: per product. Note on the page: "Counts of ticket holders only; no individual is identified or ranked."

**11 · Ticket text analysis (Zenodo)**: load `outputs/text_zenodo/*.csv` with a second parameter `TextFolder`
- Table: `cluster_vs_category` (ARI / NMI / purity / coverage per method) and `supervised_metrics`.
- Matrix: `cluster_category_crosstab` (clusters × categories) with conditional formatting, to show which categories blur together.
- Bar: `support_general_subthemes` tickets by top_terms ("what the catch-all category hides").
- Table: `near_duplicate_groups` filtered to `categories > 1` (near-identical tickets with different human labels).
- Bar: `category_profile[within_category_similarity]` by category.

These insight tables exist only in the full-run output (not the weekly replays), and each source folder
contains only its own `insight_*` tables. When `OutputFolder` points at the other source, visuals on that
source's pages show blank. That's expected; you can hide those pages per source.

## 5. Security notes for a real deployment
- Publish to an access-controlled workspace; apply row-level security if resolver groups should see only their routes.
- Free-text descriptions (when used) should only reach Power BI after `scrub_pii` and only on pages with restricted access.
- Test access with a non-admin account before release (brief: "test dashboard access before release").
