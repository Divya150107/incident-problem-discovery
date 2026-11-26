# Data dictionary

All tables are written by `python -m pdisc.pipeline --source <uci|mendeley>` to
`outputs/powerbi/<source>/` as CSV, with the same table and column names for both sources.
Personal identifiers (callers/reporters, assignees, resolvers, creators/updaters) are removed
before anything is written. Requesters appear only as a count (`distinct_callers`).
A field the source doesn't record is left **blank (NaN)**, never filled with 0 or False.

## Source: UCI "Incident management process enriched event log"

ServiceNow event log, 141,712 update rows for 24,918 incidents, anonymised
(categories, locations and groups are codes such as `Category 55`). Analysis window:
29 Feb – 29 May 2016 (13 complete weeks, 23,972 incidents). See `data_quality.csv` for gaps.

## Source: Mendeley "Help Desk Tickets" (DOI 10.17632/btm76zndnt.2, CC BY 4.0)

Customer helpdesk of an international software company: `issues.csv` (66,691 issues, 2007–2023) and
`issues_change_history.csv` (257,508 status/assignee changes). Only `issue_type = Ticket` is analysed,
window 4 Jan 2021 – 12 Mar 2023 (11,691 tickets). The staff-appraisal files in the dataset are not used.

| Shared column | Mendeley mapping |
|---|---|
| number | `MHD-` + issue id |
| category / subcategory / u_symptom | Product / Country / Client, decoded from the masked project code `C06` + `gg` + `__z0` (country + product + client); projects without the pattern → `Internal` |
| location | Country |
| assignment_group | `(not recorded)`. Escalations route to "Service owner: product \| country" |
| is_reopened, reopen_count | "reopened" status changes in the history (up to the as-of date) |
| reassignment_count, is_bounced | distinct ticket holders − 1; bounced = 3+ holders (count only) |
| is_high_priority | priority Highest or Blocker |
| is_high_impact, is_sla_missed | blank: not in source |
| is_cannot_reproduce | resolution = Cannot Reproduce (detection signal) |
| is_duplicate | resolution = Duplicate (**validation label only**, never a signal) |
| project, issue_type | raw project code and issue type (extra columns) |

## fact_incident (one row per incident, final or as-of state)

| Column | Meaning |
|---|---|
| number | Incident reference (source record key) |
| opened_at / opened_date / opened_week | When opened; week starts Monday |
| resolved_at, resolution_hours | Resolution time; null if unresolved or negative in source |
| signature | `category \| subcategory \| symptom`, the service-signature proxy (no CI data) |
| category, subcategory, u_symptom | Service classification; blanks shown as `(blank)` |
| location, assignment_group, contact_type | Where / who handled (group, never person) / channel |
| priority, impact, urgency | Ticket fields, unchanged. **Not** the investigation priority |
| is_high_priority | Priority 1 or 2 |
| is_high_impact | Impact = 1 - High |
| is_reopened | reopen_count > 0 |
| is_sla_missed | made_sla = false |
| is_bounced | reassigned 3+ times (ping-pong between groups) |
| has_known_problem, problem_id | Linked to a known Problem record |
| has_change, rfc | Linked to a change (RFC) |

## candidate_register (one row per problem candidate)

| Column | Meaning |
|---|---|
| candidate_id | Stable ID (`REC-`, `LOC-`, `BUR-`, `EME-` + hash of type/signature/location). Same pattern → same ID across runs |
| type | Recurring / Localised / Burst / Emerging (see README) |
| signature, family, location | What the pattern is about; family = category \| subcategory |
| evidence | The statistic that triggered the candidate, with q/p-value |
| hypothesis | Possible shared cause, worded as a hypothesis. Never a confirmed cause |
| alternative_explanations | Generated challenges (routing habit, normal demand, duplicates, SLA mis-set, change-induced…) |
| evidence_gaps | What data would be needed to confirm or reject |
| proposed_next_step, required_expertise | Suggested investigation action and who |
| incidents, distinct_callers, locations, weeks_active, first_seen, last_seen | Scope |
| dominant_group (+_share), dominant_location (+_share) | Where the pattern concentrates |
| reopen_rate, sla_miss_rate, bounce_rate, slow_rate, high_priority_rate, high_impact_rate, cannot_reproduce_rate | Signal rates (blank where the source lacks the field) |
| validation_label_rate | Share of the candidate's incidents carrying the source's validation label |
| burst_episodes | Number of separate bursts (Burst candidates) |
| median_resolution_hours, total_resolution_hours, total_reassignments | Operational burden |
| trend_ratio, recent_incidents | Recent 3-week share of demand vs earlier weeks |
| known_problem_status, known_problem_ids | Match against known Problems |
| shared_change_ids | Changes linked to 2+ of the candidate's incidents |
| impact, recurrence, deterioration, burden | Factor scores 0–1 |
| priority_score, priority_rank, priority_rationale | Proposed investigation priority (weighted factors) and top drivers |
| confidence, confidence_reason | Evidence confidence High/Medium/Low, kept **separate** from priority |
| in_backlog, backlog_rank, related_candidates | One backlog entry per family; related IDs listed |
| parent_candidate_id, co_occurring_bursts | Nesting and same-site/same-time burst links |
| lifecycle_status, first_detected_as_of, priority_change | New / Persistent vs previous run |
| review_decision, review_reason | From `config/reviews.csv` (Awaiting review by default) |
| example_incidents | First 10 linked incident numbers |

## Other tables

| Table | Grain | Use |
|---|---|---|
| bridge_candidate_incident | candidate × incident | Drill from a candidate to its source records |
| candidate_factors | candidate × factor | Priority breakdown chart (score, weight, contribution) |
| escalation_queue | proposed escalation | Candidates proposed for routing to a resolver group. **Proposals only** |
| candidate_history | run × candidate | New / Persistent / Closed across refreshes |
| refresh_log | run | Latest successful refresh, coverage, failures |
| data_quality | check | Missing fields, duplicates, coverage gaps |
| baseline_summary, baseline_weekly_category | metric / week × category | Phase 1 baseline and trend pages |
| validation_summary | — | Label enrichment inside vs outside candidates (`label_rate_inside`, `label_rate_outside`, `enrichment_ratio`); label = known Problem (UCI) or Duplicate (Mendeley) |
| validation_topk_vs_volume | k × ranking | Labelled incidents covered by the top-k families, rubric vs volume-only (`label_rate`, `share_of_all_labelled`) |

## Insight tables (`insight_*.csv`, full runs only)

| Table | Source | Grain / content |
|---|---|---|
| insight_sla_breakdown | UCI | dimension (priority, contact_type, family, assignment_group, opened_week) × value: incidents, sla_missed, sla_miss_rate, median/P90 resolution hours. SLA is a met/missed flag only |
| insight_problem_links | UCI | per known Problem: incidents, families, locations, groups, first/last opened, median hours from incident open to first update carrying the Problem ID, reopen/SLA rates, example incidents |
| insight_problem_link_comparison | UCI | linked vs not linked: reopen, SLA miss, 3+ reassignment, P1/P2 rates, median resolution hours |
| insight_problem_link_size | UCI | number of Problems with 1, 2, 3, 4, 5+ linked incidents |
| insight_change_link_comparison | UCI | RFC-linked vs not, plus a note on why change-induced analysis isn't possible |
| insight_resolution_mix | Mendeley | product × outcome (Done / Won't Do / Duplicate / Cannot Reproduce / Unresolved): tickets, share of product, overall share |
| insight_workflow_steps | Mendeley | per workflow step: tickets entering it, median hours (among tickets that entered it), total hours, share of tickets, share of all ticket time |
| insight_workflow_steps_by_product | Mendeley | the same by product, plus share of the product's time |
| insight_handoff_distribution | Mendeley | 1 / 2 / 3 / 4+ ticket holders: tickets, share, median resolution, reopen and Cannot Reproduce rates |
| insight_handoffs_by_product | Mendeley | per product: median holders, share with 3+ holders, median hours to first assignment, median resolution. Counts only, never per person |

## Real-text analysis (`outputs/text_zenodo/`)

| Table | Content |
|---|---|
| summary | ticket counts, templated / exact-duplicate share, near-duplicate group counts, best test ARI, supervised metrics, caveats |
| category_profile | per human category: tickets, median length, templated and duplicate share, corrupted characters, within-category similarity (mean pairwise cosine), rough language mix |
| near_duplicate_groups | groups of 3+ tickets with cosine ≥ 0.9 (char n-grams): size, kind (templated request / other), number of distinct human categories, majority category and share, example |
| cluster_vs_category | test-set ARI, NMI, purity, cluster count and coverage for the rule baseline, word TF-IDF and char TF-IDF |
| threshold_sweep_train | the training-split sweep used to choose each method's distance threshold |
| cluster_category_crosstab | test clusters × human categories, with each cluster's top terms |
| support_general_subthemes | themes inside "Support general": tickets, share, templated share, top terms, example |
| supervised_metrics | this project's classifier vs the published model (accuracy, macro-F1) |
| label_disagreements | test tickets where the classifier disagrees with the human label, by confidence: candidates for mislabelling |

## Reviewer feedback file: `config/reviews.csv`

Owned by Problem Management. Columns: `candidate_id, decision (Accepted/Rejected/Deferred), reason,
reviewer_role, reviewed_on, priority_at_review`. Read on every run. A rejected candidate is not
proposed for escalation again unless its priority rises 0.10+ above `priority_at_review`.
