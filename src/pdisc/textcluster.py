"""Text-analysis prototype: group tickets by description, compared with a non-AI baseline.

Pipeline: scrub personal data from free text -> TF-IDF (words + bigrams) ->
agglomerative clustering on cosine distance -> clusters of >= min_size tickets,
each explained by its top terms and example tickets.

Baseline (what the main pipeline can do without text): group by category.

Scored on the synthetic set's planted problems. For each planted problem the best
matching group is found and precision / recall / F1 reported. The clustering
threshold is tuned on a separate synthetic draw (dev seed), then scored on the
evaluation draw, so the reported numbers aren't tuned on the data they're measured on.

Scope note: clustering only finds groups of similar tickets. Deciding whether a
group is a PROBLEM (vs routine demand like password resets) still needs the
signals the main pipeline uses: rework, SLA, trend, concentration, bursts.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
    (re.compile(r"\+?\d[\d\s-]{7,}\d"), "<phone>"),
    (re.compile(r"\bext\.?\s*\d{3,5}\b", re.I), "<phone>"),
    (re.compile(r"\b(regards|from|thanks|cheers)\s*,?\s+[A-Z][a-z]+\b"), r"\1 <name>"),
    (re.compile(r"\s-\s[A-Z][a-z]+\b"), " - <name>"),
    (re.compile(r"\b[A-Z][a-z]+'s\b"), "<name>'s"),
]


def scrub_pii(text: str) -> str:
    for pat, rep in PII_PATTERNS:
        text = pat.sub(rep, text)
    return text


def cluster_descriptions(texts: pd.Series, distance_threshold: float, min_size: int = 5,
                         vectorizer: TfidfVectorizer | None = None, fit: bool = True):
    """Cluster ticket texts. Default: word TF-IDF (unchanged behaviour). A custom
    `vectorizer` (e.g. character n-grams for multilingual text) is clustered on a
    precomputed cosine-distance matrix, which keeps memory small for wide feature sets.
    With fit=False the vectorizer must already be fitted (e.g. on a training split)."""
    vec = vectorizer or TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, min_df=3,
                                        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    # placeholders and sign-off words carry no meaning about the fault; keep them out of the features
    cleaned = texts.str.replace(r"<\w+>|\b(regards|from|thanks|cheers|ext|call me on)\b", " ", regex=True, case=False)
    X = vec.fit_transform(cleaned.str.lower()) if fit else vec.transform(cleaned.str.lower())
    if vectorizer is None:
        model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                        distance_threshold=distance_threshold)
        labels = model.fit_predict(X.toarray())
    else:
        from sklearn.metrics.pairwise import cosine_distances
        D = np.clip(cosine_distances(X), 0, 2)
        model = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                        distance_threshold=distance_threshold)
        labels = model.fit_predict(D)
    sizes = pd.Series(labels).value_counts()
    small = sizes[sizes < min_size].index
    labels = np.where(np.isin(labels, small), -1, labels)       # -1 = not grouped
    terms = np.array(vec.get_feature_names_out())
    top = {}
    for c in np.unique(labels[labels >= 0]):
        centroid = np.asarray(X[labels == c].mean(axis=0)).ravel()
        top[c] = ", ".join(terms[centroid.argsort()[::-1][:5]])
    return labels, top


def score_groups(truth: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Best-matching group per planted problem -> precision, recall, F1."""
    rows = []
    for pid in sorted(t for t in truth.unique() if t != "none"):
        is_p = truth.eq(pid)
        best = {"problem": pid, "best_group": None, "precision": 0.0, "recall": 0.0, "f1": 0.0, "group_size": 0}
        for g in groups[is_p].unique():
            if g in (-1, "-1"):
                continue
            in_g = groups.eq(g)
            tp = (in_g & is_p).sum()
            prec, rec = tp / in_g.sum(), tp / is_p.sum()
            f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
            if f1 > best["f1"]:
                best.update(best_group=g, precision=prec, recall=rec, f1=f1, group_size=int(in_g.sum()))
        rows.append(best)
    return pd.DataFrame(rows)


def tune_threshold(df_dev: pd.DataFrame, grid=np.arange(0.5, 0.96, 0.05)) -> tuple[float, pd.DataFrame]:
    texts = df_dev["description"].map(scrub_pii)
    res = []
    for t in grid:
        labels, _ = cluster_descriptions(texts, t)
        s = score_groups(df_dev["true_problem"], pd.Series(labels))
        res.append({"distance_threshold": round(float(t), 2), "mean_f1": s["f1"].mean(),
                    "clusters": len(set(labels) - {-1})})
    res = pd.DataFrame(res)
    return float(res.loc[res["mean_f1"].idxmax(), "distance_threshold"]), res


def run_experiment(dev_seed: int = 1, eval_seed: int = 7):
    from pdisc import synthetic
    dev, ev = synthetic.generate(seed=dev_seed), synthetic.generate(seed=eval_seed)
    threshold, sweep = tune_threshold(dev)

    scrubbed = ev["description"].map(scrub_pii)
    pii_left = scrubbed.str.contains(r"@|\+44|\bext\b", regex=True).sum()
    labels, top_terms = cluster_descriptions(scrubbed, threshold)
    ev = ev.assign(description_scrubbed=scrubbed, text_cluster=labels)

    text_scores = score_groups(ev["true_problem"], ev["text_cluster"]).assign(method="text clustering")
    base_scores = score_groups(ev["true_problem"], ev["category"]).assign(method="category baseline")
    scores = pd.concat([base_scores, text_scores], ignore_index=True)
    scores["problem_name"] = scores["problem"].map({k: v["name"] for k, v in synthetic.PLANTED.items()})

    clusters = (ev[ev["text_cluster"] >= 0].groupby("text_cluster")
                .agg(tickets=("number", "size"), categories=("category", "nunique"), sites=("site", "nunique"),
                     planted_share=("true_problem", lambda s: (s != "none").mean()),
                     dominant_truth=("true_problem", lambda s: s.value_counts().index[0]),
                     example=("description_scrubbed", "first"))
                .reset_index())
    clusters["top_terms"] = clusters["text_cluster"].map(top_terms)
    clusters["kind"] = np.where(clusters["dominant_truth"] == "none", "routine demand (needs signal filter)",
                                "planted problem")
    summary = {"dev_seed": dev_seed, "eval_seed": eval_seed, "chosen_distance_threshold": threshold,
               "tickets": len(ev), "clusters": len(clusters),
               "clusters_matching_planted": int((clusters["kind"] == "planted problem").sum()),
               "tickets_with_pii_left_after_scrub": int(pii_left),
               "baseline_mean_f1": round(base_scores["f1"].mean(), 3),
               "text_mean_f1": round(text_scores["f1"].mean(), 3)}
    return summary, scores, clusters, sweep, ev


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "outputs" / "text_prototype"
    out.mkdir(parents=True, exist_ok=True)
    summary, scores, clusters, sweep, ev = run_experiment()
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    scores.to_csv(out / "scores_vs_baseline.csv", index=False)
    clusters.to_csv(out / "clusters.csv", index=False)
    sweep.to_csv(out / "threshold_sweep_dev.csv", index=False)
    ev.drop(columns=["description"]).to_csv(out / "tickets_scrubbed.csv", index=False)
    print(summary)
    print(scores[["method", "problem", "problem_name", "precision", "recall", "f1", "group_size"]].round(3).to_string())
