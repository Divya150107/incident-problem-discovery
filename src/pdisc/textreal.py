"""Real-text analysis on the Zenodo "Classification of IT Support Tickets" set.

2,229 real helpdesk tickets (EN/DE/PT/ES), each with a human-assigned category
(Fileservice, Support general, Software, O365, Active Directory, Computer-Services, EOL),
personal data already replaced by tags such as [NAME]. Published train/test split.

What it does (writes outputs/text_zenodo/):
  1. description analysis  - boilerplate stripping, templated share, length, language mix,
                             corrupted characters
  2. text similarity       - character n-gram TF-IDF + cosine; near-duplicate groups and
                             whether near-identical tickets got the same human category
  3. clustering            - rule baseline (first words) vs word TF-IDF vs char TF-IDF,
                             scored against the human categories (ARI, NMI, purity)
  4. category comparison   - per-category profile and coherence, cluster x category crosstab,
                             sub-themes hidden inside the catch-all "Support general",
                             supervised baseline vs the published model, label disagreements
Settings are chosen on the training split and scores reported on the test split.

Only the CSV data files are used. The dataset's model files (.h5 / .pickle) are never
downloaded or loaded: pickles can execute code, and their model isn't needed.

Caveats: categories are topics, not Problems; there are no timestamps, so no trend or
burst analysis is possible on this set.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             normalized_mutual_info_score)
from sklearn.pipeline import make_pipeline, make_union

from pdisc.textcluster import cluster_descriptions, scrub_pii

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "zenodo"
OUT = ROOT / "outputs" / "text_zenodo"

PUBLISHED = {"accuracy": 0.79, "macro_f1": 0.72}     # from the dataset's metrics.txt (test split)

BOILERPLATE = [
    r"\[TICKET ID\]\s*-\s*A Support Ticket was forwarded to (?:your role|you)\s*-",
    r"\[TICKET ID\]\s*-\s*New Support Ticket received\s*-",
    r"-Automated ticket creation-",
    r"Service Request:",
]
TEMPLATES = [
    "file share access", "shared mailbox access", "remove file share access", "basis user",
    "light user for external", "end of life in nexthink", "leaving employee", "intranet editor rights",
]
TAG_RE = r"\[[A-Z_ ]+\]"
STOPWORDS = {
    "en": {"the", "and", "is", "not", "my", "please", "with", "cannot", "to", "for", "on", "working", "can"},
    "de": {"und", "der", "die", "das", "nicht", "ich", "mit", "bitte", "für", "ist", "den", "von", "zu"},
    "pt": {"não", "nao", "com", "está", "esta", "impressora", "acesso", "você", "do", "da", "no", "na", "para"},
    "es": {"el", "los", "las", "con", "por", "correo", "usuario", "creación", "del", "una", "es", "y"},
}


# ------------------------------------------------------------------ load & clean
def load(raw_dir: Path = RAW) -> pd.DataFrame:
    parts = []
    for split in ["train", "test"]:
        x = pd.read_csv(raw_dir / f"X_{split}.csv")
        y = pd.read_csv(raw_dir / f"y_{split}.csv")
        parts.append(x.merge(y, on="id").assign(split=split))
    df = pd.concat(parts, ignore_index=True).rename(columns={"category_truth": "category"})
    df["text"] = df["text"].fillna("").astype(str)
    return df


def clean(text: str) -> str:
    t = text.replace("\xa0", " ")
    for p in BOILERPLATE:
        t = re.sub(p, " ", t, flags=re.I)
    t = re.sub(TAG_RE, " ", t)
    t = scrub_pii(t)
    t = re.sub(r"<\w+>", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def is_templated(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in TEMPLATES)


def language(text: str) -> str:
    words = set(re.findall(r"[a-zà-ÿ]+", text.lower()))
    hits = {lang: len(words & sw) for lang, sw in STOPWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "unknown/short"


def describe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean"] = df["text"].map(clean)
    df["templated"] = df["text"].map(is_templated)
    df["language"] = df["text"].map(language)
    df["length"] = df["clean"].str.len()
    df["corrupted_chars"] = df["text"].str.contains("�")
    df["exact_duplicate"] = df["clean"].duplicated(keep=False)
    return df


def char_vectorizer(min_df: int = 2) -> TfidfVectorizer:
    # character n-grams inside word boundaries: works across EN/DE/PT/ES and tolerates typos
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=min_df)


def word_vectorizer(min_df: int = 2) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=min_df,
                           token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")


# ------------------------------------------------------------------ similarity
def near_duplicate_groups(texts: pd.Series, threshold: float = 0.9, vectorizer=None) -> np.ndarray:
    """Connected components of the 'cosine similarity >= threshold' graph."""
    vec = vectorizer or char_vectorizer(min_df=1)
    X = vec.fit_transform(texts)
    S = (X @ X.T).tocsr()
    S.data = (S.data >= threshold).astype(np.int8)
    S.eliminate_zeros()
    _, labels = connected_components(S, directed=False)
    return labels


# ------------------------------------------------------------------ scoring
def purity(truth: pd.Series, clusters: np.ndarray) -> float:
    """Share of clustered items whose cluster's majority category matches theirs (-1 excluded)."""
    m = clusters >= 0
    if not m.any():
        return float("nan")
    t = pd.DataFrame({"c": clusters[m], "y": np.asarray(truth)[m]})
    return float(t.groupby("c")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / m.sum())


def as_partition(clusters: np.ndarray) -> np.ndarray:
    """Unclustered items (-1) become their own singleton groups, so ARI/NMI aren't flattered."""
    out = np.asarray(clusters).copy()
    noise = np.where(out < 0)[0]
    out[noise] = out.max() + 1 + np.arange(len(noise))
    return out


def score_partition(truth: pd.Series, clusters: np.ndarray) -> dict:
    p = as_partition(clusters)
    return {"ari": adjusted_rand_score(truth, p), "nmi": normalized_mutual_info_score(truth, p),
            "purity": purity(truth, clusters), "clusters": int(len(set(clusters) - {-1})),
            "coverage": float((np.asarray(clusters) >= 0).mean())}


def rule_baseline(clean_texts: pd.Series, n_words: int = 3, min_size: int = 5) -> np.ndarray:
    """Non-AI baseline: group tickets that start with the same first words."""
    key = clean_texts.str.split().str[:n_words].str.join(" ")
    codes, _ = pd.factorize(key)
    sizes = pd.Series(codes).value_counts()
    return np.where(pd.Series(codes).map(sizes) >= min_size, codes, -1)


# ------------------------------------------------------------------ experiment
def run(raw_dir: Path = RAW, out_dir: Path = OUT) -> dict:
    df = describe(load(raw_dir))
    train, test = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)

    # 1. description analysis
    profile = df.groupby("category").agg(
        tickets=("id", "size"), median_length=("length", "median"), templated_share=("templated", "mean"),
        exact_duplicate_share=("exact_duplicate", "mean"), corrupted_share=("corrupted_chars", "mean")).reset_index()
    langs = pd.crosstab(df["category"], df["language"], normalize="index").add_prefix("lang_").reset_index()
    vec_all = char_vectorizer().fit(df["clean"])
    Xall = vec_all.transform(df["clean"])
    coh = {}
    for c, idx in df.groupby("category").indices.items():
        v = np.asarray(Xall[idx].sum(axis=0)).ravel()
        n = len(idx)
        coh[c] = (v @ v - n) / (n * (n - 1)) if n > 1 else np.nan   # mean pairwise cosine (rows are unit-norm)
    profile["within_category_similarity"] = profile["category"].map(coh)
    profile = profile.merge(langs, on="category")

    # 2. near-duplicates (unsupervised, on all tickets; labels only used to check consistency)
    df["near_dup_group"] = near_duplicate_groups(df["clean"], 0.9)
    g = df.groupby("near_dup_group")
    groups = g.agg(size=("id", "size"), templated_share=("templated", "mean"),
                   categories=("category", "nunique"),
                   majority_category=("category", lambda s: s.value_counts().index[0]),
                   majority_share=("category", lambda s: s.value_counts(normalize=True).iloc[0]),
                   example=("text", "first")).reset_index()
    groups = groups[groups["size"] >= 3].sort_values("size", ascending=False)
    groups["kind"] = np.where(groups["templated_share"] >= 0.5, "templated request", "repeated issue / other")

    # 3. clustering: choose the distance threshold on train (by ARI vs categories), score on test
    sweep, best = [], {}
    for name, make in [("word_tfidf", word_vectorizer), ("char_tfidf", char_vectorizer)]:
        for t in np.arange(0.5, 0.96, 0.05):
            labels, _ = cluster_descriptions(train["clean"], float(t), min_size=5, vectorizer=make())
            s = score_partition(train["category"], labels)
            sweep.append({"method": name, "distance_threshold": round(float(t), 2), **s})
        sw = pd.DataFrame([r for r in sweep if r["method"] == name])
        best[name] = float(sw.loc[sw["ari"].idxmax(), "distance_threshold"])
    rows, test_labels = [], {}
    rb = rule_baseline(test["clean"])
    rows.append({"method": "rule baseline (first 3 words)", "distance_threshold": None,
                 **score_partition(test["category"], rb)})
    for name, make in [("word_tfidf", word_vectorizer), ("char_tfidf", char_vectorizer)]:
        vec = make().fit(train["clean"])          # vocabulary from train only
        labels, top = cluster_descriptions(test["clean"], best[name], min_size=5, vectorizer=vec, fit=False)
        test_labels[name] = (labels, top)
        rows.append({"method": name, "distance_threshold": best[name], **score_partition(test["category"], labels)})
    cluster_scores = pd.DataFrame(rows)

    labels, top = test_labels["char_tfidf"]
    t2 = test.assign(cluster=labels)
    crosstab = pd.crosstab(t2.loc[t2["cluster"] >= 0, "cluster"], t2.loc[t2["cluster"] >= 0, "category"])
    crosstab.insert(0, "top_terms", crosstab.index.map(top))
    crosstab = crosstab.reset_index()

    # 4a. sub-themes inside the catch-all category (word features for readable terms)
    sg = df[df["category"] == "Support general"].reset_index(drop=True)
    if len(sg) >= 10:
        sg_labels, sg_top = cluster_descriptions(sg["clean"], best["word_tfidf"], min_size=5, vectorizer=word_vectorizer())
    else:   # category absent or too small to sub-cluster
        sg_labels, sg_top = np.full(len(sg), -1), {}
    sg = sg.assign(theme=sg_labels)
    themes = sg[sg["theme"] >= 0].groupby("theme").agg(
        tickets=("id", "size"), templated_share=("templated", "mean"),
        example=("text", "first")).reset_index()
    themes["top_terms"] = themes["theme"].map(sg_top)
    themes = themes.sort_values("tickets", ascending=False)
    themes["share_of_support_general"] = themes["tickets"] / len(sg)

    # 4b. supervised baseline vs the published model; disagreements = possible mislabels
    clf = make_pipeline(make_union(word_vectorizer(1), char_vectorizer(2)),
                        LogisticRegression(max_iter=3000, class_weight="balanced", C=5.0))
    clf.fit(train["clean"], train["category"])
    pred = clf.predict(test["clean"])
    conf = clf.predict_proba(test["clean"]).max(axis=1)
    sup = pd.DataFrame([
        {"model": "TF-IDF (word+char) + logistic regression (this project)",
         "accuracy": accuracy_score(test["category"], pred),
         "macro_f1": f1_score(test["category"], pred, average="macro")},
        {"model": "published model (dataset authors, metrics.txt)", **PUBLISHED},
    ])
    disagree = test.assign(predicted=pred, confidence=conf)
    disagree = disagree[disagree["predicted"] != disagree["category"]].sort_values("confidence", ascending=False)
    disagree = disagree[["id", "category", "predicted", "confidence", "templated", "text"]]

    summary = {
        "tickets": len(df), "train": len(train), "test": len(test),
        "templated_share": round(df["templated"].mean(), 3),
        "exact_duplicate_share": round(df["exact_duplicate"].mean(), 3),
        "corrupted_char_tickets": int(df["corrupted_chars"].sum()),
        "near_duplicate_groups_3plus": len(groups),
        "near_duplicate_groups_with_mixed_categories": int((groups["categories"] > 1).sum()),
        "best_test_ari": round(cluster_scores["ari"].max(), 3),
        "support_general_themes": len(themes),
        "supervised_accuracy": round(sup.loc[0, "accuracy"], 3),
        "supervised_macro_f1": round(sup.loc[0, "macro_f1"], 3),
        "caveats": "categories are topics, not Problems; no timestamps (no trend/burst analysis); "
                   "short, often templated text; thresholds chosen on train, scores on test",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    profile.to_csv(out_dir / "category_profile.csv", index=False)
    groups.to_csv(out_dir / "near_duplicate_groups.csv", index=False)
    pd.DataFrame(sweep).to_csv(out_dir / "threshold_sweep_train.csv", index=False)
    cluster_scores.to_csv(out_dir / "cluster_vs_category.csv", index=False)
    crosstab.to_csv(out_dir / "cluster_category_crosstab.csv", index=False)
    themes.to_csv(out_dir / "support_general_subthemes.csv", index=False)
    sup.to_csv(out_dir / "supervised_metrics.csv", index=False)
    disagree.to_csv(out_dir / "label_disagreements.csv", index=False)
    return {"summary": summary, "cluster_scores": cluster_scores, "supervised": sup,
            "profile": profile, "groups": groups, "themes": themes, "disagree": disagree}


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(r["summary"])
    print(r["cluster_scores"].round(3).to_string())
    print(r["supervised"].round(3).to_string())
    print(r["profile"].round(3).to_string())
    print(r["groups"].head(8)[["size", "kind", "categories", "majority_category", "majority_share", "example"]].to_string())
    print(r["themes"].head(10)[["tickets", "templated_share", "top_terms", "example"]].to_string())
    print(r["disagree"].head(8).to_string())
