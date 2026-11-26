import pandas as pd

from pdisc import synthetic, textcluster


def test_pii_is_scrubbed():
    s = textcluster.scrub_pii("vpn drops - Asha (asha.k@example.com) call me on +44 7700 912345 regards Ravi, ext 4411")
    assert "@" not in s and "7700" not in s and "Asha" not in s and "Ravi" not in s and "4411" not in s


def test_synthetic_set_has_planted_ground_truth():
    df = synthetic.generate(n_routine=500, seed=3)
    counts = df["true_problem"].value_counts()
    assert {"P1", "P2", "P3", "P4", "P5"} <= set(counts.index)
    assert counts["P1"] == synthetic.PLANTED["P1"]["n"]
    # P1 deliberately spans categories (what defeats category grouping)
    assert df.loc[df["true_problem"] == "P1", "category"].nunique() > 1


def test_score_groups_perfect_and_split():
    truth = pd.Series(["P1"] * 4 + ["none"] * 4)
    perfect = textcluster.score_groups(truth, pd.Series([1, 1, 1, 1, 2, 2, 2, 2]))
    assert perfect.loc[0, "f1"] == 1.0
    split = textcluster.score_groups(truth, pd.Series([1, 1, 3, 3, 2, 2, 2, 2]))
    assert split.loc[0, "recall"] == 0.5 and split.loc[0, "precision"] == 1.0


def test_clustering_beats_category_baseline_on_held_out_seed():
    summary, scores, clusters, _, _ = textcluster.run_experiment(dev_seed=1, eval_seed=7)
    assert summary["text_mean_f1"] > summary["baseline_mean_f1"] + 0.3
    assert summary["tickets_with_pii_left_after_scrub"] == 0
    # PII placeholders / sign-offs must not drive the clusters
    assert not clusters["top_terms"].str.contains(r"\b(?:phone|name|regards)\b").any()
