"""Zenodo real-text module, tested on fabricated tickets (no network, no real data)."""
from pathlib import Path

import numpy as np
import pandas as pd

from pdisc import textreal


def test_boilerplate_and_tags_are_stripped():
    raw = "[TICKET ID] - A Support Ticket was forwarded to your role  - Printer [NAME] offline ([COMPANY]. [LOCATION])"
    c = textreal.clean(raw)
    assert c == "printer offline (. )" or ("printer" in c and "offline" in c)
    assert "ticket id" not in c and "forwarded" not in c and "[" not in c
    assert textreal.clean("[TICKET ID] - New Support Ticket received  - new phone") == "new phone"


def test_templates_detected():
    assert textreal.is_templated("File Share Access - [TICKET ID] - [NAME]")
    assert not textreal.is_templated("printer jams on floor 2")


def test_near_duplicates_group_across_languages_with_char_ngrams():
    texts = pd.Series([
        "outlook kalender synchronisation fehler",      # German
        "outlook kalender synchronisation fehlerhaft",  # near-identical German
        "outlook calendar synchronisation error",       # English near-translation
        "printer out of toner",
        "vpn drops every five minutes",
    ])
    labels = textreal.near_duplicate_groups(texts, threshold=0.55)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0] and labels[4] != labels[0]


def test_ari_and_purity_on_perfect_and_shuffled_labels():
    truth = pd.Series(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    perfect = np.repeat([0, 1, 2], 20)
    s = textreal.score_partition(truth, perfect)
    assert s["ari"] == 1.0 and s["purity"] == 1.0
    shuffled = np.random.default_rng(0).permutation(perfect)
    assert textreal.score_partition(truth, shuffled)["ari"] < 0.1


def test_unclustered_items_do_not_flatter_ari():
    truth = pd.Series(["a"] * 10 + ["b"] * 10)
    mostly_noise = np.array([0, 0] + [-1] * 18)
    assert textreal.score_partition(truth, mostly_noise)["ari"] < 0.2


def test_rule_baseline_groups_same_opening_words():
    texts = pd.Series(["file share access a"] * 6 + ["vpn is down"] * 2)
    labels = textreal.rule_baseline(texts, n_words=3, min_size=5)
    assert len(set(labels[:6])) == 1 and labels[0] >= 0 and (labels[6:] == -1).all()


def test_module_never_touches_model_files():
    # the dataset ships a pickled tokenizer/encoder and a Keras .h5 model; loading a pickle can run code
    src = Path(textreal.__file__).read_text()
    for forbidden in ["import pickle", "pickle.load", "joblib", "load_model", "h5py", "keras", "tensorflow"]:
        assert forbidden not in src, forbidden


def test_end_to_end_on_fabricated_split(tmp_path):
    rng = np.random.default_rng(1)
    cats = {"Fileservice": "file share access - [TICKET ID] - [NAME]",
            "O365": "outlook mailbox full cannot send email",
            "Computer-Services": "impressora nao imprime papel preso"}
    for split, n in [("train", 60), ("test", 30)]:
        rows = []
        for i in range(n):
            c = list(cats)[i % 3]
            rows.append({"id": f"{split}{i}", "text": cats[c] + rng.choice(["", " urgent", " please"]), "category_truth": c})
        df = pd.DataFrame(rows)
        df[["id", "text"]].to_csv(tmp_path / f"X_{split}.csv", index=False)
        df[["id", "category_truth"]].to_csv(tmp_path / f"y_{split}.csv", index=False)
    r = textreal.run(raw_dir=tmp_path, out_dir=tmp_path / "out")
    assert r["summary"]["tickets"] == 90
    assert r["supervised"].loc[0, "accuracy"] == 1.0
    for f in ["summary", "cluster_vs_category", "category_profile", "near_duplicate_groups",
              "support_general_subthemes", "supervised_metrics", "label_disagreements"]:
        assert (tmp_path / "out" / f"{f}.csv").exists()
