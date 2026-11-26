import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def cfg():
    from pdisc.pipeline import load_config
    return load_config("uci")


@pytest.fixture
def mendeley_cfg():
    from pdisc.pipeline import load_config
    return load_config("mendeley")


def make_incidents(n: int, seed: int = 0, start="2016-03-07", weeks=12, **fixed) -> pd.DataFrame:
    """Incident-level rows shaped like ingest.build_incidents output.

    Background: 30 signatures x 40 locations, ~37% SLA miss, low reopen/bounce,
    evenly spread over `weeks`. Keyword args pin a column to a value.
    """
    rng = np.random.default_rng(seed)
    opened = pd.Timestamp(start) + pd.to_timedelta(rng.uniform(0, weeks * 7 * 24, n), unit="h")
    cat = rng.integers(1, 6, n)
    sub = cat * 10 + rng.integers(0, 6, n)
    df = pd.DataFrame({
        "number": [f"INC{seed:02d}{i:06d}" for i in range(n)],
        "opened_at": opened,
        "category": [f"Category {c}" for c in cat],
        "subcategory": [f"Subcategory {s}" for s in sub],
        "u_symptom": "Symptom 1",
        "location": [f"Location {l}" for l in rng.integers(1, 41, n)],
        "assignment_group": [f"Group {g}" for g in rng.integers(1, 11, n)],
        "caller_key": [str(c) for c in rng.integers(1, 5000, n)],
        "is_high_priority": rng.random(n) < 0.03,
        "is_high_impact": rng.random(n) < 0.03,
        "is_reopened": rng.random(n) < 0.01,
        "is_sla_missed": rng.random(n) < 0.37,
        "is_bounced": rng.random(n) < 0.11,
        "reassignment_count": rng.integers(0, 3, n),
        "resolution_hours": rng.gamma(1.5, 20, n),
        "problem_id": None,
        "rfc": None,
    })
    for k, v in fixed.items():
        df[k] = v
    return finish(df)


def finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signature"] = df["category"] + " | " + df["subcategory"] + " | " + df["u_symptom"]
    df["opened_week"] = df["opened_at"].dt.to_period("W-SUN").dt.start_time
    df["opened_date"] = df["opened_at"].dt.normalize()
    df["has_known_problem"] = df["problem_id"].notna()
    df["has_change"] = df["rfc"].notna()
    df["resolved_at"] = df["opened_at"] + pd.to_timedelta(df["resolution_hours"], unit="h")
    return df


@pytest.fixture
def background():
    return make_incidents(6000)
