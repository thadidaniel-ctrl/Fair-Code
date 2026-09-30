"""Embedded sample dataset for `faircode profile --sample`.

Mirrors assets/profiler-ui.js:buildSampleCSV() exactly - same column names,
same row values, same deliberately imbalanced distributions. The web and
Python versions are kept in sync by a test (tests/test_cli.py) that compares
the generated CSV strings.
"""

from __future__ import annotations

import io

import pandas as pd

# Column order must match the web sample exactly.
COLUMNS = ["patient_id", "age", "sex", "race", "region", "diabetic"]

# Same seed data as the JS buildSampleCSV().
_RACES = [
    "Caucasian", "Caucasian", "Caucasian", "Caucasian", "Caucasian",
    "Caucasian", "AfricanAmerican", "Hispanic",
]
_REGIONS = ["Northeast", "Northeast", "Northeast", "Northeast", "Midwest", "Midwest"]
_AGES = [27, 29, 31, 33, 34, 36, 38, 41]


def build_sample_csv() -> str:
    """Return the sample dataset as a CSV string, matching the web profiler."""
    rows = [",".join(COLUMNS)]
    for i in range(160):
        age = _AGES[i % len(_AGES)]
        if i % 53 == 0:
            age = 72
        sex = "male" if i % 10 < 7 else "female"
        race = _RACES[i % len(_RACES)]
        if i % 80 == 0:
            race = "Asian"
        region = _REGIONS[i % len(_REGIONS)]
        if i % 80 == 0:
            region = "West"
        diabetic = "Yes" if i % 3 == 0 else "No"
        rows.append(f"{1000 + i},{age},{sex},{race},{region},{diabetic}")
    return "\n".join(rows)


def sample_df() -> pd.DataFrame:
    """Return the sample dataset as a pandas DataFrame."""
    return pd.read_csv(io.StringIO(build_sample_csv()))