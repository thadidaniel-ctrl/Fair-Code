"""Deterministic sample dataset shared by the CLI's `--sample` flag and the
web profiler's "Try it with a sample dataset" button (assets/profiler-ui.js's
buildSampleCSV()). Both MUST produce byte-identical CSV text, so a
new/first-time user sees the same demo regardless of which engine they use.

Deliberately imbalanced (skewed sex, rare elderly band, an under-sampled
race and region) so a profile run against it visibly flags representation
gaps rather than reporting a clean, uninteresting dataset.
"""

from __future__ import annotations

SAMPLE_FILENAME = "sample-health-data.csv"


def build_sample_csv() -> str:
    """Return the sample dataset as CSV text (no trailing newline)."""
    rows = [["patient_id", "age", "sex", "race", "region", "diabetic"]]
    # Heavily Caucasian; minorities rare. Skewed male. Concentrated young/mid age.
    races = ["Caucasian", "Caucasian", "Caucasian", "Caucasian", "Caucasian",
             "Caucasian", "AfricanAmerican", "Hispanic"]
    regions = ["Northeast", "Northeast", "Northeast", "Northeast", "Midwest", "Midwest"]
    ages = [27, 29, 31, 33, 34, 36, 38, 41]  # tightly clustered; few elderly
    for i in range(160):
        age = ages[i % len(ages)]
        if i % 53 == 0:
            age = 72  # a rare elderly row
        sex = "male" if i % 10 < 7 else "female"  # ~70/30 skew
        race = races[i % len(races)]
        if i % 80 == 0:
            race = "Asian"  # a barely-present group
        region = regions[i % len(regions)]
        if i % 80 == 0:
            region = "West"  # a barely-present region
        diabetic = "Yes" if i % 3 == 0 else "No"
        rows.append([str(1000 + i), str(age), sex, race, region, diabetic])
    return "\n".join(",".join(r) for r in rows)
