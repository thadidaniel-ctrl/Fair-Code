"""Tests for the opt-in chi-squared proxy hints (needs the scipy extra).

Run from the repo root:  pytest tests/ -q
"""

from pathlib import Path

import pandas as pd
import pytest

from faircode import profile
from faircode.proxy import proxy_hints

pytest.importorskip("scipy", reason="proxy hints need the optional scipy extra")


def test_benefits_denial_native_proxy_summary_is_not_stale():
    # Regression test for #733: the top-of-file proxy summary must keep
    # Native applicants separate from Black applicants, with Native's 20.3%
    # rate preserved instead of the stale ~15% lumping that appears elsewhere.
    source = (Path(__file__).resolve().parents[1] / "Benefits Denial" / "unfair.py").read_text(encoding="utf-8")
    assert "Native applicants are at\n#                   20.3% (~77% of White's rate)." in source
    assert "Native applicants are at\n#                   ~15.5%" not in source


def test_perfect_proxy_is_flagged():
    # occupation is a perfect function of sex -> maximal association.
    df = pd.DataFrame({
        "sex": ["male", "female"] * 100,
        "occupation": ["engineer", "nurse"] * 100,
    })
    dims = profile(df)["dimensions"]
    hints = proxy_hints(df, dims)
    pair = next(h for h in hints
                if {h["a"], h["b"]} == {"sex", "occupation"})
    assert pair["p_value"] < 0.05
    assert pair["cramers_v"] > 0.9   # near-perfect association (Yates-corrected)


def test_held_out_column_catches_proxy_for_a_dropped_attribute():
    # "we dropped race so it's fine": race isn't in the profiled dataframe at
    # all, so it never becomes a dimension - proxy_hints() would otherwise
    # have no way to flag zip_code as a proxy for it (#328).
    race = (["A"] * 100 + ["B"] * 100)
    zip_code = (["111"] * 100 + ["222"] * 100)  # perfectly aligned with race
    df = pd.DataFrame({"zip_code": zip_code, "sex": ["M", "F"] * 100})

    held_out = {"race": pd.Series(race, index=df.index)}
    hints = proxy_hints(df, profile(df)["dimensions"], held_out=held_out)

    pair = next(h for h in hints if {h["a"], h["b"]} == {"zip_code", "race"})
    assert pair["p_value"] < 0.05
    assert pair["cramers_v"] > 0.9


def test_held_out_column_not_flagged_against_an_independent_column():
    # sex cycles every 2 rows, race every 3 - deliberately different periods
    # so the two are independent (verified: no hint), unlike the perfectly-
    # aligned zip_code/race case above.
    df = pd.DataFrame({"sex": ["male", "female"] * 150})
    held_out = {"race": pd.Series((["A", "B", "C"] * 100), index=df.index)}

    hints = proxy_hints(df, profile(df)["dimensions"], held_out=held_out)

    assert not any("race" in (h["a"], h["b"]) for h in hints)


def test_independent_columns_not_flagged():
    # Deterministic independence: sex alternates every row, grp every 3 rows,
    # so the two are (near) independent and should not be flagged.
    df = pd.DataFrame({
        "sex": ["male", "female"] * 150,
        "grp": ["x", "y", "z"] * 100,
    })
    hints = proxy_hints(df, profile(df)["dimensions"])
    assert not any({h["a"], h["b"]} == {"sex", "grp"} for h in hints)
