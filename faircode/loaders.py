"""Read a tabular dataset file into a DataFrame, regardless of format.

Supports .csv, .tsv, and .xlsx (the last requires the optional `openpyxl`
extra: `pip install faircode[excel]`). Files with an unrecognized or missing
extension fall back to sniffing the delimiter from their content, so a
tab-separated export saved with a `.csv` extension still reads correctly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SNIFF_DELIMITERS = ",\t;|"
SNIFF_SAMPLE_BYTES = 8192


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()

    if suffix == ".xlsx":
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise RuntimeError(
                "reading .xlsx files requires the 'openpyxl' package "
                "(install with: pip install faircode[excel])"
            ) from exc

    if suffix == ".tsv":
        return _read_delimited(path, default="\t")

    if suffix == ".csv":
        return _read_delimited(path, default=",")

    return _read_delimited(path, default=",")


def _read_delimited(path: str, *, default: str) -> pd.DataFrame:
    """Read delimited text, using the extension's convention only as fallback."""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        sample = fh.read(SNIFF_SAMPLE_BYTES)
    return pd.read_csv(path, sep=_sniff_delimiter(sample, default=default))


def _logical_row_delimiter_counts(text: str, delimiter: str, max_rows: int = 5) -> list[int]:
    """Per-row delimiter counts for the first `max_rows` quote-aware logical rows.

    Mirrors assets/profiler-engine.js's logicalRowDelimiterCounts() exactly, so
    both engines sniff the same file the same way. csv.Sniffer() used to do this
    job but scans the *whole* sample for consistency: one messy row anywhere in
    an 8KB sample (e.g. a free-text field with a stray, unquoted delimiter) broke
    it and silently fell back to the extension-implied default, while the JS
    engine - checking only the first 5 logical rows - still sniffed correctly.
    See issue #729.
    """
    counts: list[int] = []
    count = 0
    in_quotes = False
    at_field_start = True
    has_content = False
    i = 0
    n = len(text)
    while i < n and len(counts) < max_rows:
        c = text[i]
        if in_quotes:
            if c == '"':
                if i + 1 < n and text[i + 1] == '"':
                    i += 1
                else:
                    in_quotes = False
        elif c == '"' and at_field_start:
            in_quotes = True
            has_content = True
        elif c == delimiter:
            count += 1
            at_field_start = True
            has_content = True
        elif c in ("\n", "\r"):
            if c == "\r" and i + 1 < n and text[i + 1] == "\n":
                i += 1
            if has_content:
                counts.append(count)
            count = 0
            at_field_start = True
            has_content = False
        else:
            at_field_start = False
            has_content = True
        i += 1
    if len(counts) < max_rows and has_content and not in_quotes:
        counts.append(count)
    return counts


def _sniff_delimiter(sample: str, default: str = ",") -> str:
    sample = sample[:SNIFF_SAMPLE_BYTES]
    if not sample:
        return default
    best, best_count = default, -1
    for d in SNIFF_DELIMITERS:
        counts = _logical_row_delimiter_counts(sample, d)
        if not counts:
            continue
        first = counts[0]
        if first <= 0:
            continue
        if all(c == first for c in counts) and first > best_count:
            best_count = first
            best = d
    return best
