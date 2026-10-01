"""Parity tests between the Python and JavaScript profiler implementations."""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from faircode import compare, profile
from faircode.loaders import read_table

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

requires_openpyxl = pytest.mark.skipif(
    importlib.util.find_spec("openpyxl") is None,
    reason="optional 'excel' extra not installed",
)


def _extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    assert match, f"Could not find {pattern!r}"
    return match.group(1)


def test_sheetjs_cdn_url_matches():
    engine = (REPO_ROOT / "assets" / "profiler-engine.js").read_text(encoding="utf-8")
    cli = (REPO_ROOT / "scripts" / "engine-js.js").read_text(encoding="utf-8")

    engine_url = _extract(r'script\.src\s*=\s*"([^"]+)"', engine)
    cli_url = _extract(r'XLSX_CDN_URL\s*=\s*"([^"]+)"', cli)

    assert engine_url == cli_url


def test_compare_card_renderers_special_case_kind_mismatch():
    """driftCard() and buildCompareHtmlReport()'s per-dimension section must
    both read cd.kind_mismatch, so a skipped comparison isn't drawn as a
    "none drift" badge next to a real score change (#519). Source-level check
    (mirrors test_sheetjs_cdn_url_matches) - these renderers are DOM-coupled
    and have no unit harness."""
    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")

    drift_card = src[src.index("function driftCard("):]
    drift_card = drift_card[: drift_card.index("\n  }\n")]
    assert "kind_mismatch" in drift_card
    assert "comparison skipped" in drift_card

    report = src[src.index("function buildCompareHtmlReport("):]
    assert "if (cd.kind_mismatch)" in report
    # the skipped badge is styled in both the live css and the report's own <style>
    assert ".drift-badge.skipped" in src
    assert ".drift-badge.skipped" in (REPO_ROOT / "assets" / "profiler.css").read_text(encoding="utf-8")


def test_buildHtmlReport_output_structure():
    """buildHtmlReport() must produce structured output matching the Python to_html()
    contract. Source-level check (mirrors test_compare_card_renderers_special_case_kind_mismatch)
    - buildHtmlReport() is DOM-coupled and has no unit harness."""

    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")

    # Isolate buildHtmlReport function body
    func_start = src.index("function buildHtmlReport(")
    func_body = src[func_start:]
    brace_count = 1
    i = 0
    while brace_count > 0 and i < len(func_body):
        if func_body[i] == "{":
            brace_count += 1
        elif func_body[i] == "}":
            brace_count -= 1
        i += 1
    func_body = func_body[:i]

    # Check for essential structural elements from Python to_html()

    # 1. Dimension/group rendering structure
    assert "r.dimensions.map" in func_body
    assert "d.groups.slice(0, DISPLAY_GROUPS)" in func_body
    assert "class=\"dim\"" in func_body
    assert "class=\"score\"" in func_body

    # 2. Flags section
    assert "r.flags.length" in func_body
    assert "class=\"flags\"" in func_body
    assert "Flags" in func_body

    # 3. Reference information
    assert "if (d.reference)" in func_body
    assert "class=\"reference\"" in func_body
    assert "Reference" in func_body

    # 4. Report-level structure
    assert "Dataset Representation Profile" in func_body
    assert "overall_score" in func_body
    assert "grade" in func_body
    assert "!DOCTYPE html" in func_body

    # 5. Feature flags from the Python implementation
    assert "under-represented" in func_body
    assert "small-group" in func_body


# ── Proxy hints in the web profiler (issue #758) ────────────────────────────
# The browser "Check for proxy columns" feature used to draw its results and
# drop them, so neither export carried them. These cover the engine port and
# the export path that lost the data.

requires_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="optional 'proxy' extra not installed",
)


def _proxied_dataset() -> str:
    """CSV where zip_code is a perfect proxy for sex and region is independent."""
    regions = ["north", "south", "east", "west"]
    rows = ["sex,zip_code,region,age"]
    for i in range(200):
        sex = "male" if i % 2 else "female"
        zip_code = "10001" if sex == "male" else "10002"
        rows.append(f"{sex},{zip_code},{regions[(i // 2) % 4]},{20 + (i % 55)}")
    return "\n".join(rows) + "\n"


@requires_scipy
def test_js_engine_proxy_hints_match_python(tmp_path):
    """The JS proxyHints() port must flag the same pairs, with the same numbers,
    as faircode/proxy.py - otherwise the web profiler would contradict the
    CLI's --proxy-hints on the same file."""
    from faircode.proxy import proxy_hints

    csv = tmp_path / "proxied.csv"
    csv.write_text(_proxied_dataset(), encoding="utf-8")

    df = pd.read_csv(csv)
    py_hints = proxy_hints(df, profile(df)["dimensions"])
    # Guard against a vacuous pass: zip_code really is a proxy for sex here.
    assert [(h["a"], h["b"]) for h in py_hints] == [("sex", "zip_code")]

    script = (
        "const fs=require('fs');"
        "require(process.argv[1]);"
        "const E=globalThis.FairCodeProfiler;"
        "const table=E.parseCSV(fs.readFileSync(process.argv[2],'utf8'));"
        "process.stdout.write(JSON.stringify("
        "E.proxyHints(table, E.profile(table).dimensions)));"
    )
    completed = subprocess.run(
        ["node", "-e", script,
         str(REPO_ROOT / "assets" / "profiler-engine.js"), str(csv)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    js_hints = json.loads(completed.stdout)

    assert [(h["a"], h["b"]) for h in js_hints] == [(h["a"], h["b"]) for h in py_hints]
    for js_hint, py_hint in zip(js_hints, py_hints):
        assert js_hint["cramers_v"] == pytest.approx(py_hint["cramers_v"], rel=1e-6)
        assert js_hint["chi2"] == pytest.approx(py_hint["chi2"], rel=1e-6)
        # A tiny p-value must survive the port rather than cancel to 0.
        assert js_hint["p_value"] == pytest.approx(py_hint["p_value"], rel=1e-6)


def _run_ui_harness() -> dict:
    completed = subprocess.run(
        ["node", str(REPO_ROOT / "tests" / "js_profiler_ui_harness.js")],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return json.loads(completed.stdout)


def test_proxy_hints_reach_both_exports_after_check():
    """#758 regression: the computed pairs must land on currentResult, which is
    what buildHtmlReport() and copyResultAsJSON() both read. Before the fix the
    pair was drawn on screen and absent from both exports."""
    report = _run_ui_harness()

    # Shown on screen ...
    assert report["check"]["panelHidden"] is False
    assert any("sex ↔ zip_code" in item for item in report["check"]["items"])

    # ... and in the standalone HTML report, in to_html()'s wording.
    assert "Proxy Hints" in report["afterCheckHtml"]
    assert "chi-squared association, informational" in report["afterCheckHtml"]
    assert "sex ↔ zip_code" in report["afterCheckHtml"]

    # ... and in the JSON export.
    hints = report["afterCheckJson"]["proxy_hints"]
    assert [(h["a"], h["b"]) for h in hints] == [("sex", "zip_code")]
    assert hints[0]["p_value"] < 0.05


def test_exports_omit_proxy_hints_until_checked():
    """A profile the user never proxy-checked must not grow a proxy_hints key or
    an empty Proxy Hints heading - the exports keep their pre-#758 shape."""
    report = _run_ui_harness()

    assert "proxy_hints" not in report["beforeCheckJsonParsed"]
    assert "Proxy Hints" not in report["beforeCheckHtml"]


def test_proxy_hints_not_fabricated_or_left_stale():
    """Re-profiling a new dataset, and checking a dataset with no associated
    pair, must both leave the exports clean - no leftover pairs from the earlier
    check, and no invented ones for the empty result."""
    report = _run_ui_harness()

    # A fresh profile replaces currentResult, so the previous check's pairs go
    # with it rather than riding along into the new file's export.
    assert "proxy_hints" not in report["afterReprofileJson"]
    assert "Proxy Hints" not in report["afterReprofileHtml"]

    # Checking finds nothing: recorded as an empty list, drawn as nothing.
    assert report["emptyCheck"]["panelHidden"] is True
    assert report["emptyCheck"]["items"] == []
    assert report["emptyCheckJson"]["proxy_hints"] == []
    assert "Proxy Hints" not in report["emptyCheckHtml"]


def test_buildHtmlReport_proxy_section_source_shape():
    """Source-level guard on the two halves of the fix: the report must read
    r.proxy_hints, and renderProxyHints() must write to currentResult. Mirrors
    test_threshold_input_recovers_panel_after_invalid_value - these functions
    are DOM-coupled, so the behavioural coverage above runs them through the
    Node harness rather than a JS unit runner."""
    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")

    assert "currentResult.proxy_hints = hints;" in src

    report = src[src.index("function buildHtmlReport("):]
    report = report[: report.index("\n  function ")]
    assert "r.proxy_hints" in report
    # Gated on length, not truthiness: an empty list is truthy in JS, so a bare
    # `if (r.proxy_hints)` would draw an empty Proxy Hints heading.
    assert "r.proxy_hints.length" in report


# Real audit datasets are already tracked in their own audit folders - reuse
# them instead of keeping a second multi-megabyte copy under tests/fixtures.
CSV_PATHS = {
    "small.csv": FIXTURES / "small.csv",
    "adult.csv": REPO_ROOT / "Benefits Denial" / "adult.csv",
    "compas-scores-raw.csv": REPO_ROOT / "COMPAS" / "compas-scores-raw.csv",
    "credit_customers.csv": REPO_ROOT / "German Credit Lending" / "credit_customers.csv",
    "AI_Fair_Recruitment_Dataset.csv": REPO_ROOT / "AI Fair Recruitment" / "AI_Fair_Recruitment_Dataset.csv",
}


@pytest.mark.parametrize("csv_name", list(CSV_PATHS))
def test_python_js_profiler_parity(csv_name):
    """The Python and JavaScript profilers should produce equivalent structured JSON."""

    csv = CSV_PATHS[csv_name]

    python_result = profile(pd.read_csv(csv))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    javascript_result = json.loads(completed.stdout)

    # Flags are human-readable messages. They duplicate information already
    # present in the structured output and may differ because Python and
    # JavaScript format floating-point values differently (e.g. 6.25 -> 6.2
    # vs 6.3). Compare the structured data instead.
    python_result = dict(python_result)
    javascript_result = dict(javascript_result)

    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result


def test_python_js_profiler_parity_preserves_mid_field_quotes(tmp_path):
    """Quotes after field content are literal in pandas and the browser parser."""
    csv = tmp_path / "space_before_quote.csv"
    csv.write_text(
        'sex, race, age\n'
        'Male, "White", 25\n'
        'Female, "Black", 30\n'
        'Male, "White", 45\n'
        'Female, "Asian", 22\n',
        encoding="utf-8",
    )

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result


def test_python_js_profiler_parity_sniffs_quoted_newlines(tmp_path):
    """Embedded newlines do not split logical rows during delimiter sniffing."""
    dataset = tmp_path / "sniff_quoted_newline.dat"
    dataset.write_text(
        'sex;race;age;notes\n'
        'M;White;25;"single line"\n'
        'F;Black;30;"multi\nline note"\n'
        'M;White;45;"ok"\n'
        'F;Asian;22;"fine"\n'
        'M;White;50;"good"\n',
        encoding="utf-8",
    )

    python_result = profile(read_table(str(dataset)))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(dataset)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result


def test_python_js_profiler_parity_detects_dates_appended_after_numeric_ages(tmp_path):
    """A merged-in birthdate block cannot become a fabricated elderly group."""
    csv = tmp_path / "mixed-age-and-birthdate.csv"
    ages = [str(age) for age in range(20, 80)]
    dates = ["1985-03-21", "1990-07-14", "2001-11-02"] * 20
    csv.write_text("age\n" + "\n".join(ages + dates) + "\n", encoding="utf-8")

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result
    assert all(dim["name"] != "age" for dim in python_result["dimensions"])


def test_python_js_profiler_parity_rejects_negative_age_sentinels(tmp_path):
    """Signed sentinel ages stay missing in both profiler engines."""
    csv = tmp_path / "negative-age-sentinels.csv"
    csv.write_text(
        "age,sex\n25,F\n30,M\nage -5,F\n-1,M\n45,F\n",
        encoding="utf-8",
    )

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result
    age = next(d for d in python_result["dimensions"] if d["name"] == "age")
    assert "75+" not in {group["label"] for group in age["groups"]}
    assert age["missing_pct"] == 0.4


def test_python_js_na_token_parity_on_literal_na_and_none(tmp_path):
    """NA_TOKENS / isMissing() must match pandas' default STR_NA_VALUES
    exactly and case-sensitively: literal "None" is missing, bare lowercase
    "na" is a real category. The JS engine used to have both backwards and
    lower-cased the cell before comparing (#491)."""
    csv = tmp_path / "na_test.csv"
    csv.write_text(
        "status,x\n"
        "active,1\ninactive,2\nna,3\nna,4\nNone,5\nNone,6\n"
        "active,7\ninactive,8\nactive,9\ninactive,10\n",
        encoding="utf-8",
    )

    python_result = profile(read_table(str(csv)))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)
    assert javascript_result == python_result

    status = next(d for d in python_result["dimensions"] if d["name"] == "status")
    labels = {g["label"] for g in status["groups"]}
    assert "na" in labels          # bare lowercase "na" is NOT a pandas NA token
    assert "None" not in labels    # "None" IS a pandas NA token
    assert status["missing_pct"] == 0.2


def test_python_js_public_params_parity_for_a_defaulted_run():
    """A web-profiler export with no threshold ever touched must still record
    the 7 resolved defaults in provenance.params, matching the CLI/MCP path -
    E.publicParams({}) mirrors provenance.public_params(_resolve_opts(None)) (#490)."""
    from faircode.profiler import _resolve_opts
    from faircode.provenance import public_params

    expected = public_params(_resolve_opts(None))

    script = (
        "require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(globalThis.FairCodeProfiler.publicParams({})));"
    )
    completed = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert json.loads(completed.stdout) == expected
    assert set(expected) == {
        "cross", "imbalance_flag", "intersection_floor", "min_group_size",
        "min_share", "missing_flag", "reference_flag",
    }
    assert "reference" not in expected


def test_python_js_profiler_parity_with_overrides_cross_and_thresholds(tmp_path):
    """Non-default options - --map/--cross/--reference/thresholds - only ever
    had cross-engine parity coverage for their default-off path (issue #376).
    A future change to either _resolve_opts (Python) or resolveOpts (JS), or
    to either engine's override-handling branch, could silently diverge here
    with nothing in this suite to catch it."""
    csv = CSV_PATHS["adult.csv"]
    overrides = {"education": "categorical"}
    reference = {"race": {"White": 0.7, "Black": 0.2, "Other": 0.1}}
    opts = {"cross": ["age", "race"], "min_group_size": 500, "reference": reference}

    python_result = profile(pd.read_csv(csv), overrides, opts)

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"overrides": overrides, "opts": opts}), encoding="utf-8")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result
    # Confirm the options actually took effect on both sides, not just that
    # both silently ignored them the same way.
    assert any(d["name"] == "education" and d["kind"] == "categorical"
               for d in python_result["dimensions"])
    assert python_result["intersections"][0]["dims"] == ["age", "race"]
    assert any("reference" in d for d in python_result["dimensions"])


def test_python_js_reject_out_of_range_min_share_parity(tmp_path):
    """Both engines reject an out-of-range tunable (min_share=1.5) rather
    than silently producing a self-contradictory report (#511)."""
    csv = CSV_PATHS["small.csv"]

    with pytest.raises(ValueError, match="min_share must be between 0 and 1"):
        profile(pd.read_csv(csv), opts={"min_share": 1.5})

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"opts": {"min_share": 1.5}}), encoding="utf-8")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert completed.returncode != 0
    assert "min_share must be between 0 and 1" in completed.stderr


def test_python_js_intersection_parity_keeps_non_numeric_age_sentinels(tmp_path):
    """labelize() gives a non-numeric age sentinel its own crosstab label on
    both engines, instead of mapping it to null and dropping the row (#524)."""
    csv = tmp_path / "age_sentinels.csv"
    ages = ["25", "30", "45", "unknown", "unknown", "unknown",
            "prefer not to say", "22", "33", "41"] * 3
    sexes = ["M", "F"] * 15
    csv.write_text(
        "age,sex\n" + "\n".join(a + "," + s for a, s in zip(ages, sexes)) + "\n",
        encoding="utf-8",
    )

    opts = {"cross": ["age", "sex"]}
    python_result = profile(pd.read_csv(csv, dtype={"age": str}), opts=opts)

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"opts": opts}), encoding="utf-8")
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)
    assert javascript_result == python_result

    a_labels = {c["a"] for c in python_result["intersections"][0]["cells"]}
    assert "prefer not to say" in a_labels


def test_python_js_cross_parity_on_unmatched_column(tmp_path):
    """An unmatched `cross` column raises the same error on both engines
    instead of the JS engine silently falling back to the first two detected
    dimensions with no error (#420)."""
    csv = CSV_PATHS["adult.csv"]

    with pytest.raises(ValueError, match="cross column\\(s\\) don't match any profiled dimension: nonexistent_col"):
        profile(pd.read_csv(csv), opts={"cross": ["age", "nonexistent_col"]})

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"opts": {"cross": ["age", "nonexistent_col"]}}), encoding="utf-8")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode != 0
    assert "cross column(s) don't match any profiled dimension: nonexistent_col" in completed.stderr


def test_python_js_reference_parity_on_unmatched_column(tmp_path):
    """A reference baseline whose column(s) don't match any profiled
    dimension raises the same error on both engines instead of the JS
    engine silently applying nothing (#419)."""
    csv = CSV_PATHS["adult.csv"]
    reference = {"totally_wrong_col": {"a": 0.5, "b": 0.5}}

    with pytest.raises(ValueError, match="reference file's column\\(s\\) don't match any profiled dimension: totally_wrong_col"):
        profile(pd.read_csv(csv), opts={"reference": reference})

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"opts": {"reference": reference}}), encoding="utf-8")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode != 0
    assert "reference file's column(s) don't match any profiled dimension: totally_wrong_col" in completed.stderr


def test_python_js_parse_reference_mixed_scale_parity(tmp_path):
    """parse_reference / parseReference decide percent-vs-fraction per column,
    not once across the whole table, so a reference file mixing conventions
    between columns parses identically on both engines (#513)."""
    from faircode.profiler import parse_reference

    ref_df = pd.DataFrame({
        "column": ["sex", "sex", "race", "race", "race"],
        "group": ["Female", "Male", "White", "Black", "Other"],
        "share": [0.6, 0.4, 70, 20, 10],
    })
    py_result = parse_reference(ref_df)
    assert py_result == {
        "sex": {"Female": 0.6, "Male": 0.4},
        "race": {"White": 0.7, "Black": 0.2, "Other": 0.1},
    }

    table_json = tmp_path / "table.json"
    table_json.write_text(json.dumps({
        "columns": list(ref_df.columns),
        "rows": ref_df.to_dict(orient="records"),
    }), encoding="utf-8")

    script = (
        "const fs=require('fs');"
        "require(process.argv[1]);"
        "const t=JSON.parse(fs.readFileSync(process.argv[2],'utf-8'));"
        "process.stdout.write(JSON.stringify(globalThis.FairCodeProfiler.parseReference(t)));"
    )
    completed = subprocess.run(
        ["node", "-e", script,
         str(REPO_ROOT / "assets" / "profiler-engine.js"), str(table_json)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert json.loads(completed.stdout) == py_result


def test_python_js_json_parity_inconsistent_keys():
    """Records-orient JSON where later records add columns the first one
    doesn't have (#144). The JS parseJSON() used to derive columns from only
    the first record, silently dropping any column that first appeared later
    - pandas' read_json unions keys across every record instead."""

    json_path = FIXTURES / "inconsistent_keys.json"

    python_result = profile(pd.read_json(json_path))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile-json", str(json_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result


def test_python_js_json_parity_columns_orientation():
    """Columns-orient JSON ({"col": {"0": v, ...}}, pandas' read_json default
    for a plain object) - #155 documented and tested this for the CLI, but
    the JS engine's parseJSON() only handled records/split and threw on it.
    Now handled the same way as the records branch (union of index keys)."""

    json_path = FIXTURES / "columns_orient.json"

    python_result = profile(pd.read_json(json_path))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile-json", str(json_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result


@requires_openpyxl
def test_python_js_xlsx_parity():
    """.xlsx support (#158) - the JS engine's parseXLSX() (via SheetJS,
    fetched from the same pinned CDN profiler.html loads) should agree with
    pandas.read_excel() on the same workbook. Skips if the CDN is
    unreachable rather than failing the suite - see scripts/engine-js.js.
    """
    xlsx_path = FIXTURES / "adult_sample.xlsx"

    python_result = profile(pd.read_excel(xlsx_path))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile-xlsx", str(xlsx_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode == 3:
        pytest.skip("SheetJS CDN unreachable: " + completed.stderr.strip())
    assert completed.returncode == 0, completed.stderr

    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result


def test_python_js_compare_parity_age_banding_mismatch(tmp_path):
    """The age-banding-mismatch guard (#318) agrees between engines too -
    `kind` alone can't detect it (it's set from the column name and is
    identical on both sides), so this exercises isAgeBandLabel()'s port of
    _is_age_band_label() directly, not just the already-covered common path.
    """
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    path_a.write_text("DOB\n" + "\n".join(["15/05/1980"] * 50 + ["20/06/1985"] * 50))
    path_b.write_text("DOB\n" + "\n".join(["18"] * 50 + ["35"] * 50))

    python_result = compare(
        profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b)), "a.csv", "b.csv"
    )

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_a), str(path_b)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result
    dim = python_result["dimensions"][0]
    assert dim["kind_mismatch"] is True
    assert dim["kind_a"] == dim["kind_b"] == "age"


@pytest.mark.parametrize("csv_name", list(CSV_PATHS))
def test_python_js_compare_parity(csv_name):
    """faircode.compare() and the JS engine's compare() should agree too (#111).

    Compares each fixture against itself - not meant to exercise every drift
    level, just to confirm the two independent compare()/compare_to_html()
    implementations (faircode/report.py and assets/profiler-compare.js) are
    working off identically-shaped, identically-valued structured data.
    """

    csv = CSV_PATHS[csv_name]
    df = pd.read_csv(csv)

    python_result = compare(profile(df), profile(df), "a.csv", "b.csv")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(csv), str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    # As above: flags are human-readable and float-formatted differently
    # between Python and JS. Compare the structured data instead. Names also
    # differ (JS uses the file's basename via path.basename, Python uses
    # whatever the caller passed) - normalize both before comparing.
    python_result = dict(python_result)
    javascript_result = dict(javascript_result)

    for result in (python_result, javascript_result):
        result.pop("flags", None)
        for side in ("a", "b"):
            result[side] = dict(result[side])
            result[side].pop("name", None)

    assert javascript_result == python_result


def test_threshold_input_recovers_panel_after_invalid_value():
    """profiler-ui.js's threshold-input handler must recover when the engine
    rejects the typed value (e.g. min_share 1.5, see #602): reprofile() fails
    and showError() hides the whole #results panel, including the very input
    the user needs to correct. The handler must revert that opt to its
    last-known-good value and retry once, the same recovery the mapping-select
    handler got in #466. Source-level check (mirrors
    test_compare_card_renderers_special_case_kind_mismatch) - this handler is
    DOM-coupled and has no unit harness."""
    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")

    marker = "thresholdInputs.forEach(function (input) {\n    input.addEventListener('input'"
    handler = src[src.index(marker):]
    handler = handler[: handler.index("\n  });\n")]

    # it must notice that the re-profile failed ...
    assert "if (!reprofile(false))" in handler
    # ... revert the offending opt to the value it held before ...
    assert "currentOpts[opt] = previous;" in handler
    # ... and retry once so the #results panel (and this input) come back
    assert handler.count("reprofile(false)") >= 2
