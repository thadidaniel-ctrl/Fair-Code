"""Behavioral tests for the benchmark results dashboard (issue #744).

assets/benchmark-dashboard.js is DOM-coupled (no jsdom in this repo, matching
the rest of the web profiler's test story - see test_js_parity.py's #740
tests), so this drives the REAL file with a minimal hand-built DOM stub
rather than re-implementing its logic in the test. Every expected number is
computed from the real results/*.csv files via pandas rather than hardcoded,
so this stays correct as the benchmark harness's results/ evolves (no paper
freeze is in effect - see CLAUDE.md).
"""

import json
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# A minimal DOM stub: enough getElementById/createElement/appendChild/
# addEventListener plumbing to actually execute assets/benchmark-dashboard.js
# end to end (fetch mocked to serve the real results/ CSVs), then drive a
# filter-select change, a "significant only" checkbox toggle, a sort-header
# click, and a tab switch - the same interactions a person would perform in
# a browser - and read back what actually got rendered.
_DOM_STUB = r"""
'use strict';
var fs = require('fs');
var path = require('path');
var REPO = process.argv[1];

function makeEl(id) {
  var el = {
    id: id,
    _listeners: {},
    dataset: {},
    style: { setProperty: function () {} },
    classList: { add: function () {}, remove: function () {}, contains: function () { return false; } },
    hidden: false,
    textContent: '',
    _innerHTML: '',
    parentElement: { hidden: false },
    addEventListener: function (type, fn) {
      (el._listeners[type] = el._listeners[type] || []).push(fn);
    },
    setAttribute: function (k, v) { el['attr_' + k] = v; },
    getAttribute: function (k) { return el['attr_' + k]; },
    appendChild: function (child) { (el._children = el._children || []).push(child); return child; },
    querySelectorAll: function (sel) {
      if (sel === '.bench-sort-btn') return el._sortButtons || [];
      return [];
    },
    click: function () { (el._listeners.click || []).forEach(function (f) { f(); }); },
  };
  Object.defineProperty(el, 'innerHTML', {
    get: function () { return el._innerHTML; },
    set: function (html) {
      el._innerHTML = html;
      el._sortButtons = [];
      var re = /class="bench-sort-btn" data-field="([^"]+)"/g, m;
      while ((m = re.exec(html))) {
        (function (field) {
          var btn = { dataset: { field: field }, _listeners: {},
            addEventListener: function (t, f) { (btn._listeners[t] = btn._listeners[t] || []).push(f); },
            click: function () { (btn._listeners.click || []).forEach(function (f) { f(); }); } };
          el._sortButtons.push(btn);
        })(m[1]);
      }
    },
  });
  return el;
}

var ids = ['loadBundledBtn', 'benchDropzone', 'benchFileInput', 'benchError', 'benchStatus',
  'benchResults', 'benchFilters', 'significantOnlyInput', 'benchSummary', 'benchTable',
  'benchChart', 'benchChartNote', 'benchChartBlock'];
var elements = {};
ids.forEach(function (id) { elements[id] = makeEl(id); });

var createdSelects = {};
global.document = {
  getElementById: function (id) {
    if (!elements[id]) elements[id] = makeEl(id);
    return elements[id];
  },
  querySelectorAll: function (sel) {
    if (sel === '.bench-tab') return global.__tabButtons;
    return [];
  },
  createElement: function (tag) {
    var node = makeEl('(created:' + tag + ')');
    node.tagName = tag;
    node.appendChild = function (child) {
      (node._children = node._children || []).push(child);
      if (tag === 'label' && child.tagName === 'select' && child.dataset.field) {
        createdSelects[child.dataset.field] = child;
      }
      return child;
    };
    if (tag === 'select') {
      node._options = [];
      node.appendChild = function (opt) { node._options.push(opt); return opt; };
      Object.defineProperty(node, 'value', {
        get: function () { return node._value || ''; },
        set: function (v) { node._value = v; },
      });
    }
    return node;
  },
};

function makeTabButton(tab, selected) {
  var b = makeEl('tab-' + tab);
  b.dataset.tab = tab;
  b.setAttribute('aria-selected', String(selected));
  return b;
}
global.__tabButtons = [makeTabButton('fairness', true), makeTabButton('performance', false)];

var fetchMap = {
  'results/results_fairness.csv': fs.readFileSync(path.join(REPO, 'results', 'results_fairness.csv'), 'utf-8'),
  'results/results_performance.csv': fs.readFileSync(path.join(REPO, 'results', 'results_performance.csv'), 'utf-8'),
};
global.fetch = function (url) {
  return Promise.resolve({ ok: true, text: function () { return Promise.resolve(fetchMap[url]); } });
};
global.window = global;

require(path.join(REPO, 'assets', 'profiler-engine.js'));

var results = {};

(async function () {
  require(path.join(REPO, 'assets', 'benchmark-dashboard.js'));

  elements.loadBundledBtn.click();
  await new Promise(function (r) { setTimeout(r, 20); });

  results.results_hidden_after_load = elements.benchResults.hidden;
  results.summary_unfiltered = elements.benchSummary.textContent;

  var auditSelect = createdSelects['audit'];
  results.audit_select_found = !!auditSelect;
  if (auditSelect) {
    auditSelect.value = 'compas';
    (auditSelect._listeners.change || []).forEach(function (f) { f(); });
  }
  results.summary_after_audit_filter = elements.benchSummary.textContent;
  var otherAudits = ['ai_fair_recruitment', 'benefits_denial', 'german_credit_lending',
    'healthcare_readmission', 'insurance_denial', 'tenant_screening'];
  results.table_has_only_compas = otherAudits.every(function (a) {
    return elements.benchTable.innerHTML.indexOf(a) === -1;
  });

  elements.significantOnlyInput.checked = true;
  (elements.significantOnlyInput._listeners.change || []).forEach(function (f) { f(); });
  results.summary_after_significant_only = elements.benchSummary.textContent;

  function firstRowValue() {
    var body = (elements.benchTable.innerHTML.match(/<tbody>([\s\S]*?)<\/tbody>/) || [])[1] || '';
    var firstTr = (body.match(/<tr[^>]*>([\s\S]*?)<\/tr>/) || [])[1] || '';
    var tds = [];
    var re = /<td[^>]*>([^<]*)<\/td>/g, m;
    while ((m = re.exec(firstTr))) tds.push(m[1]);
    return tds[5] === undefined || tds[5] === '' ? null : parseFloat(tds[5]);
  }
  var valueBtn = elements.benchTable._sortButtons.filter(function (b) { return b.dataset.field === 'value'; })[0];
  results.value_sort_btn_found = !!valueBtn;
  if (valueBtn) valueBtn.click(); // ascending
  results.first_row_value_asc = firstRowValue();
  var valueBtn2 = elements.benchTable._sortButtons.filter(function (b) { return b.dataset.field === 'value'; })[0];
  if (valueBtn2) valueBtn2.click(); // descending
  results.first_row_value_desc = firstRowValue();

  var perfTab = global.__tabButtons[1];
  perfTab.click();
  results.performance_tab_summary = elements.benchSummary.textContent;
  results.chart_block_hidden_on_performance = elements.benchChartBlock.hidden;

  process.stdout.write(JSON.stringify(results));
})();
"""


def _run_dom_stub():
    completed = subprocess.run(
        ["node", "-e", _DOM_STUB, str(REPO_ROOT)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return json.loads(completed.stdout)


def test_benchmark_dashboard_loads_filters_sorts_and_switches_tabs():
    """Drives the real assets/benchmark-dashboard.js through the interactions
    a person would perform in a browser, against the real results/*.csv, and
    checks every number against pandas ground truth rather than a hardcoded
    snapshot (results/ has no paper freeze and is expected to keep moving)."""
    fairness = pd.read_csv(REPO_ROOT / "results" / "results_fairness.csv")
    performance = pd.read_csv(REPO_ROOT / "results" / "results_performance.csv")

    total = len(fairness)
    total_significant = int(fairness["significant"].sum())
    compas = fairness[fairness["audit"] == "compas"]
    compas_significant = compas[compas["significant"]]

    r = _run_dom_stub()

    assert r["results_hidden_after_load"] is False
    assert r["summary_unfiltered"] == f"{total:,} of {total:,} rows shown · {total_significant:,} significant"

    assert r["audit_select_found"] is True
    assert r["table_has_only_compas"] is True
    assert r["summary_after_audit_filter"] == (
        f"{len(compas):,} of {total:,} rows shown · {int(compas['significant'].sum()):,} significant"
    )

    assert r["summary_after_significant_only"] == (
        f"{len(compas_significant):,} of {total:,} rows shown · {len(compas_significant):,} significant"
    )

    assert r["value_sort_btn_found"] is True
    assert r["first_row_value_asc"] == _rounded(compas_significant["value"].min())
    assert r["first_row_value_desc"] == _rounded(compas_significant["value"].max())

    assert r["performance_tab_summary"] == f"{len(performance):,} of {len(performance):,} rows shown"
    assert r["chart_block_hidden_on_performance"] is True


def _rounded(x):
    # Table cells are rendered with .toFixed(4); round the pandas ground
    # truth the same way for an exact equality check.
    return round(float(x), 4)


def test_benchmark_dashboard_ui_wiring_present_in_html_and_css():
    """Source-level check (matches the #740 precedent for DOM-coupled code):
    the dashboard page must expose the ids benchmark-dashboard.js binds to,
    and ROADMAP.md's Phase 5 checklist item should be checked off now that
    this exists."""
    html = (REPO_ROOT / "benchmark.html").read_text(encoding="utf-8")
    for expected_id in ["loadBundledBtn", "benchDropzone", "benchFileInput", "benchError",
                         "benchStatus", "benchResults", "benchFilters", "significantOnlyInput",
                         "benchSummary", "benchTable", "benchChart", "benchChartNote", "benchChartBlock"]:
        assert f'id="{expected_id}"' in html, expected_id

    css = (REPO_ROOT / "assets" / "benchmark.css").read_text(encoding="utf-8")
    assert ".bench-table" in css

    roadmap = (REPO_ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "- [x] Fairness dashboard for the benchmark harness results" in roadmap
