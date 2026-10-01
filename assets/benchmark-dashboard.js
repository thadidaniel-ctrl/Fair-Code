/* ════════════════════════════════════════════════════════════════════════
   Fair Code - Benchmark Results Dashboard

   Interactive, filterable explorer for results/results_fairness.csv and
   results/results_performance.csv (issue #744, ROADMAP.md Phase 5) -
   mirrors the Open Dataset Profiler's web/CLI split: same numbers the
   benchmark harness writes to results/, no server, nothing uploaded.
   Depends on assets/profiler-engine.js for CSV parsing only (parseCSV);
   none of the demographic-profiling logic in that file is used here.
   ════════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  var E = window.FairCodeProfiler;

  var FAIRNESS_COLUMNS = ['audit', 'strategy', 'model', 'protected_attribute', 'metric',
    'value', 'ci_low', 'ci_high', 'p_value', 'significant',
    'n_disadvantaged', 'n_advantaged', 'small_sample_warning', 'note'];
  var PERFORMANCE_COLUMNS = ['audit', 'strategy', 'model', 'metric', 'value', 'ci_low', 'ci_high', 'n'];

  var FILTER_FIELDS = {
    fairness: ['audit', 'strategy', 'model', 'protected_attribute', 'metric'],
    performance: ['audit', 'strategy', 'model', 'metric']
  };

  var state = {
    fairness: null,   // { rows: [...] } once loaded
    performance: null,
    tab: 'fairness',
    filters: { fairness: {}, performance: {} },
    significantOnly: false,
    sort: { fairness: null, performance: null } // { field, dir }
  };

  var tabButtons = Array.prototype.slice.call(document.querySelectorAll('.bench-tab'));
  var loadBundledBtn = document.getElementById('loadBundledBtn');
  var dropzone = document.getElementById('benchDropzone');
  var fileInput = document.getElementById('benchFileInput');
  var errorEl = document.getElementById('benchError');
  var statusEl = document.getElementById('benchStatus');
  var resultsEl = document.getElementById('benchResults');
  var filterBar = document.getElementById('benchFilters');
  var significantOnlyInput = document.getElementById('significantOnlyInput');
  var summaryEl = document.getElementById('benchSummary');
  var tableHost = document.getElementById('benchTable');
  var chartHost = document.getElementById('benchChart');
  var chartNote = document.getElementById('benchChartNote');

  function isMissing(v) { return v === null || v === undefined || v === ''; }
  function toNum(v) { return isMissing(v) ? null : parseFloat(v); }
  function toBool(v) { return v === 'True'; }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = !msg;
  }

  // ── Loading ───────────────────────────────────────────────────────────
  function detectKind(columns) {
    if (columns.indexOf('protected_attribute') !== -1) return 'fairness';
    if (columns.indexOf('audit') !== -1 && columns.indexOf('metric') !== -1) return 'performance';
    return null;
  }

  function ingest(kind, table) {
    var rows = table.rows.map(function (r) {
      var row = {};
      (kind === 'fairness' ? FAIRNESS_COLUMNS : PERFORMANCE_COLUMNS).forEach(function (col) {
        row[col] = r[col] === undefined ? null : r[col];
      });
      row.value = toNum(row.value);
      row.ci_low = toNum(row.ci_low);
      row.ci_high = toNum(row.ci_high);
      if (kind === 'fairness') {
        row.p_value = toNum(row.p_value);
        row.significant = toBool(row.significant);
        row.n_disadvantaged = toNum(row.n_disadvantaged);
        row.n_advantaged = toNum(row.n_advantaged);
        row.small_sample_warning = toBool(row.small_sample_warning);
      } else {
        row.n = toNum(row.n);
      }
      return row;
    });
    state[kind] = { rows: rows };
    state.filters[kind] = {};
    state.sort[kind] = null;
  }

  function loadText(kind, text, sourceName) {
    var table = E.parseCSV(text);
    var detected = detectKind(table.columns);
    if (detected && detected !== kind) kind = detected;
    ingest(kind, table);
    statusEl.textContent = (statusEl.textContent ? statusEl.textContent + ' · ' : '') +
      sourceName + ' (' + table.rows.length + ' rows, ' + kind + ')';
  }

  function loadBundled() {
    showError('');
    statusEl.textContent = 'Loading results/results_fairness.csv and results/results_performance.csv…';
    Promise.all([
      fetch('results/results_fairness.csv').then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.text();
      }),
      fetch('results/results_performance.csv').then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.text();
      })
    ]).then(function (texts) {
      statusEl.textContent = '';
      loadText('fairness', texts[0], 'results/results_fairness.csv');
      loadText('performance', texts[1], 'results/results_performance.csv');
      render();
    }).catch(function (err) {
      statusEl.textContent = '';
      showError('Could not fetch the bundled results/ CSVs (' + err.message + '). ' +
        'This works once the site is served over HTTP - locally over file:// the browser ' +
        'blocks it. Drop results_fairness.csv / results_performance.csv below instead.');
    });
  }

  function readDroppedFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var table = E.parseCSV(String(reader.result));
        var kind = detectKind(table.columns);
        if (!kind) {
          showError(file.name + ' does not look like a results_fairness.csv or ' +
            'results_performance.csv export (no "protected_attribute" or "audit"/"metric" columns).');
          return;
        }
        showError('');
        ingest(kind, table);
        statusEl.textContent = file.name + ' (' + table.rows.length + ' rows, ' + kind + ')';
        render();
      } catch (err) {
        showError('Could not parse ' + file.name + ': ' + err.message);
      }
    };
    reader.onerror = function () { showError('Could not read ' + file.name + '.'); };
    reader.readAsText(file);
  }

  loadBundledBtn.addEventListener('click', loadBundled);
  dropzone.addEventListener('click', function () { fileInput.click(); });
  dropzone.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', function () {
    Array.prototype.forEach.call(fileInput.files, readDroppedFile);
    fileInput.value = '';
  });
  ['dragenter', 'dragover'].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.add('dragover'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) {
      e.preventDefault();
      if (ev === 'dragleave' && dropzone.contains(e.relatedTarget)) return;
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', function (e) {
    var files = e.dataTransfer && e.dataTransfer.files;
    if (files) Array.prototype.forEach.call(files, readDroppedFile);
  });

  // ── Tabs ──────────────────────────────────────────────────────────────
  tabButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      state.tab = btn.dataset.tab;
      tabButtons.forEach(function (b) { b.setAttribute('aria-selected', String(b === btn)); });
      render();
    });
  });

  significantOnlyInput.addEventListener('change', function () {
    state.significantOnly = significantOnlyInput.checked;
    render();
  });

  // ── Filtering + sorting ───────────────────────────────────────────────
  function uniqueValues(rows, field) {
    var seen = Object.create(null), out = [];
    rows.forEach(function (r) {
      var v = r[field];
      if (v !== null && !seen[v]) { seen[v] = 1; out.push(v); }
    });
    out.sort();
    return out;
  }

  function filteredRows(kind) {
    var data = state[kind];
    if (!data) return [];
    var filters = state.filters[kind];
    return data.rows.filter(function (r) {
      if (kind === 'fairness' && state.significantOnly && !r.significant) return false;
      return FILTER_FIELDS[kind].every(function (f) {
        return !filters[f] || r[f] === filters[f];
      });
    });
  }

  function sortedRows(kind, rows) {
    var sort = state.sort[kind];
    if (!sort) return rows;
    var out = rows.slice();
    out.sort(function (a, b) {
      var x = a[sort.field], y = b[sort.field];
      if (x === null && y === null) return 0;
      if (x === null) return 1;
      if (y === null) return -1;
      if (x < y) return sort.dir === 'asc' ? -1 : 1;
      if (x > y) return sort.dir === 'asc' ? 1 : -1;
      return 0;
    });
    return out;
  }

  // ── Rendering: filter bar ─────────────────────────────────────────────
  function renderFilters(kind) {
    filterBar.innerHTML = '';
    var data = state[kind];
    if (!data) return;
    FILTER_FIELDS[kind].forEach(function (field) {
      var label = document.createElement('label');
      label.className = 'bench-filter';
      var caption = document.createElement('span');
      caption.textContent = field.replace(/_/g, ' ');
      var select = document.createElement('select');
      select.dataset.field = field;
      var allOpt = document.createElement('option');
      allOpt.value = ''; allOpt.textContent = 'All';
      select.appendChild(allOpt);
      uniqueValues(data.rows, field).forEach(function (v) {
        var opt = document.createElement('option');
        opt.value = v; opt.textContent = v;
        if (state.filters[kind][field] === v) opt.selected = true;
        select.appendChild(opt);
      });
      select.addEventListener('change', function () {
        state.filters[kind][field] = select.value || null;
        render();
      });
      label.appendChild(caption);
      label.appendChild(select);
      filterBar.appendChild(label);
    });
  }

  // ── Rendering: table ──────────────────────────────────────────────────
  function ciText(r) {
    return (r.ci_low !== null && r.ci_high !== null)
      ? r.ci_low.toFixed(4) + ' - ' + r.ci_high.toFixed(4) : '';
  }

  function headerCell(kind, field, label) {
    var sort = state.sort[kind];
    var active = sort && sort.field === field;
    var arrow = active ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : '';
    return '<th><button type="button" class="bench-sort-btn" data-field="' + field + '">' +
      esc(label) + arrow + '</button></th>';
  }

  function renderTable(kind, rows) {
    if (!rows.length) {
      tableHost.innerHTML = '<p class="section-note">No rows match the current filters.</p>';
      return;
    }
    var isFairness = kind === 'fairness';
    var head = isFairness
      ? headerCell(kind, 'audit', 'Audit') + headerCell(kind, 'strategy', 'Strategy') +
        headerCell(kind, 'model', 'Model') + headerCell(kind, 'protected_attribute', 'Attribute') +
        headerCell(kind, 'metric', 'Metric') + headerCell(kind, 'value', 'Value') +
        '<th>95% CI</th>' + headerCell(kind, 'p_value', 'p-value') + headerCell(kind, 'significant', 'Sig.')
      : headerCell(kind, 'audit', 'Audit') + headerCell(kind, 'strategy', 'Strategy') +
        headerCell(kind, 'model', 'Model') + headerCell(kind, 'metric', 'Metric') +
        headerCell(kind, 'value', 'Value') + '<th>95% CI</th>' + headerCell(kind, 'n', 'N');

    var body = rows.map(function (r) {
      var valueText = r.value === null ? '' : r.value.toFixed(4);
      if (isFairness) {
        var sigCls = r.significant ? 'bench-sig-yes' : 'bench-sig-no';
        var rowCls = r.small_sample_warning ? ' class="bench-row-warn"' : '';
        return '<tr' + rowCls + (r.note ? ' title="' + esc(r.note) + '"' : '') + '>' +
          '<td>' + esc(r.audit) + '</td><td>' + esc(r.strategy) + '</td><td>' + esc(r.model) + '</td>' +
          '<td>' + esc(r.protected_attribute) + '</td><td>' + esc(r.metric) + '</td>' +
          '<td>' + valueText + '</td><td>' + ciText(r) + '</td>' +
          '<td>' + (r.p_value === null ? '' : r.p_value.toExponential(2)) + '</td>' +
          '<td class="' + sigCls + '">' + (r.significant ? 'yes' : 'no') + '</td></tr>';
      }
      return '<tr>' +
        '<td>' + esc(r.audit) + '</td><td>' + esc(r.strategy) + '</td><td>' + esc(r.model) + '</td>' +
        '<td>' + esc(r.metric) + '</td><td>' + valueText + '</td><td>' + ciText(r) + '</td>' +
        '<td>' + (r.n === null ? '' : r.n) + '</td></tr>';
    }).join('');

    tableHost.innerHTML = '<table class="bench-table"><thead><tr>' + head + '</tr></thead>' +
      '<tbody>' + body + '</tbody></table>';

    Array.prototype.forEach.call(tableHost.querySelectorAll('.bench-sort-btn'), function (btn) {
      btn.addEventListener('click', function () {
        var field = btn.dataset.field;
        var current = state.sort[kind];
        var dir = (current && current.field === field && current.dir === 'asc') ? 'desc' : 'asc';
        state.sort[kind] = { field: field, dir: dir };
        render();
      });
    });
  }

  // ── Rendering: chart (fairness only) ───────────────────────────────────
  function renderChart(rows) {
    var filters = state.filters.fairness;
    if (!filters.metric || !filters.protected_attribute) {
      chartHost.innerHTML = '';
      chartNote.textContent = 'Pick a metric and a protected attribute above to chart every ' +
        'audit x strategy x model combination on the same scale.';
      chartHost.hidden = true;
      return;
    }
    chartNote.textContent = '';
    chartHost.hidden = false;
    if (!rows.length) { chartHost.innerHTML = ''; return; }

    var maxAbs = rows.reduce(function (m, r) {
      return r.value === null ? m : Math.max(m, Math.abs(r.value));
    }, 0) || 1;

    var sorted = rows.slice().sort(function (a, b) { return Math.abs(b.value || 0) - Math.abs(a.value || 0); });
    chartHost.innerHTML = sorted.map(function (r) {
      var w = r.value === null ? 0 : (Math.abs(r.value) / maxAbs) * 100;
      var cls = r.significant ? 'bad' : 'good';
      var label = r.audit + ' · ' + r.strategy + ' · ' + r.model;
      return '<div class="bar-row">' +
        '<span class="bar-label" title="' + esc(label) + '">' + esc(label) + '</span>' +
        '<span class="bar-track"><span class="bar-fill ' + cls + '" style="width:' + w.toFixed(1) + '%"></span></span>' +
        '<span class="bar-pct">' + (r.value === null ? 'n/a' : r.value.toFixed(4)) + '</span>' +
        '</div>';
    }).join('');
  }

  // ── Orchestrator ────────────────────────────────────────────────────────
  function render() {
    var kind = state.tab;
    var data = state[kind];
    resultsEl.hidden = !(state.fairness || state.performance);
    document.getElementById('benchChartBlock').hidden = kind !== 'fairness';
    if (!data) {
      filterBar.innerHTML = '';
      significantOnlyInput.parentElement.hidden = true;
      summaryEl.textContent = '';
      tableHost.innerHTML = '<p class="section-note">Load ' + kind + ' results above to explore them.</p>';
      if (kind === 'fairness') { chartHost.innerHTML = ''; chartHost.hidden = true; chartNote.textContent = ''; }
      return;
    }
    significantOnlyInput.parentElement.hidden = kind !== 'fairness';
    renderFilters(kind);
    var rows = sortedRows(kind, filteredRows(kind));
    var sigCount = kind === 'fairness' ? rows.filter(function (r) { return r.significant; }).length : null;
    summaryEl.textContent = rows.length.toLocaleString() + ' of ' + data.rows.length.toLocaleString() + ' rows shown' +
      (sigCount !== null ? ' · ' + sigCount.toLocaleString() + ' significant' : '');
    renderTable(kind, rows);
    if (kind === 'fairness') renderChart(rows);
  }

  render();
})();
