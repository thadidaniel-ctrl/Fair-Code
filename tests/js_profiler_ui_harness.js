/* Minimal DOM harness that drives assets/profiler-ui.js in Node so the
   #758 regression can be tested end to end - load a dataset, click "Check for
   proxy columns", then read what the two exports actually contain.

   profiler-ui.js is DOM-coupled, so this stubs just enough of the document to
   load it rather than pulling in a browser-automation dependency. The stubs
   only have to satisfy the controller; nothing here decides the behaviour
   under test.

   Prints one JSON object on stdout; tests/test_js_parity.py asserts on it. */
'use strict';

const path = require('path');
const fs = require('fs');

const REPO = path.resolve(__dirname, '..');
require(path.join(REPO, 'assets', 'profiler-engine.js'));

// ── Element stub ────────────────────────────────────────────────────────────
function makeEl(id) {
  const el = {
    id,
    hidden: false,
    textContent: '',
    value: '',
    files: null,
    dataset: {},
    style: { setProperty() {} },
    classList: { add() {}, remove() {}, contains: () => false, toggle() {} },
    children: [],
    // innerHTML assignment drops existing children, as a real DOM does.
    get innerHTML() { return el._html || ''; },
    set innerHTML(v) { el._html = v; if (v === '') el.children = []; },
    addEventListener(type, fn) { (this.handlers[type] = this.handlers[type] || []).push(fn); },
    appendChild(child) { this.children.push(child); return child; },
    removeChild(child) {
      this.children = this.children.filter((c) => c !== child);
      return child;
    },
    querySelectorAll: () => [],
    getAttribute: () => null,
    closest: () => null,
    contains: () => false,
    focus() {},
    click() { this.fire('click', {}); },
    select() {},
    scrollIntoView() {},
    fire(type, event) { (this.handlers[type] || []).forEach((fn) => fn(event)); },
    handlers: {},
  };
  return el;
}

const elements = new Map();
const documentStub = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeEl(id));
    return elements.get(id);
  },
  createElement: () => makeEl('created'),
  execCommand: () => true,
  body: makeEl('body'),
};
// querySelectorAll('[data-opt]') on the threshold panel must yield the five
// number inputs the controller wires up, each carrying data-opt.
const thresholdInputs = ['min_share', 'intersection_floor', 'imbalance_flag',
  'missing_flag', 'min_group_size']
  .map((opt) => { const el = makeEl('input-' + opt); el.dataset.opt = opt; return el; });

// ── Captured side effects ───────────────────────────────────────────────────
const captured = { html: null, json: null };
let lastBlobText = null;

// Node has Blob but not the object-URL pair the download path uses.
global.Blob = class Blob {
  constructor(parts) { lastBlobText = parts.join(''); }
};
global.URL.createObjectURL = () => 'blob:stub';
global.URL.revokeObjectURL = () => {};

class FileReaderStub {
  readAsText(file) { this.result = file.__text; this.onload(); }
  readAsArrayBuffer() { throw new Error('xlsx path not used by this harness'); }
}

global.document = documentStub;
global.FileReader = FileReaderStub;
global.window = {
  FairCodeProfiler: globalThis.FairCodeProfiler,
  matchMedia: () => ({ matches: false }),
  location: { search: '' },
  print() {},
};
// navigator is a read-only global in modern Node, so define rather than assign.
Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: {
    clipboard: {
      writeText(text) { captured.json = text; return Promise.resolve(); },
    },
  },
});

require(path.join(REPO, 'assets', 'profiler-ui.js'));

// The controller reads thresholdControls once at load, before we can reach in.
const thresholdControls = elements.get('thresholdControls')
  || (() => { documentStub.getElementById('thresholdControls'); return elements.get('thresholdControls'); })();
thresholdControls.querySelectorAll = () => thresholdInputs;

const el = (id) => documentStub.getElementById(id);

// ── Scenarios ───────────────────────────────────────────────────────────────
// A: zip_code is a perfect proxy for sex, region is independent of both.
// B: every column is independent, so the check must find nothing.
function datasetA() {
  const rows = ['sex,zip_code,region,age'];
  for (let i = 0; i < 200; i++) {
    const sex = i % 2 ? 'male' : 'female';
    const zip = sex === 'male' ? '10001' : '10002';
    rows.push([sex, zip, ['north', 'south', 'east', 'west'][(i / 2 | 0) % 4], 20 + (i % 55)].join(','));
  }
  return rows.join('\n') + '\n';
}
function datasetB() {
  const rows = ['sex,region,age'];
  for (let i = 0; i < 200; i++) {
    rows.push([i % 2 ? 'male' : 'female', ['north', 'south', 'east', 'west'][(i / 2 | 0) % 4], 20 + (i % 55)].join(','));
  }
  return rows.join('\n') + '\n';
}

function loadDataset(text, name) {
  el('fileInput').fire('change', {
    target: { files: [{ name, type: 'text/csv', __text: text }] },
  });
}

function checkProxyHints() {
  el('proxyHintsBtn').fire('click', {});
  return {
    panelHidden: el('proxyHintsBlock').hidden,
    items: el('proxyHintsList').children.map((li) => li.textContent),
  };
}

function exportHtml() {
  lastBlobText = null;
  el('downloadHtmlBtn').fire('click', {});
  captured.html = lastBlobText;
  return captured.html;
}

function exportJson() {
  captured.json = null;
  el('copyJsonBtn').fire('click', {});
  return captured.json;
}

// copyResultAsJSON is async (it awaits fileDigest); drain the microtask queue.
async function settle() { await new Promise((resolve) => setImmediate(resolve)); }

(async () => {
  const out = {};

  // 1. Proxied dataset, never checked: exports must carry no proxy_hints key.
  loadDataset(datasetA(), 'proxied.csv');
  out.beforeCheckHtml = exportHtml();
  out.beforeCheckJson = await (async () => { exportJson(); await settle(); return captured.json; })();
  out.beforeCheckJsonParsed = JSON.parse(captured.json);

  // 2. Check: hints must reach the screen AND both exports (#758).
  out.check = checkProxyHints();
  out.afterCheckHtml = exportHtml();
  exportJson();
  await settle();
  out.afterCheckJson = JSON.parse(captured.json);

  // 3. Profiling a different dataset must not carry the old hints forward.
  loadDataset(datasetB(), 'independent.csv');
  out.afterReprofileJson = await (async () => { exportJson(); await settle(); return JSON.parse(captured.json); })();
  out.afterReprofileHtml = exportHtml();

  // 4. Check the independent dataset: no hints, and none fabricated or stale.
  out.emptyCheck = checkProxyHints();
  out.emptyCheckHtml = exportHtml();
  exportJson();
  await settle();
  out.emptyCheckJson = JSON.parse(captured.json);

  process.stdout.write(JSON.stringify(out));
})().catch((err) => {
  process.stderr.write(String((err && err.stack) || err));
  process.exit(1);
});
