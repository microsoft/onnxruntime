// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Dependency-free Node controller: alternating unprofiled captures and a local comparison report.
import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export function median(values) {
  if (!values.length) return null;
  const sorted = values.toSorted((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function summarize(captures) {
  if (!captures.length) throw new Error('No captures.');
  const first = captures[0].capture;
  const variants = new Map();
  for (const { id, label, round, capture } of captures) {
    if (capture.trace) throw new Error('Performance comparisons require unprofiled captures.');
    for (const key of ['prompt_sha256', 'prompt_tokens', 'decode_steps', 'backend', 'ort_version', 'ort_module']) {
      if (capture[key] !== first[key]) throw new Error(`Incompatible capture ${id}: ${key}.`);
    }
    if (!Array.isArray(capture.runs) || !capture.runs.length) throw new Error(`Empty capture ${id}.`);
    const variant = variants.get(id) ?? { id, label, rounds: {}, runs: [], generated: [], settings: capture };
    for (const key of [
      'model_sha256',
      'static_kv',
      'graph_capture',
      'max_pending_dispatches',
      'intra_op_threads',
      'disabled_optimizers',
    ]) {
      if (JSON.stringify(variant.settings[key]) !== JSON.stringify(capture[key])) {
        throw new Error(`Configuration changed within ${id}: ${key}.`);
      }
    }
    for (const run of capture.runs.filter((run) => !run.warmup)) {
      if (!['prefill', 'decode'].includes(run.stage)) throw new Error('Unknown decoder stage.');
      for (const key of ['wall_ms', 'step_wall_ms']) {
        if (!Number.isFinite(run[key]) || run[key] < 0) throw new Error(`Invalid ${key}.`);
      }
      variant.runs.push(run);
      (variant.rounds[round] ??= []).push(run);
    }
    variant.generated.push(...(capture.generated_token_ids ?? []).slice(capture.warmup));
    variants.set(id, variant);
  }
  const reference = [...variants.values()][0].generated[0];
  return [...variants.values()].map(({ id, label, rounds, runs, generated, settings }) => {
    const metrics = {};
    for (const stage of ['prefill', 'decode']) {
      const selected = runs.filter((run) => run.stage === stage);
      metrics[stage] = { samples: selected.length };
      if (!selected.length) throw new Error(`No measured ${stage} samples for ${id}.`);
      for (const key of ['wall_ms', 'step_wall_ms']) {
        metrics[stage][key] = median(selected.map((run) => run[key]));
        metrics[stage][`${key}_rounds`] = Object.values(rounds).map((rows) =>
          median(rows.filter((run) => run.stage === stage).map((run) => run[key])),
        );
      }
    }
    return {
      id,
      label,
      metrics,
      sameTokens:
        !!reference &&
        generated.length > 0 &&
        generated.every((tokens) => JSON.stringify(tokens) === JSON.stringify(reference)),
      settings: Object.fromEntries(
        [
          'model',
          'model_sha256',
          'backend',
          'ort_version',
          'max_pending_dispatches',
          'intra_op_threads',
          'static_kv',
          'graph_capture',
          'disabled_optimizers',
        ].map((key) => [key, settings[key]]),
      ),
    };
  });
}

const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );

export function renderReport(summary, manifest) {
  const payload = JSON.stringify({ summary, manifest }).replace(/</g, '\\u003c');
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WebGPU decoder performance lab</title><style>
:root {
  color-scheme: dark;
  font: 16px system-ui;
  background: #0b1220;
  color: #e5edf8;
}
body {
  max-width: 1180px;
  margin: 48px auto;
  padding: 0 24px;
}
h1 {
  font-size: 38px;
  letter-spacing: -1px;
  margin: 12px 0;
}
.eyebrow {
  color: #5eead4;
  font-weight: 650;
  letter-spacing: 2px;
  font-size: 12px;
}
p {
  color: #a7b6cd;
  line-height: 1.6;
}
a {
  color: #67e8f9;
}
button {
  background: #15263e;
  border: 1px solid #30445e;
  border-radius: 8px;
  color: inherit;
  padding: 10px 16px;
  cursor: pointer;
}
button.active {
  background: #0d534e;
  border-color: #2dd4bf;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 16px;
  margin: 28px 0;
}
.card,
section {
  background: #101d30;
  border: 1px solid #21344e;
  border-radius: 14px;
  padding: 22px;
}
.card strong {
  display: block;
  font-size: 30px;
  margin: 12px 0;
}
.card span {
  font-size: 13px;
  color: #99aac2;
}
.barrow {
  display: grid;
  grid-template-columns: 215px 1fr 115px;
  align-items: center;
  gap: 14px;
  margin: 22px 0;
}
.bar {
  height: 30px;
  border-radius: 5px;
  background: linear-gradient(90deg, #14b8a6, #67e8f9);
}
.barrow:first-child .bar {
  background: #586e90;
}
.value {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
th,
td {
  text-align: left;
  padding: 14px 10px;
  border-bottom: 1px solid #25364e;
}
th {
  color: #92aacb;
}
section {
  margin: 22px 0;
  overflow: auto;
}
h2 {
  font-size: 20px;
  margin-top: 0;
}
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.note {
  font-size: 13px;
}
details {
  margin-top: 16px;
}
pre {
  white-space: pre-wrap;
  font-size: 12px;
  color: #a7b6cd;
}
.tag {
  color: #5eead4;
}
footer {
  margin: 32px 0;
  color: #8da1bf;
  font-size: 13px;
}
@media (max-width: 700px) {
  .barrow {
    grid-template-columns: 125px 1fr 90px;
  }
  body {
    margin-top: 22px;
  }
  h1 {
    font-size: 29px;
  }
}

</style><div class="eyebrow">WEBGPU · DECODER PERFORMANCE LAB</div><h1>${escapeHtml(manifest.title ?? 'Measured decoder optimization')}</h1>
<p>Alternating forward/reverse runs on one device. Performance samples have profiling disabled. Open the aligned CPU/GPU timelines below to inspect submission gaps and shader work.</p>
<div class="controls"><button id="run">Run + logits readback</button><button id="step" class="active">Complete decoder step</button><button id="download">Download measurements</button></div><div class="cards" id="cards"></div>
<section><h2>Decode latency · lower is better</h2><div id="bars"></div><p class="note">Complete step also includes input upload, binding and greedy token selection. Model/session initialization and warmup are excluded. Round ranges expose timing variation.</p></section>
<section><h2>Measured configurations</h2><table><thead><tr><th>Configuration</th><th>Prefill</th><th>Decode</th><th>Speedup</th><th>Round medians</th><th>Samples</th></tr></thead><tbody id="rows"></tbody></table></section>
<section><h2>Aligned CPU + GPU traces</h2>${(manifest.timelines ?? [])
    .map((item) => {
      if (!/^[\w./-]+\.html$/.test(item.href) || item.href.startsWith('/') || item.href.split('/').includes('..'))
        throw new Error('Timeline links must be relative HTML files within the output directory.');
      return `<p><a href="${escapeHtml(item.href)}">${escapeHtml(item.label)}</a></p>`;
    })
    .join(
      '',
    )}<p class="note">Trace gaps are measured only inside the first-to-last GPU dispatch interval; timing/query overhead can enlarge them. They are not an unprofiled GPU utilization measurement.</p></section>
<section><h2>Numerical validation and scope</h2><p>${escapeHtml(manifest.validation ?? 'Compare logits, live KV and generated tokens separately. Matching text alone does not establish numerical equivalence.')}</p><p class="note">${escapeHtml(manifest.scope ?? 'Results apply to this device and workload; no universal WebGPU default is changed.')}</p><details><summary>Reproduction manifest and configuration</summary><pre id="manifest"></pre></details></section><footer>Self-contained report · offline Node generation · no runtime profiling overhead</footer>
<script type="application/json" id="data">${payload}</script><script>
const data = JSON.parse(document.getElementById('data').textContent);
let metric = 'step_wall_ms';
const fmt = (n) => (n === null ? '—' : n.toFixed(3));
const el = (id) => document.getElementById(id);
function render() {
  const base = data.summary[0].metrics.decode[metric];
  const max = Math.max(...data.summary.map((v) => v.metrics.decode[metric] ?? 0), 0.001);
  const recommended = data.summary.find((v) => v.id === data.manifest.recommendedId);
  const best =
    recommended?.metrics.decode[metric] ??
    Math.min(...data.summary.map((v) => v.metrics.decode[metric] ?? Infinity));
  el('cards').replaceChildren();
  for (const [title, value, sub] of [
    ['Baseline decode', fmt(base) + ' ms', 'Median measured latency'],
    [
      recommended ? 'Validated preset' : 'Best measured decode',
      fmt(best) + ' ms',
      recommended ? recommended.label : 'See numerical scope before choosing',
    ],
    [
      recommended ? 'Validated speedup' : 'Maximum speedup',
      (base / best).toFixed(2) + '×',
      'On this workload and device',
    ],
  ]) {
    const card = document.createElement('div');
    card.className = 'card';
    const label = document.createElement('span');
    label.textContent = title;
    const big = document.createElement('strong');
    big.textContent = value;
    const detail = document.createElement('span');
    detail.textContent = sub;
    card.append(label, big, detail);
    el('cards').append(card);
  }
  el('bars').replaceChildren();
  el('rows').replaceChildren();
  for (const v of data.summary) {
    const d = v.metrics.decode;
    const row = document.createElement('div');
    row.className = 'barrow';
    const name = document.createElement('span');
    name.textContent = v.label;
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.width = (100 * (d[metric] ?? 0)) / max + '%';
    const value = document.createElement('span');
    value.className = 'value';
    value.textContent = fmt(d[metric]) + ' ms';
    row.append(name, bar, value);
    el('bars').append(row);
    const tr = document.createElement('tr');
    const rounds = d[metric + '_rounds'].filter((x) => x !== null);
    for (const content of [
      v.label,
      fmt(v.metrics.prefill[metric]) + ' ms',
      fmt(d[metric]) + ' ms',
      d[metric] === null ? '—' : (base / d[metric]).toFixed(2) + '×',
      rounds.length ? fmt(Math.min(...rounds)) + '–' + fmt(Math.max(...rounds)) + ' ms' : '—',
      String(d.samples),
    ]) {
      const td = document.createElement('td');
      td.textContent = content;
      tr.append(td);
    }
    el('rows').append(tr);
  }
}
el('run').onclick = () => {
  metric = 'wall_ms';
  el('run').classList.add('active');
  el('step').classList.remove('active');
  render();
};
el('step').onclick = () => {
  metric = 'step_wall_ms';
  el('step').classList.add('active');
  el('run').classList.remove('active');
  render();
};
el('download').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'measurements.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
el('manifest').textContent = JSON.stringify(data, null, 2);
render();

</script></html>`;
}

async function runCapture(python, args, log) {
  const output = createWriteStream(log);
  try {
    await new Promise((accept, reject) => {
      const child = spawn(python, args, { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
      child.stdout.pipe(output, { end: false });
      child.stderr.pipe(output, { end: false });
      child.on('error', reject);
      child.on('close', (code) => (code === 0 ? accept() : reject(new Error(`Capture exited ${code}; see ${log}`))));
    });
  } finally {
    await new Promise((accept) => output.end(accept));
  }
}

export async function benchmark(manifest, directory) {
  if (!Array.isArray(manifest.variants) || !manifest.variants.length) throw new Error('Specify variants.');
  const ids = new Set();
  for (const variant of manifest.variants) {
    if (!/^[a-z0-9-]+$/.test(variant.id) || ids.has(variant.id))
      throw new Error('Variant IDs must be unique simple names.');
    ids.add(variant.id);
    if (!variant.model || !Array.isArray(variant.args ?? [])) throw new Error('Specify a model and argument array.');
  }
  const rounds = manifest.rounds ?? 3;
  if (!Number.isInteger(rounds) || rounds < 1) throw new Error('rounds must be a positive integer.');
  for (const arg of [...(manifest.args ?? []), ...manifest.variants.flatMap((v) => v.args ?? [])]) {
    if (typeof arg !== 'string' || /^(--output-dir|--no-profile|--validation-output)(=|$)/.test(arg))
      throw new Error('Capture arguments cannot override output, profiling or validation.');
  }
  await mkdir(directory); // A fresh directory avoids silently mixing old and new measurements.
  await writeFile(resolve(directory, 'manifest.json'), JSON.stringify(manifest, null, 2));
  const script = fileURLToPath(new URL('./profile_decoder.py', import.meta.url));
  const captures = [];
  for (let round = 0; round < rounds; round++) {
    const order = [...manifest.variants, ...manifest.variants.toReversed()];
    for (const [position, variant] of order.entries()) {
      const out = resolve(directory, `r${round}-${position}-${variant.id}`);
      await runCapture(
        manifest.python ?? 'python',
        [script, variant.model, ...(manifest.args ?? []), ...(variant.args ?? []), '--no-profile', '--output-dir', out],
        `${out}.log`,
      );
      const capture = JSON.parse(await readFile(resolve(out, 'capture.json'), 'utf8'));
      captures.push({ id: variant.id, label: variant.label ?? variant.id, round, capture });
      console.log(`Completed round ${round + 1}/${rounds}: ${variant.id} (${position + 1}/${order.length})`);
    }
  }
  const summary = summarize(captures);
  await writeFile(resolve(directory, 'measurements.json'), JSON.stringify({ manifest, summary, captures }, null, 2));
  await writeFile(resolve(directory, 'index.html'), renderReport(summary, manifest));
  return summary;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.length !== 4)
    throw new Error('Usage: node benchmark_decoder.mjs <manifest.json> <new-output-directory>');
  const manifest = JSON.parse(await readFile(process.argv[2], 'utf8'));
  const output = resolve(process.argv[3]);
  await mkdir(dirname(output), { recursive: true });
  console.log(JSON.stringify(await benchmark(manifest, output), null, 2));
}
