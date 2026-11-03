// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { buildReport, laneOf, mergedDuration, renderReport } from './visualize_profile.mjs';

const event = (name, ts, dur, args = {}) => ({ name, ts, dur, args, ph: 'X', pid: 7, tid: 8 });
const gpu = (ts, dur, name = 'node&MatMul') => event(name, ts, dur, { clock_alignment: 'first_submit' });
const trace = [
  event('model_run', 100, 100),
  gpu(90, 30),
  gpu(110, 30),
  gpu(170, 50),
  event('WebGPU::Encode', 105, 15, { execution_side: 'cpu' }),
];
const capture = {
  model_sha256: 'hash',
  ort_version: '1.32',
  backend: 'D3D12',
  workload: 'test',
  runs: [{ stage: 'inference', duration: -1, start: -1 }],
};

test('clips GPU intervals to a run and counts overlap only once', () => {
  assert.equal(
    mergedDuration([
      [20, 40],
      [10, 30],
      [40, 50],
      [70, 80],
    ]),
    50,
  );
  const report = buildReport(trace, capture);
  assert.equal(report.runs[0].busy, 70);
  assert.equal(report.runs[0].gap, 30);
  assert.equal(report.runs[0].cpu[5], 15);
  assert.equal(report.runs[0].programs[0].us, 80);
  assert.equal(report.runs[0].start, 100);
  assert.equal(report.runs[0].duration, 100);
  assert.deepEqual(report.alignment, ['first_submit']);
  assert.equal(report.events[0][0], 90);
});

test('distinguishes CPU preparations, fallbacks and copies', () => {
  const node = (op, provider) => ({ name: 'node_kernel_time', cat: 'Node', args: { op_name: op, provider } });
  assert.equal(laneOf(node('Abs', 'WebGpuExecutionProvider')), 2);
  assert.equal(laneOf(node('Abs', 'CPUExecutionProvider')), 3);
  assert.equal(laneOf(node('MemcpyFromHost', 'WebGpuExecutionProvider')), 4);
});

test('accepts traceEvents objects and rejects invalid traces or metadata counts', () => {
  assert.equal(buildReport({ traceEvents: trace }).runs.length, 1);
  assert.throws(() => buildReport(null), /event array/);
  assert.throws(() => buildReport([]), /No WebGPU/);
  assert.throws(() => buildReport(trace, { runs: [{}, {}] }), /run count/);
  assert.throws(() => buildReport(trace, { runs: {} }), /array/);
  assert.equal(buildReport([...trace, gpu(NaN, 1), gpu(150, -1)]).gpuEvents, 3);
});

test('handles large GPU-only captures and zero-duration timestamps', () => {
  const report = buildReport(Array.from({ length: 150000 }, (_, index) => gpu(index, 1)));
  assert.equal(report.syntheticRun, true);
  assert.equal(report.runs[0].duration, 150000);
  assert.equal(report.runs[0].busy, 150000);
  const zero = buildReport([gpu(42, 0)]);
  assert.equal(zero.runs[0].duration, 0);
  assert.equal(zero.runs[0].indices.length, 1);
});

test('matches unprofiled workload and excludes warmup from baseline means', () => {
  const baseline = {
    ...capture,
    trace: null,
    runs: [
      { stage: 'inference', warmup: true, wall_ms: 100 },
      { stage: 'inference', wall_ms: 10 },
      { stage: 'inference', wall_ms: 20 },
    ],
  };
  assert.equal(buildReport(trace, capture, baseline).baseline.inference, 15);
  assert.throws(() => buildReport(trace, capture, { ...baseline, backend: 'Vulkan' }), /backend/);
  assert.throws(() => buildReport(trace, capture, { ...baseline, trace: 'profile.json' }), /unprofiled/);
  assert.throws(() => buildReport(trace, {}, baseline), /model_sha256/);
  for (const [key, value] of Object.entries({
    capture_schema: 2,
    inputs_sha256: 'changed',
    session_options: { intraOpNumThreads: 1 },
  })) {
    assert.throws(() => buildReport(trace, { ...capture, [key]: value }, baseline), new RegExp(key));
    assert.equal(
      buildReport(trace, { ...capture, [key]: value }, { ...baseline, [key]: value }).baseline.inference,
      15,
    );
  }
});

test('standalone payload round trips safely and summary omits bulky event data', async () => {
  const root = await mkdtemp(join(tmpdir(), 'ort-webgpu-report-'));
  try {
    const source = join(root, 'trace.summary.json');
    const hostile = '</script><img src=x onerror=alert(1)>';
    await writeFile(source, JSON.stringify([event('model_run', 100, 100), gpu(120, 10, hostile)]));
    const result = await renderReport(source, join(root, 'report.html'));
    const page = await readFile(result.output, 'utf8');
    assert.ok(!page.includes(hostile));
    assert.ok(!page.includes('REPORT_PAYLOAD'));
    assert.ok(!page.includes('/* REPORT_CLIENT */'));
    const payload = page.match(/type="application\/octet-stream">\s*([A-Za-z0-9+/=\s]+)<\/script>/)[1];
    const report = JSON.parse(gunzipSync(Buffer.from(payload, 'base64')));
    assert.ok(report.strings.includes(hostile));
    assert.equal(report.events.length, 2);
    const summary = JSON.parse(await readFile(result.summary, 'utf8'));
    assert.equal(summary.events, undefined);
    assert.equal(summary.strings, undefined);
    assert.equal(summary.runs[0].indices, undefined);
    assert.equal(summary.wholeTrace.indices, undefined);
    assert.equal(report.title, 'trace.summary.json');
    await assert.rejects(renderReport(source, source), /overwrite/);
    if (process.platform === 'win32') await assert.rejects(renderReport(source, source.toUpperCase()), /overwrite/);
    await assert.rejects(renderReport(source, join(root, 'trace.html')), /overwrite/);
    assert.equal(JSON.parse(await readFile(source, 'utf8')).length, 2);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('raw application traces retain gaps and events outside individual inference runs', () => {
  const raw = [
    event('model_run', 100, 50),
    gpu(110, 10),
    event('WebGPU::Wait', 250, 10, { execution_side: 'cpu' }),
    event('model_run', 300, 50),
    gpu(320, 10),
    gpu(370, 10),
  ];
  const report = buildReport(raw);
  assert.deepEqual(
    report.runs.map((run) => [run.start, run.duration, run.stage]),
    [
      [100, 50, 'inference'],
      [300, 50, 'inference'],
    ],
  );
  assert.equal(report.wholeTrace.start, 100);
  assert.equal(report.wholeTrace.duration, 280);
  assert.equal(report.wholeTrace.busy, 30);
  assert.equal(report.wholeTrace.gap, 250);
  assert.equal(report.wholeTrace.cpu[7], 10);
  assert.equal(report.wholeTrace.indices.length, raw.length);
  const withPoint = buildReport([...raw, gpu(380, 0)]);
  assert.equal(withPoint.wholeTrace.indices.length, raw.length + 1);
  assert.equal(
    report.runs.reduce((total, run) => total + run.busy, 0),
    20,
  );
});

test('CLI processes a saved profiling log without metadata, models or runtime dependencies', async () => {
  const root = await mkdtemp(join(tmpdir(), 'ort-webgpu-raw-log-'));
  try {
    const source = join(root, 'application-profile.json');
    const output = join(root, 'timeline.html');
    await writeFile(source, JSON.stringify({ traceEvents: trace }));
    const stdout = execFileSync(
      process.execPath,
      [fileURLToPath(new URL('./visualize_profile.mjs', import.meta.url)), source, '--output', output],
      { cwd: root, env: { ...process.env, PATH: dirname(process.execPath) }, encoding: 'utf8', windowsHide: true },
    );
    assert.equal(JSON.parse(stdout).gpuEvents, 3);
    const summary = JSON.parse(await readFile(join(root, 'timeline.summary.json'), 'utf8'));
    assert.equal(summary.title, 'application-profile.json');
    assert.equal(summary.runs.length, 1);
    assert.equal(summary.wholeTrace.start, 90);
    assert.equal(summary.wholeTrace.duration, 130);
    assert.ok((await readFile(output, 'utf8')).includes('id="whole-trace"'));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
