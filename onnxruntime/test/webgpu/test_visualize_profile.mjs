// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
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
  assert.throws(() => buildReport(trace, capture, { ...baseline, trace: 'profile.json' }), /no-profile/);
  assert.throws(() => buildReport(trace, {}, baseline), /model_sha256/);
  for (const [key, value] of Object.entries({
    capture_schema: 2,
    inputs_sha256: 'changed',
    session_options: { intraOpNumThreads: 1 },
    ort_module: '/build/node',
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
    await assert.rejects(renderReport(source, source), /overwrite/);
    if (process.platform === 'win32') await assert.rejects(renderReport(source, source.toUpperCase()), /overwrite/);
    await assert.rejects(renderReport(source, join(root, 'trace.html')), /overwrite/);
    assert.equal(JSON.parse(await readFile(source, 'utf8')).length, 2);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
