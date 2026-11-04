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
import { buildReport, createEventIndex, laneOf, mergedDuration, renderReport } from './visualize_profile.mjs';

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
  assert.equal(laneOf(event('WebGPU::Finish', 0, 1)), 5);
});

test('links exact submissions to all their GPU dispatches across run boundaries', () => {
  const dispatch = (ts, id) => ({
    ...gpu(ts, 2, 'same&kernel'),
    args: { clock_alignment: 'd3d12_clock_calibration', submission_id: id },
  });
  const report = buildReport([
    event('model_run', 100, 50),
    event('model_run', 200, 50),
    event('WebGPU::Submit', 90, 5, { submission_id: '7:1' }),
    event('WebGPU::Submit', 105, 20, { submission_id: '7:2' }),
    dispatch(110, '7:2'),
    dispatch(110, '7:1'),
    dispatch(210, '7:1'),
  ]);
  assert.equal(report.submissionCount, 2);
  const groups = new Map(report.submissions.map((group) => [group.id, group]));
  for (const [id, count] of [
    ['7:1', 2],
    ['7:2', 1],
  ]) {
    const group = groups.get(id);
    assert.equal(report.events[group.submit][2], 6);
    assert.equal(group.dispatches.length, count);
    for (const index of [group.submit, ...group.dispatches]) {
      assert.equal(JSON.parse(report.strings[report.events[index][4]]).submission_id, id);
    }
  }
  assert.ok(!report.runs[0].indices.includes(groups.get('7:1').submit));
  const group = groups.get('7:2');
  assert.ok(report.events[group.dispatches[0]][0] < report.events[group.submit][0] + report.events[group.submit][1]);
});

test('legacy, missing and ambiguous submission metadata never produce guessed links', () => {
  assert.deepEqual(buildReport(trace).submissions, []);
  const report = buildReport([
    gpu(50, 1),
    event('WebGPU::Submit', 10, 1, { submission_id: 'duplicate' }),
    event('WebGPU::Submit', 20, 1, { submission_id: 'duplicate' }),
    event('WebGPU::Submit', 30, 1, { submission_id: 'copy-only' }),
    event('WebGPU::Submit', 40, 1, { submission_id: 42 }),
    ...['duplicate', 'missing', 42].map((id) => ({
      ...gpu(60, 1),
      args: { clock_alignment: 'first_submit', submission_id: id },
    })),
  ]);
  assert.equal(report.submissionCount, 0);
  assert.deepEqual(report.submissions, []);
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

test('baseline options compare nested values independently of object key order', () => {
  const options = { executionMode: 'parallel', threads: { intra: 4, inter: 2 }, providers: ['webgpu', 'cpu'] };
  const reordered = { providers: ['webgpu', 'cpu'], threads: { inter: 2, intra: 4 }, executionMode: 'parallel' };
  const metadata = { ...capture, session_options: options };
  assert.doesNotThrow(() => buildReport(trace, metadata, { ...metadata, session_options: reordered }));
  for (const changed of [
    { ...reordered, threads: { inter: 2, intra: 1 } },
    { ...reordered, providers: ['cpu', 'webgpu'] },
    { ...reordered, extra: true },
  ]) {
    assert.throws(() => buildReport(trace, metadata, { ...metadata, session_options: changed }), /session_options/);
  }
});

test('indexed summaries preserve crossing spans, nested runs and boundary points', () => {
  const source = [
    event('model_run', 10, 100),
    event('model_run', 20, 10),
    event('model_run', 25, 40),
    event('model_run', 30, 0),
    event('model_run', 110, 10),
    gpu(0, 40),
    gpu(10, 0),
    gpu(25, 60),
    gpu(30, 0),
    gpu(110, 0),
    gpu(120, 0),
    event('WebGPU::Encode', 5, 120),
  ];
  const report = buildReport(source);
  for (const run of [...report.runs, report.wholeTrace]) {
    const stop = run.start + run.duration;
    const excludeEnd = run.index !== -1 && run.duration > 0;
    const expected = report.events.flatMap(([ts, dur], id) => {
      const overlap = Math.min(stop, ts + dur) - Math.max(run.start, ts);
      return overlap < 0 || (overlap === 0 && (ts < run.start || (ts >= stop && excludeEnd))) ? [] : [id];
    });
    assert.deepEqual(run.indices, expected);
    assert.equal(
      run.busy,
      mergedDuration(
        expected
          .filter((id) => report.events[id][2] === 10)
          .map((id) => [
            Math.max(run.start, report.events[id][0]),
            Math.min(stop, report.events[id][0] + report.events[id][1]),
          ]),
      ),
    );
    assert.equal(run.cpu[5], Math.min(stop, 125) - Math.max(run.start, 5));
  }
});

test('index visits local events without rescanning preceding runs', () => {
  const count = 20000;
  let reads = 0;
  const events = new Proxy(
    Array.from({ length: count }, (_, id) => [id, 0.5]),
    {
      get(target, key) {
        if (/^\d+$/.test(String(key))) reads++;
        return Reflect.get(target, key);
      },
    },
  );
  const query = createEventIndex(events);
  reads = 0;
  for (let run = 0; run < 1000; run++) {
    assert.deepEqual(
      [...query(run * 20, run * 20 + 20, true)],
      Array.from({ length: 20 }, (_, id) => run * 20 + id),
    );
  }
  // Count accesses rather than wall time so slow CI hosts do not make this regression flaky.
  assert.ok(reads < count * 5, `Index read ${reads} events for ${count} matches.`);
});

test('standalone payload round trips safely and summary omits bulky event data', async () => {
  const root = await mkdtemp(join(tmpdir(), 'ort-webgpu-report-'));
  try {
    const source = join(root, 'trace.summary.json');
    const hostile = '</script><img src=x onerror=alert(1)>';
    await writeFile(
      source,
      JSON.stringify([
        event('model_run', 100, 100),
        event('WebGPU::Submit', 110, 1, { submission_id: hostile }),
        { ...gpu(120, 10, hostile), args: { clock_alignment: 'first_submit', submission_id: hostile } },
      ]),
    );
    const result = await renderReport(source, join(root, 'report.html'));
    const page = await readFile(result.output, 'utf8');
    assert.ok(!page.includes(hostile));
    assert.ok(!page.includes('REPORT_PAYLOAD'));
    assert.ok(!page.includes('/* REPORT_CLIENT */'));
    const payload = page.match(/type="application\/octet-stream">\s*([A-Za-z0-9+/=\s]+)<\/script>/)[1];
    const report = JSON.parse(gunzipSync(Buffer.from(payload, 'base64')));
    assert.ok(report.strings.includes(hostile));
    assert.equal(report.events.length, 3);
    assert.equal(report.submissions[0].id, hostile);
    assert.equal(report.submissions[0].dispatches.length, 1);
    const summary = JSON.parse(await readFile(result.summary, 'utf8'));
    assert.equal(summary.events, undefined);
    assert.equal(summary.strings, undefined);
    assert.equal(summary.runs[0].indices, undefined);
    assert.equal(summary.wholeTrace.indices, undefined);
    assert.equal(summary.submissions, undefined);
    assert.equal(summary.submissionCount, 1);
    assert.equal(report.title, 'trace.summary.json');
    await assert.rejects(renderReport(source, source), /overwrite/);
    if (process.platform === 'win32') await assert.rejects(renderReport(source, source.toUpperCase()), /overwrite/);
    await assert.rejects(renderReport(source, join(root, 'trace.html')), /overwrite/);
    assert.equal(JSON.parse(await readFile(source, 'utf8')).length, 3);
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
