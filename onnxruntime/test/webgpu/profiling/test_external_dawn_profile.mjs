// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const binary = process.argv[2];
assert.ok(binary, 'Usage: node test_external_dawn_profile.mjs <external-Dawn test executable> [output directory]');
const executable = resolve(binary);
const output = resolve(process.argv[3] ?? 'build/external-dawn-profile-tests');
mkdirSync(output, { recursive: true });
const hash = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const results = [];
const modes = [
  ['fallback', [], false, 0],
  ['calibrated', ['--clock_calibration'], true, 2],
  ['failed-callback', ['--clock_calibration', '--calibration_failure'], false, 2],
  ['invalid-sample', ['--clock_calibration', '--calibration_invalid'], false, 2],
];

for (const [mode, flags, calibrated, callbackCalls] of modes) {
  for (const hostDevice of [false, true]) {
    const name = `${mode}-${hostDevice ? 'host-device' : 'ort-device'}`;
    const args = ['--profiling', ...flags, ...(hostDevice ? ['--host_device'] : [])];
    const child = spawnSync(executable, args, {
      cwd: dirname(executable),
      encoding: 'utf8',
      windowsHide: true,
      timeout: 30000,
    });
    writeFileSync(join(output, `${name}.stdout.log`), child.stdout ?? '');
    writeFileSync(join(output, `${name}.stderr.log`), child.stderr ?? '');
    assert.ifError(child.error);
    assert.equal(child.status, 0, `${name}: ${child.stderr}`);
    assert.equal(child.stdout.split('Profiling off: zero additional').length - 1, 2, name);
    assert.ok(child.stdout.includes(`Calibration: calls=${callbackCalls},`), name);
    if (callbackCalls) assert.ok(child.stdout.includes('Invalid and conflicting calibration options rejected.'), name);
    const traceName = child.stdout.match(/trace=([^\r\n]+)/)?.[1];
    assert.ok(traceName, `${name}: missing trace`);
    const raw = readFileSync(resolve(dirname(executable), traceName));
    writeFileSync(join(output, `${name}.trace.json`), raw);
    const trace = JSON.parse(raw);
    const gpu = trace.filter((event) => typeof event.args?.clock_alignment === 'string').sort((a, b) => a.ts - b.ts);
    const runs = trace.filter((event) => event.name === 'model_run').sort((a, b) => a.ts - b.ts);
    assert.equal(runs.length, 2, name);
    assert.equal(gpu.length, 2, name);
    const alignment = calibrated ? 'd3d12_clock_calibration' : 'first_submit';
    const submits = trace.filter((event) => event.name === 'WebGPU::Submit');
    const submissions = new Map();
    for (const span of submits) {
      assert.equal(typeof span.args.submission_id, 'string', name);
      assert.ok(span.args.submission_id.length > 0, name);
      assert.ok(!submissions.has(span.args.submission_id), `${name}: duplicate submission ID`);
      submissions.set(span.args.submission_id, span);
    }
    for (const event of gpu) {
      assert.equal(event.args.clock_alignment, alignment, name);
      assert.ok(submissions.has(event.args.submission_id), `${name}: GPU dispatch missing its CPU submit`);
    }
    assert.equal(new Set(gpu.map((event) => event.args.submission_id)).size, 2, name);
    for (const span of ['StartProfiling', 'Encode', 'Finish', 'Submit', 'Wait', 'ProfileReadback']) {
      assert.ok(
        trace.some((event) => event.name === `WebGPU::${span}`),
        `${name}: missing ${span}`,
      );
    }
    const queueDelayMs = [];
    if (calibrated) {
      for (let i = 0; i < runs.length; ++i) {
        const run = runs[i];
        const event = gpu[i];
        assert.ok(event.ts >= run.ts - 1000 && event.ts + event.dur <= run.ts + run.dur + 1000, name);
        const submit = submissions.get(event.args.submission_id);
        assert.ok(submit.ts >= run.ts && submit.ts <= run.ts + run.dur, name);
        const delay = event.ts - submit.ts;
        assert.ok(delay >= 15000, `${name}: calibration hid the delayed queue (${delay} us)`);
        queueDelayMs.push(delay / 1000);
      }
    }
    results.push({ name, args, exitCode: child.status, alignment, callbackCalls, queueDelayMs });
    console.log(`${name}: passed${calibrated ? `; queue delays ${queueDelayMs.join(', ')} ms` : ''}`);
  }
}

const missing = spawnSync(executable, ['--no_proc_table'], {
  cwd: dirname(executable),
  encoding: 'utf8',
  windowsHide: true,
  timeout: 30000,
});
assert.ifError(missing.error);
assert.equal(missing.status, 0, missing.stderr);
assert.ok(missing.stdout.includes('exception is thrown as expected'));
writeFileSync(join(output, 'no-proc-table.stdout.log'), missing.stdout);
writeFileSync(join(output, 'no-proc-table.stderr.log'), missing.stderr);
results.push({ name: 'no-proc-table', exitCode: missing.status });
writeFileSync(
  join(output, 'results.json'),
  `${JSON.stringify({ executable, executableSha256: hash(executable), results }, null, 2)}\n`,
);
console.log(`${results.length} external-Dawn profiling checks passed.`);
