// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { capture, loadInputs, summarizeOutputs, tensorData, validateOptions } from './profile_model.mjs';
import { buildReport } from './visualize_profile.mjs';

class Tensor {
  constructor(type, data, dims) {
    Object.assign(this, { type, data, dims, disposed: false });
  }
  dispose() {
    this.disposed = true;
  }
}

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'ort-webgpu-capture-'));
  const model = join(root, 'abs.onnx');
  const inputs = join(root, 'inputs.json');
  // ONNX backend test_abs, also used by external_dawn/main.cc: float32 x/y [3,4,5].
  await writeFile(
    model,
    Buffer.from([
      8, 7, 18, 12, 98, 97, 99, 107, 101, 110, 100, 45, 116, 101, 115, 116, 58, 73, 10, 11, 10, 1, 120, 18, 1, 121, 34,
      3, 65, 98, 115, 18, 8, 116, 101, 115, 116, 95, 97, 98, 115, 90, 23, 10, 1, 120, 18, 18, 10, 16, 8, 1, 18, 12, 10,
      2, 8, 3, 10, 2, 8, 4, 10, 2, 8, 5, 98, 23, 10, 1, 121, 18, 18, 10, 16, 8, 1, 18, 12, 10, 2, 8, 3, 10, 2, 8, 4, 10,
      2, 8, 5, 66, 4, 10, 0, 16, 13,
    ]),
  );
  await writeFile(inputs, JSON.stringify({ x: { type: 'float32', dims: [3, 4, 5], data: Array(60).fill(-2) } }));
  return { root, model, inputs, backend: 'D3D12', warmup: 1, repeat: 2, intraOpThreads: 1, ortModule: '/mock' };
}

test('typed inputs retain numeric values and reject shape/type/range mistakes', () => {
  assert.deepEqual(
    tensorData({ type: 'int64', dims: [2], data: ['9223372036854775807', -2] }),
    new BigInt64Array([9223372036854775807n, -2n]),
  );
  assert.equal(tensorData({ type: 'float32', dims: [], data: [1.5] })[0], 1.5);
  for (const spec of [
    { type: 'missing', dims: [], data: [1] },
    { type: 'float32', dims: [-1], data: [] },
    { type: 'float32', dims: [2], data: [1] },
    { type: 'uint8', dims: [1], data: [256] },
    { type: 'int32', dims: [1], data: [1.5] },
    { type: 'bool', dims: [1], data: [2] },
    { type: 'int64', dims: [1], data: ['9223372036854775808'] },
    { type: 'uint64', dims: [1], data: [-1] },
    { type: 'float32', dims: [1], data: [1e100] },
    { type: 'float32', dims: [1], data: [NaN] },
  ])
    assert.throws(() => tensorData(spec));
});

test('binary input is aligned, length checked, and hashed by content', async () => {
  const options = await fixture();
  try {
    const bytes = Buffer.from(new Float32Array([-2, 3]).buffer);
    await writeFile(join(options.root, 'data.bin'), bytes);
    await writeFile(options.inputs, JSON.stringify({ x: { type: 'float32', dims: [2], data_file: 'data.bin' } }));
    const binary = await loadInputs(options.inputs, Tensor);
    assert.deepEqual(binary.feeds.x.data, new Float32Array([-2, 3]));
    await writeFile(options.inputs, JSON.stringify({ x: { type: 'float32', dims: [2], data: [-2, 3] } }));
    const inline = await loadInputs(options.inputs, Tensor);
    assert.equal(inline.inputsSha256, binary.inputsSha256);
    await writeFile(options.inputs, JSON.stringify({ x: { type: 'float32', dims: [2], data: [-2, 4] } }));
    assert.notEqual((await loadInputs(options.inputs, Tensor)).inputsSha256, inline.inputsSha256);
    assert.throws(() => tensorData({ type: 'float32', dims: [2] }, bytes.subarray(1)), /byte length/);
    assert.throws(() => tensorData({ type: 'float32', dims: [2], data: [1, 2] }, bytes), /not both/);
  } finally {
    await rm(options.root, { recursive: true, force: true });
  }
});

test('output summaries preserve signed zeros and reject nonfinite float32/float16', () => {
  const a = summarizeOutputs({ y: new Tensor('float32', new Float32Array([0]), [1]) });
  const b = summarizeOutputs({ y: new Tensor('float32', new Float32Array([-0]), [1]) });
  assert.notEqual(a.y.sha256, b.y.sha256);
  for (const [type, data] of [
    ['float32', new Float32Array([Infinity])],
    ['float16', new Uint16Array([0x7e00])],
  ])
    assert.throws(() => summarizeOutputs({ y: new Tensor(type, data, [1]) }), /Non-finite/);
});

function mockRuntime(state, fail = false) {
  return {
    Tensor,
    env: { versions: { node: 'mock' } },
    listSupportedBackends: () => [{ name: 'webgpu', bundled: true }],
    InferenceSession: {
      create: async (model, options) => {
        state.options = options;
        return {
          inputNames: ['x'],
          run: async (feeds) => {
            state.feeds = feeds;
            if (fail) throw new Error('inference failed');
            const value = new Tensor('float32', Float32Array.from(feeds.x.data, Math.abs), feeds.x.dims);
            (state.outputs ??= []).push(value);
            return { y: value };
          },
          endProfiling: () => {
            state.ended = true;
          },
          release: async () => {
            state.released = true;
          },
        };
      },
    },
  };
}

test('disabled capture avoids profiler calls, checks outputs and disposes resources', async () => {
  const options = await fixture();
  const state = {};
  try {
    const result = await capture(
      { ...options, outputDir: join(options.root, 'off'), noProfile: true },
      mockRuntime(state),
    );
    assert.equal(state.options.enableProfiling, false);
    assert.equal(state.options.profileFilePrefix, undefined);
    assert.equal(state.options.extra['session.disable_cpu_ep_fallback'], '1');
    assert.equal(state.options.extra['ep.webgpuexecutionprovider.dawnBackendType'], 'D3D12');
    assert.equal(state.ended, undefined);
    assert.equal(result.trace, null);
    assert.deepEqual(
      result.runs.map((run) => run.warmup),
      [true, false, false],
    );
    assert.deepEqual(result.outputs.y.sample, Array(8).fill(2));
    assert.equal(result.session_options.enableProfiling, undefined);
    assert.equal(state.released, true);
    assert.ok(state.outputs.every((tensor) => tensor.disposed));
    assert.equal(state.feeds.x.disposed, true);
    assert.deepEqual(await readdir(join(options.root, 'off')), ['capture.json']);
    await assert.rejects(capture({ ...options, outputDir: join(options.root, 'off') }, mockRuntime({})), /EEXIST/);
  } finally {
    await rm(options.root, { recursive: true, force: true });
  }
});

test('failure releases inputs/session; invalid counts and CPU-only runtimes fail explicitly', async () => {
  const options = await fixture();
  const state = {};
  try {
    await assert.rejects(
      capture({ ...options, outputDir: join(options.root, 'failure') }, mockRuntime(state, true)),
      /inference failed/,
    );
    assert.equal(state.released, true);
    assert.equal(state.feeds.x.disposed, true);
    for (const extra of [{ warmup: -1 }, { repeat: 0 }, { repeat: 1.5 }, { intraOpThreads: -1 }, { backend: 'CPU' }])
      assert.throws(() => validateOptions({ ...options, ...extra }));
    await assert.rejects(capture(options, { listSupportedBackends: () => [{ name: 'cpu' }] }), /native WebGPU/);
  } finally {
    await rm(options.root, { recursive: true, force: true });
  }
});

test(
  'native WebGPU off/on/off captures match outputs and produce aligned CPU/GPU events',
  {
    skip: !process.env.ORT_WEBGPU_NODE_MODULE && 'Set ORT_WEBGPU_NODE_MODULE to a WebGPU-enabled native Node package.',
  },
  async () => {
    const options = await fixture();
    try {
      const ortModule = resolve(process.env.ORT_WEBGPU_NODE_MODULE);
      const ort = createRequire(import.meta.url)(ortModule);
      const baseline = await capture(
        { ...options, ortModule, outputDir: join(options.root, 'off'), noProfile: true },
        ort,
      );
      const profiled = await capture({ ...options, ortModule, outputDir: join(options.root, 'on') }, ort);
      const after = await capture(
        { ...options, ortModule, outputDir: join(options.root, 'off-again'), noProfile: true },
        ort,
      );
      assert.deepEqual(profiled.outputs, baseline.outputs);
      assert.deepEqual(after.outputs, baseline.outputs);
      assert.equal(after.trace, null);
      const report = buildReport(JSON.parse(await readFile(profiled.trace, 'utf8')), profiled, baseline);
      assert.equal(report.runs.length, options.warmup + options.repeat);
      assert.ok(report.gpuEvents > 0);
      assert.ok(report.events.some((event) => event[2] === 6));
      assert.ok(report.events.some((event) => event[2] === 7));
      if (process.platform === 'win32') assert.deepEqual(report.alignment, ['d3d12_clock_calibration']);
      assert.ok(report.baseline.inference >= 0);
    } finally {
      await rm(options.root, { recursive: true, force: true });
    }
  },
);
