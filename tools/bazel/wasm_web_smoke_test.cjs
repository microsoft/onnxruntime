// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

async function main() {
  const output = path.resolve(process.argv[2]);
  const ort = require(path.join(output, 'ort-web.js'));
  ort.env.wasm.wasmBinary = fs.readFileSync(path.join(output, 'ort_wasm.wasm'));
  for (const [name, shape, values] of [
    ['mul_1', [3, 2], [1, 4, 9, 16, 25, 36]],
    ['matmul_1', [3, 1], [5, 11, 17]],
  ]) {
    const model = fs.readFileSync(`onnxruntime/test/testdata/${name}.onnx`);
    const session = await ort.InferenceSession.create(model, {executionProviders: ['wasm']});
    try {
      const input = new ort.Tensor('float32', Float32Array.from([1, 2, 3, 4, 5, 6]), [3, 2]);
      const {Y} = await session.run({X: input});
      assert.deepEqual(Y.dims, shape);
      assert.deepEqual([...Y.data], values);
    } finally {
      await session.release();
    }
  }
  await assert.rejects(
      ort.InferenceSession.create(Uint8Array.from([0]), {executionProviders: ['wasm']}),
      /session|model|protobuf/i);
  // Every declared inference WASM API must remain exported, not just the smoke path.
  const {default: factory} = await import(pathToFileURL(path.join(output, 'ort_wasm.mjs')).href);
  const wasm = await factory({wasmBinary: fs.readFileSync(path.join(output, 'ort_wasm.wasm'))});
  const fullHeader = fs.readFileSync('onnxruntime/wasm/api.h', 'utf8');
  const names = [...fullHeader.matchAll(/EMSCRIPTEN_KEEPALIVE\s+(Ort\w+)\s*\(/g)]
      .map((match) => match[1]).filter((name) => !name.startsWith('OrtTraining'));
  assert(names.length > 20, 'Expected the checked-in inference API declarations to be discoverable.');
  for (const name of names) {
    // GPU-only entrypoints are guarded by USE_WEBGPU in api.h.
    if (!name.includes('WebGpu')) {
      assert.equal(typeof wasm[`_${name}`], 'function', `Missing inference WASM export ${name}.`);
    }
  }
  console.log('Source-built WASM/frontend Mul and MatMul inference passed, including shape and every output value.');
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
