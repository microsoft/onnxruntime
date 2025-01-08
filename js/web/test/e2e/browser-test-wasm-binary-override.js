// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const documentUrl = document.currentScript.src;

it('Browser E2E testing - WebAssembly backend', async function () {
  // preload .wasm file binary
  const wasmUrl = new URL('./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm', documentUrl).href;
  const response = await fetch(wasmUrl);

  // make sure the .wasm file is loaded successfully
  assert(response.ok);
  assert(response.headers.get('Content-Type') === 'application/wasm');

  // override wasm binary
  const binary = await response.arrayBuffer();
  ort.env.wasm.wasmBinary = binary;

  await testFunction(ort, { executionProviders: ['wasm'] });
});
