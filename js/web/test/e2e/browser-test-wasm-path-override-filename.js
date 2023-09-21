// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebAssembly backend (path override filename)', async function() {
  // disable SIMD and multi-thread
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = false;

  // override .wasm file path for 'ort-wasm.wasm'
  ort.env.wasm.wasmPaths = {'ort-wasm.wasm': new URL('./test-wasm-path-override/renamed.wasm', document.baseURI).href};

  await testFunction(ort, {executionProviders: ['wasm']});
});
