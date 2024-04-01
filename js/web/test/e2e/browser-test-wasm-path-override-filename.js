// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebAssembly backend (path override filename)', async function() {
  // check base URL from karma args
  if (typeof __ort_arg_base === 'undefined') {
    throw new Error('karma flag --base-url=<BASE_URL> is required');
  }

  // disable SIMD and multi-thread
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = false;

  // override .wasm file path for 'ort-wasm.wasm'
  const overrideUrl = new URL('./test-wasm-path-override/renamed.wasm', __ort_arg_base).href;
  console.log(`ort.env.wasm.wasmPaths['ort-wasm.wasm'] = ${JSON.stringify(overrideUrl)};`);
  ort.env.wasm.wasmPaths = {'ort-wasm.wasm': overrideUrl};

  await testFunction(ort, {executionProviders: ['wasm']});
});
