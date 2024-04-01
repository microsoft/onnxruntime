// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebAssembly backend (path override prefix)', async function() {
  // check base URL from karma args
  if (typeof __ort_arg_base === 'undefined') {
    throw new Error('karma flag --base-url=<BASE_URL> is required');
  }

  // disable SIMD and multi-thread
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = false;

  // override .wasm file path prefix
  const prefix = new URL('./test-wasm-path-override/', __ort_arg_base).href;
  console.log(`ort.env.wasm.wasmPaths = ${JSON.stringify(prefix)};`);
  ort.env.wasm.wasmPaths = prefix;

  await testFunction(ort, {executionProviders: ['wasm']});
});
