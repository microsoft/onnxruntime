// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebAssembly backend (path override prefix)', async function () {
  // check base URL port from test args
  if (typeof __ort_arg_port === 'undefined') {
    throw new Error('test flag --port=<PORT> is required');
  }
  const base = `http://localhost:${__ort_arg_port}/`;

  // override .wasm file path prefix
  const prefix = new URL('./test-wasm-path-override/', base).href;
  console.log(`ort.env.wasm.wasmPaths = ${JSON.stringify(prefix)};`);
  ort.env.wasm.wasmPaths = prefix;

  await testFunction(ort, { executionProviders: ['wasm'] });
});
