// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebAssembly backend', async function() {
  if (typeof __ort_arg_num_threads === 'undefined') {
    window.__ort_arg_num_threads = '1';
  }
  const numThreads = parseInt(__ort_arg_num_threads);
  console.log(`numThreads = ${numThreads}`);
  ort.env.wasm.numThreads = numThreads;

  if (typeof __ort_arg_proxy === 'undefined') {
    window.__ort_arg_proxy = '0';
  }
  const proxy = __ort_arg_proxy === '1';
  console.log(`proxy = ${proxy}`);
  ort.env.wasm.proxy = proxy;

  await testFunction(ort, {executionProviders: ['wasm']});
});
