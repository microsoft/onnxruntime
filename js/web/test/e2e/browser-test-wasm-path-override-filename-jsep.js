// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebAssembly backend (path override filename)', async function () {
  // check base URL port from test args
  if (typeof __ort_arg_port === 'undefined') {
    throw new Error('test flag --port=<PORT> is required');
  }
  const base = `http://localhost:${__ort_arg_port}/`;

  ort.env.wasm.wasmPaths = {};

  if (typeof __ort_arg_files === 'string' && __ort_arg_files.includes('wasm')) {
    const overrideWasmUrl = new URL('./test-wasm-path-override/jsep-renamed.wasm', base).href;
    console.log(`ort.env.wasm.wasmPaths['wasm'] = ${JSON.stringify(overrideWasmUrl)};`);
    ort.env.wasm.wasmPaths.wasm = overrideWasmUrl;
  }

  if (typeof __ort_arg_files === 'string' && __ort_arg_files.includes('mjs')) {
    const overrideMjsUrl = new URL('./test-wasm-path-override/jsep-renamed.mjs', base).href;
    console.log(`ort.env.wasm.wasmPaths['mjs'] = ${JSON.stringify(overrideMjsUrl)};`);
    ort.env.wasm.wasmPaths.mjs = overrideMjsUrl;
  }

  await testFunction(ort, { executionProviders: ['wasm'] });
});
