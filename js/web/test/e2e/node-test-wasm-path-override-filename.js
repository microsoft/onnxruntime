// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');
const ort = require('onnxruntime-web');
const testFunction = require('./common');

it('Node.js E2E testing - WebAssembly backend (path override filename)', async function () {
  // disable SIMD and multi-thread
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = false;

  // override .wasm file path for 'ort-wasm.wasm'
  ort.env.wasm.wasmPaths = {
    'ort-wasm.wasm': path.join(__dirname, 'test-wasm-path-override/renamed.wasm')
  };

  await testFunction(ort, { executionProviders: ['wasm'] });
});
