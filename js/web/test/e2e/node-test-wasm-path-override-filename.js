// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');
const ort = require('onnxruntime-web');
const testFunction = require('./common');
const { pathToFileURL } = require('url');

it('Node.js E2E testing - WebAssembly backend (path override filename)', async function () {
  // override .wasm file path for 'ort-wasm.wasm'
  ort.env.wasm.wasmPaths = {
    mjs: pathToFileURL(path.join(__dirname, 'test-wasm-path-override/renamed.mjs')),
    wasm: pathToFileURL(path.join(__dirname, 'test-wasm-path-override/renamed.wasm')),
  };

  await testFunction(ort, { executionProviders: ['wasm'] });
});
