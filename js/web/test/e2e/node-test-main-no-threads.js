// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const ort = require('onnxruntime-web');
const testFunction = require('./common');

it('Browser E2E testing - WebAssembly backend', async function () {
  ort.env.wasm.numThreads = 1;
  await testFunction(ort, { executionProviders: ['wasm'] });
});
