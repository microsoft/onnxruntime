// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const ort = require('onnxruntime-web/wasm');
const { setupMultipleThreads, testInferenceAndValidate } = require('./shared');

if (typeof SharedArrayBuffer === 'undefined') {
  it('Browser package consuming test - single-thread - [js][commonjs]', async function () {
    await testInferenceAndValidate(ort, { executionProviders: ['wasm'] });
  });
} else {
  it('Browser package consuming test - multi-thread - [js][commonjs]', async function () {
    setupMultipleThreads(ort);
    await testInferenceAndValidate(ort, { executionProviders: ['wasm'] });
  });
}
