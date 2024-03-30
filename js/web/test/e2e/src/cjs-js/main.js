// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const ort = require('onnxruntime-web/wasm');
const {setupMultipleThreads, testInferenceAndValidate} = require('./shared');

it('Browser package consuming test - [.js][commonjs]', async function() {
  setupMultipleThreads(ort);
  await testInferenceAndValidate(ort, {executionProviders: ['wasm']});
});
