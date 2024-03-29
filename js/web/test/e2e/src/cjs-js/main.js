// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const ort = require('onnxruntime-web');
const {testInferenceAndValidate} = require('./shared');

it('Browser package consuming test - [.js][commonjs]', async function() {
  ort.env.wasm.numThreads = 1;
  await testInferenceAndValidate(ort, {executionProviders: ['wasm']});
});
