// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

describe('Browser E2E testing for training package', function() {
  it('Check that training package encompasses inference', async function() {
    ort.env.wasm.numThreads = 1;
    await testInferenceFunction(ort, {executionProviders: ['wasm']});
  });

  it('Check training functionality, all options', async function() {
    ort.env.wasm.numThreads = 1;
    await testTrainingFunctionAll(ort, {executionProviders: ['wasm']});
  });

  it('Check training functionality, minimum options', async function() {
    ort.env.wasm.numThreads = 1;
    await testTrainingFunctionMin(ort, {executionProviders: ['wasm']});
  });
});
