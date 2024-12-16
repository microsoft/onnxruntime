// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import * as ort from 'onnxruntime-web/wasm';
import { setupMultipleThreads, default as testInferenceAndValidate } from './shared.js';

if (typeof SharedArrayBuffer === 'undefined') {
  it('Browser package consuming test - single-thread - [js][esm]', async function () {
    await testInferenceAndValidate(ort, { executionProviders: ['wasm'] });
  });
} else {
  it('Browser package consuming test - multi-thread - [js][esm]', async function () {
    setupMultipleThreads(ort);
    await testInferenceAndValidate(ort, { executionProviders: ['wasm'] });
  });
}
