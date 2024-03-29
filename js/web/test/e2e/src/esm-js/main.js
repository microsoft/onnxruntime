// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import * as ort from 'onnxruntime-web/experimental';
import testInferenceAndValidate from './shared.js';

it('Browser package consuming test - [.js][esm]', async function() {
  ort.env.wasm.numThreads = 1;
  await testInferenceAndValidate(ort, {executionProviders: ['wasm']});
});
