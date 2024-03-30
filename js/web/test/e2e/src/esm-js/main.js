// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import * as ort from 'onnxruntime-web/wasm';
import {setupMultipleThreads, default as testInferenceAndValidate} from './shared.js';

it('Browser package consuming test - [.js][esm]', async function() {
  setupMultipleThreads(ort);
  await testInferenceAndValidate(ort, {executionProviders: ['wasm']});
});
