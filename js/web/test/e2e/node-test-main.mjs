// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable */

import * as ort from 'onnxruntime-web';

import testFunction from './common.mjs';

it('Node.js E2E testing - WebAssembly backend[esm]', async function () {
  await testFunction(ort, { executionProviders: ['wasm'] });
});
