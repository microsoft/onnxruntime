// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebAssembly backend (no threads)', async function() {
  ort.env.wasm.numThreads = 1;
  await testFunction(ort, {executionProviders: ['wasm']});
});
