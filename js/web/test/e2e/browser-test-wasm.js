// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebAssembly backend', async function () {
  await testFunction(ort, { executionProviders: ['wasm'] });
});
