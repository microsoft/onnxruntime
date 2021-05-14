// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

it('Browser E2E testing - WebGL backend', async function () {
  await testFunction(ort, { executionProviders: ['webgl'] });
});
