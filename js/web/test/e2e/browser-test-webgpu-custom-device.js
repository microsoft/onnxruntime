// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebGPU backend with a user-provided GPUDevice', async function () {
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter);

  const device = await adapter.requestDevice();
  try {
    await testFunction(ort, { executionProviders: [{ name: 'webgpu', device }] });
  } finally {
    device.destroy();
  }
});
