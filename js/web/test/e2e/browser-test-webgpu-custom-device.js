// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebGPU backend with a user-provided GPUDevice', async function () {
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter);

  const device = await adapter.requestDevice();
  const requestAdapter = navigator.gpu.requestAdapter;
  navigator.gpu.requestAdapter = () => {
    throw new Error('A user-provided GPUDevice must not trigger another adapter request.');
  };
  try {
    await testFunction(ort, {
      executionProviders: [{ name: 'webgpu', device }],
      extra: { session: { disable_cpu_ep_fallback: '1' } },
    });
  } finally {
    navigator.gpu.requestAdapter = requestAdapter;
    device.destroy();
  }
});
