// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - explicit WebGPU options override extra config', async function () {
  await testFunction(ort, {
    executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }],
    extra: { 'ep.webgpuexecutionprovider.preferredLayout': 'invalid-extra-value' },
  });
});

it('Browser E2E testing - WebGPU validates extra config before creating the EP', async function () {
  let rejected = false;
  try {
    const session = await ort.InferenceSession.create('./model.onnx', {
      executionProviders: ['webgpu'],
      extra: { 'ep.webgpuexecutionprovider.enableGraphCapture': 'invalid-extra-value' },
    });
    await session.release();
  } catch {
    rejected = true;
  }
  // Before the fix, provider construction ignores the invalid value and succeeds.
  assert(rejected);
});
