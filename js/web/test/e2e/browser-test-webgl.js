// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebGL backend', async function () {
  await testFunction(ort, { executionProviders: ['webgl'] });
});

it('Browser E2E testing - invalid buffer', async () => {
  try {
    await ort.InferenceSession.create(new Uint8Array(Array.from({ length: 100 }, () => 42)), {
      executionProviders: ['webgl'],
    });

    // Should not reach here.
    assert(false);
  } catch (e) {
    assert(e.message.includes('as ONNX format'));
    assert(e.message.includes('as ORT format'));
  }
});
