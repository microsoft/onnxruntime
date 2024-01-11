// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebGPU backend with external data', async function() {
  const session = await createSession('./model_with_orig_ext_data.onnx', {
    executionProviders: ['webgpu'],
    externalData: [{data: './model_with_orig_ext_data.bin', path: './model_with_orig_ext_data.bin'}]
  });

  const fetches = await session.run({X: new ort.Tensor('float32', [1, 1], [1, 2])});

  const Y = fetches.Y;

  assert(Y instanceof ort.Tensor);
  assert(Y.dims.length === 2 && c.dims[0] === 2 && c.dims[1] === 3);
  assert(c.data[0] === 1);
  assert(c.data[1] === 1);
  assert(c.data[2] === 0);
  assert(c.data[3] === 0);
  assert(c.data[4] === 0);
  assert(c.data[5] === 0);
});
