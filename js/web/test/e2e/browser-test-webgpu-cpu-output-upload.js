// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// model_cpu_output_upload.onnx: i0 = Cast(idx0), i1 = Cast(idx1), Y0 = Gather(X, i0), Y1 = Gather(X, i1).
// Cast to int64 runs on the CPU EP, so i0 and i1 are uploaded to the GPU during the run, each right before the
// Gather that reads it. Both uploads land in the same GPU buffer, so the second upload must not overtake the
// still-unsubmitted Gather that reads the first.
it('Browser E2E testing - WebGPU backend reading CPU EP outputs uploaded during a run', async function () {
  const session = await ort.InferenceSession.create('./model_cpu_output_upload.onnx', {
    executionProviders: ['webgpu'],
  });

  const x = Float32Array.from({ length: 16 }, (_, i) => i); // 8 rows of 2
  const fetches = await session.run({
    X: new ort.Tensor('float32', x, [8, 2]),
    idx0: new ort.Tensor('float32', [0, 2, 4, 6], [4]),
    idx1: new ort.Tensor('float32', [1, 3, 5, 7], [4]),
  });

  const expected = { Y0: [0, 1, 4, 5, 8, 9, 12, 13], Y1: [2, 3, 6, 7, 10, 11, 14, 15] };
  for (const [name, want] of Object.entries(expected)) {
    const Y = fetches[name];
    assert(Y instanceof ort.Tensor);
    assert(Y.dims.length === 2 && Y.dims[0] === 4 && Y.dims[1] === 2);
    want.forEach((v, i) => assert(Y.data[i] === v));
  }
});
