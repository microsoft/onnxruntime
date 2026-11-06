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

it('Browser E2E testing - WebGPU backend with a user-provided GPUDevice and gpu-buffer inputs/outputs', async function () {
  setupEnvFlags(ort);

  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter);

  const device = await adapter.requestDevice();
  try {
    const session = await createSession(ort, {
      executionProviders: [{ name: 'webgpu', device }],
      preferredOutputLocation: 'gpu-buffer',
    });

    const createInput = (data, dims) => {
      const buffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, data);
      return ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims });
    };

    const fetches = await session.run({
      a: createInput(Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), [3, 4]),
      b: createInput(Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]), [4, 3]),
    });

    const c = fetches.c;
    assert(c instanceof ort.Tensor);
    assert(c.location === 'gpu-buffer');
    assert(c.dims.length === 2 && c.dims[0] === 3 && c.dims[1] === 3);

    const expected = [700, 800, 900, 1580, 1840, 2100, 2460, 2880, 3300];
    const data = await c.getData(true);
    for (let i = 0; i < expected.length; i++) {
      assert(data[i] === expected[i]);
    }

    await session.release();
  } finally {
    device.destroy();
  }
});
