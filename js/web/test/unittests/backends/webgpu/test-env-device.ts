// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import * as ort from 'onnxruntime-common';

const ONNX_MODEL_TEST_ABS_STATIC = Uint8Array.from([
  8, 9, 58, 83, 10, 31, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 8, 111, 117, 116, 112, 117, 116, 95, 48, 26, 3, 65,
  98, 115, 34, 3, 65, 98, 115, 58, 0, 18, 3, 97, 98, 115, 90, 25, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 14, 10,
  12, 8, 1, 18, 8, 10, 2, 8, 2, 10, 2, 8, 4, 98, 16, 10, 8, 111, 117, 116, 112, 117, 116, 95, 48, 18, 4, 10, 2, 8, 1,
  66, 4, 10, 0, 16, 21,
]);

const describeWebGpu = typeof navigator !== 'undefined' && navigator.gpu ? describe : describe.skip;

describeWebGpu('#UnitTest# - WebGPU environment device', () => {
  it('uses an application-created device for inference and GPU tensors', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    expect(adapter).not.to.equal(null);
    const device = await adapter!.requestDevice();

    ort.env.webgpu.device = device;
    expect(ort.env.webgpu.device).to.equal(device);

    const inputData = new Float32Array([-1, 2, -3, 4, -5, 6, -7, 8]);
    const inputBuffer = device.createBuffer({
      // eslint-disable-next-line no-bitwise
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      size: inputData.byteLength,
      mappedAtCreation: true,
    });
    new Float32Array(inputBuffer.getMappedRange()).set(inputData);
    inputBuffer.unmap();

    const outputBuffer = device.createBuffer({
      // eslint-disable-next-line no-bitwise
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      size: inputData.byteLength,
    });
    const input = ort.Tensor.fromGpuBuffer(inputBuffer, {
      dataType: 'float32',
      dims: [2, 4],
      dispose: () => inputBuffer.destroy(),
    });
    const output = ort.Tensor.fromGpuBuffer(outputBuffer, {
      dataType: 'float32',
      dims: [2, 4],
      dispose: () => outputBuffer.destroy(),
      download: async () => {
        const stagingBuffer = device.createBuffer({
          // eslint-disable-next-line no-bitwise
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          size: outputBuffer.size,
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputBuffer.size);
        device.queue.submit([encoder.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        return data;
      },
    });

    const session = await ort.InferenceSession.create(ONNX_MODEL_TEST_ABS_STATIC, {
      executionProviders: ['webgpu'],
    });
    try {
      const results = await session.run({ input_0: input }, { output_0: output });
      const resultData = (await results.output_0.getData()) as Float32Array;
      expect(Array.from(resultData)).to.deep.equal([1, 2, 3, 4, 5, 6, 7, 8]);
      expect(ort.env.webgpu.device).to.equal(device);
    } finally {
      input.dispose();
      output.dispose();
      await session.release();
    }
  });
});
