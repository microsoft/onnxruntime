// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { env } from 'onnxruntime-common';

import { WebGpuBackend } from '../../../../lib/wasm/jsep/backend-webgpu';

describe('WebGPU upload ordering', function () {
  let backend: WebGpuBackend;

  beforeEach(async function (this: Mocha.Context) {
    backend = new WebGpuBackend();
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
      this.skip();
    }
    await backend.initialize(
      { ...env, trace: false, wasm: {}, webgpu: { ...env.webgpu, profiling: { mode: 'off' } } },
      adapter,
    );
    backend.device.pushErrorScope('validation');
  });

  afterEach(async function () {
    if (!backend.device) {
      return;
    }
    try {
      backend.flush();
      expect(await backend.device.popErrorScope()).to.equal(null);
    } finally {
      backend.dispose();
      backend.device.destroy();
    }
  });

  for (const operation of ['compute', 'memcpy'] as const) {
    for (const { name, size, offset } of [
      { name: 'aligned data', size: 16, offset: 0 },
      { name: 'unaligned data offset', size: 16, offset: 1 },
      { name: 'padded data', size: 3, offset: 0 },
    ]) {
      it(`preserves pending ${operation} inputs when uploading ${name} to the same buffer`, async function () {
        const input = backend.alloc(size);
        const firstOutput = backend.alloc(size);
        const secondOutput = backend.alloc(size);
        const first = new Uint8Array(new ArrayBuffer(size + offset), offset, size).fill(1);
        const second = new Uint8Array(new ArrayBuffer(size + offset), offset, size).fill(2);

        const pipeline =
          operation === 'compute'
            ? backend.device.createComputePipeline({
                layout: 'auto',
                compute: {
                  module: backend.device.createShaderModule({
                    code: `
                      @group(0) @binding(0) var<storage, read> input: array<u32>;
                      @group(0) @binding(1) var<storage, read_write> output: array<u32>;
                      @compute @workgroup_size(1)
                      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                        output[id.x] = input[id.x];
                      }
                    `,
                  }),
                  entryPoint: 'main',
                },
              })
            : undefined;

        const recordRead = (output: number) => {
          if (!pipeline) {
            backend.memcpy(input, output);
            // Copies leave the dispatch counter at zero but still need submission before the upload.
            expect(backend.pendingDispatchNumber).to.equal(0);
            return;
          }
          const pass = backend.getComputePassEncoder();
          pass.setPipeline(pipeline);
          pass.setBindGroup(
            0,
            backend.device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: backend.gpuDataManager.get(input)!.buffer } },
                { binding: 1, resource: { buffer: backend.gpuDataManager.get(output)!.buffer } },
              ],
            }),
          );
          pass.dispatchWorkgroups(4);
          backend.pendingDispatchNumber++;
        };

        // Model the allocation planner reusing one input buffer for successive CPU EP outputs.
        // Neither read is explicitly flushed before the next upload.
        backend.upload(input, first);
        recordRead(firstOutput);
        backend.upload(input, second);
        recordRead(secondOutput);

        const firstResult = new Uint8Array(size);
        const secondResult = new Uint8Array(size);
        await backend.download(firstOutput, () => firstResult);
        await backend.download(secondOutput, () => secondResult);
        expect(Array.from(firstResult)).to.deep.equal(Array.from(first));
        expect(Array.from(secondResult)).to.deep.equal(Array.from(second));
      });
    }
  }
});
