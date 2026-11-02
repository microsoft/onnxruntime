// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

let legacyFactoryUrl;
before(() => {
  setupEnvFlags(ort);
  if (typeof __ort_arg_legacy === 'undefined') return;
  const prefix = new URL(globalThis.__ort_arg_wasm_prefix || '/base/node_modules/onnxruntime-web/dist/', location.href);
  // Exercise legacy registration with the real native provider even in non-minimal CI builds.
  // Minimal builds return ORT_NOT_IMPLEMENTED from device discovery.
  const factory = `
    import createModule from ${JSON.stringify(new URL('ort-wasm-simd-threaded.asyncify.mjs', prefix).href)};
    export default async function(options) {
      const wasm = await createModule(options);
      wasm._OrtGetEpDevices = () => 9;
      return wasm;
    }
  `;
  legacyFactoryUrl = URL.createObjectURL(new Blob([factory], { type: 'text/javascript' }));
  ort.env.wasm.wasmPaths = {
    mjs: legacyFactoryUrl,
    wasm: new URL('ort-wasm-simd-threaded.asyncify.wasm', prefix).href,
  };
});

after(() => {
  if (legacyFactoryUrl) URL.revokeObjectURL(legacyFactoryUrl);
});

// Three Sigmoid nodes, with graph optimizations disabled so all three are dispatched.
const threeSigmoids = Uint8Array.from([
  8, 8, 58, 101, 10, 15, 10, 1, 120, 18, 1, 97, 34, 7, 83, 105, 103, 109, 111, 105, 100, 10, 15, 10, 1, 97, 18, 1, 98,
  34, 7, 83, 105, 103, 109, 111, 105, 100, 10, 15, 10, 1, 98, 18, 1, 121, 34, 7, 83, 105, 103, 109, 111, 105, 100, 18,
  14, 116, 104, 114, 101, 101, 95, 115, 105, 103, 109, 111, 105, 100, 115, 90, 15, 10, 1, 120, 18, 10, 10, 8, 8, 1, 18,
  4, 10, 2, 8, 4, 98, 15, 10, 1, 121, 18, 10, 10, 8, 8, 1, 18, 4, 10, 2, 8, 4, 66, 4, 10, 0, 16, 13,
]);

for (const nested of [false, true]) {
  it(`Browser E2E testing - WebGPU honors ${nested ? 'nested' : 'flat'} EP session configuration`, async function () {
    const adapter = await navigator.gpu.requestAdapter();
    assert(adapter);
    const device = await adapter.requestDevice();
    let session;
    const dispatchCounts = new WeakMap();
    const submittedDispatchCounts = [];
    const createCommandEncoder = device.createCommandEncoder.bind(device);
    device.createCommandEncoder = (...args) => {
      const encoder = createCommandEncoder(...args);
      let dispatches = 0;
      const beginComputePass = encoder.beginComputePass.bind(encoder);
      encoder.beginComputePass = (...passArgs) => {
        const pass = beginComputePass(...passArgs);
        const dispatchWorkgroups = pass.dispatchWorkgroups.bind(pass);
        pass.dispatchWorkgroups = (...dispatchArgs) => {
          dispatches++;
          return dispatchWorkgroups(...dispatchArgs);
        };
        return pass;
      };
      const finish = encoder.finish.bind(encoder);
      encoder.finish = (...finishArgs) => {
        const buffer = finish(...finishArgs);
        dispatchCounts.set(buffer, dispatches);
        return buffer;
      };
      return encoder;
    };
    const submit = device.queue.submit.bind(device.queue);
    device.queue.submit = (buffers) => {
      const commands = Array.from(buffers);
      submittedDispatchCounts.push(commands.reduce((sum, buffer) => sum + dispatchCounts.get(buffer), 0));
      return submit(commands);
    };
    try {
      session = await ort.InferenceSession.create(threeSigmoids, {
        executionProviders: [{ name: 'webgpu', device }],
        graphOptimizationLevel: 'disabled',
        extra: nested
          ? { ep: { webgpuexecutionprovider: { maxNumPendingDispatches: '1' } } }
          : { 'ep.webgpuexecutionprovider.maxNumPendingDispatches': '1' },
      });
      const input = [-2, -1, 0, 1];
      const result = await session.run({ x: new ort.Tensor('float32', input, [4]) });
      for (let i = 0; i < input.length; i++) {
        let expected = input[i];
        for (let j = 0; j < 3; j++) expected = 1 / (1 + Math.exp(-expected));
        assert(Math.abs(result.y.data[i] - expected) < 1e-6);
      }
      assert(submittedDispatchCounts.reduce((sum, count) => sum + count, 0) === 3);
      const maxDispatches = Math.max(...submittedDispatchCounts);
      if (maxDispatches !== 1) throw new Error(`Expected at most 1 dispatch per submission, got ${maxDispatches}`);
    } finally {
      if (session) await session.release();
      device.destroy();
    }
  });
}
