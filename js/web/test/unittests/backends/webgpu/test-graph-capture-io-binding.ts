// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { env, InferenceSession, Tensor } from 'onnxruntime-common';

import { onnx } from '../../../../lib/onnxjs/ort-schema/protobuf/onnx';

// y = 2x + 1, z = x * x over float32 [n].
function createModel(): Uint8Array {
  const valueInfo = (name: string) => ({
    name,
    type: { tensorType: { elemType: onnx.TensorProto.DataType.FLOAT, shape: { dim: [{ dimParam: 'n' }] } } },
  });
  const scalar = (name: string, value: number) => ({
    name,
    dataType: onnx.TensorProto.DataType.FLOAT,
    dims: [1],
    floatData: [value],
  });
  const model = onnx.ModelProto.create({
    irVersion: 8,
    opsetImport: [{ domain: '', version: 17 }],
    graph: {
      name: 'graph_capture_io_binding',
      node: [
        { opType: 'Mul', input: ['x', 'two'], output: ['t'] },
        { opType: 'Add', input: ['t', 'one'], output: ['y'] },
        { opType: 'Mul', input: ['x', 'x'], output: ['z'] },
      ],
      input: [valueInfo('x')],
      initializer: [scalar('two', 2), scalar('one', 1)],
      output: [valueInfo('y'), valueInfo('z')],
    },
  });
  // Proxy mode transfers this buffer; keep protobufjs's shared allocation pool attached.
  return onnx.ModelProto.encode(model).finish().slice();
}

describe('WebGPU graph capture IO binding', function () {
  const model = createModel();
  // The EP entry must be an object: only then is enableGraphCapture passed to the WebGPU EP options.
  const sessionOptions: InferenceSession.SessionOptions = {
    executionProviders: [{ name: 'webgpu' }],
    preferredOutputLocation: 'gpu-buffer',
    enableGraphCapture: true,
  };
  let session: InferenceSession | undefined;
  let gpuDevice: GPUDevice;
  const buffers: GPUBuffer[] = [];

  before(async function (this: Mocha.Context) {
    if (!(await navigator.gpu?.requestAdapter())) {
      this.skip();
    }
  });
  afterEach(async () => {
    await session?.release();
    session = undefined;
    buffers.splice(0).forEach((buffer) => buffer.destroy());
  });

  const device = () => gpuDevice;
  const createSession = async () => {
    session = await InferenceSession.create(model, sessionOptions);
    gpuDevice = await env.webgpu.device;
  };
  const createBuffer = (n: number) => {
    const buffer = device().createBuffer({
      size: n * 4,
      // eslint-disable-next-line no-bitwise
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    buffers.push(buffer);
    return buffer;
  };
  const gpuTensor = (buffer: GPUBuffer, n: number) => Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims: [n] });
  const read = async (buffer: GPUBuffer, n: number) => {
    // eslint-disable-next-line no-bitwise
    const staging = device().createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device().createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, n * 4);
    device().queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const values = Array.from(new Float32Array(staging.getMappedRange()));
    staging.destroy();
    return values;
  };
  const inputValues = (run: number, n: number) => Array.from({ length: n }, (_, i) => i + 1 + 10 * run);
  const expectOutputs = (x: number[], y: ArrayLike<number>, z: ArrayLike<number>) => {
    expect(Array.from(y)).to.deep.equal(x.map((v) => 2 * v + 1));
    expect(Array.from(z)).to.deep.equal(x.map((v) => v * v));
  };
  /** Runs with pre-allocated input/output buffers and checks the outputs. */
  async function runPreallocated(run: number, graphId: string, x: GPUBuffer, y: GPUBuffer, z: GPUBuffer, n: number) {
    const values = inputValues(run, n);
    device().queue.writeBuffer(x, 0, new Float32Array(values));
    await session!.run({ x: gpuTensor(x, n) }, { y: gpuTensor(y, n), z: gpuTensor(z, n) }, {
      extra: { gpu_graph_id: graphId },
    } as InferenceSession.RunOptions);
    expectOutputs(values, await read(y, n), await read(z, n));
  }

  it('replays with ORT-allocated outputs', async () => {
    await createSession();
    const x = createBuffer(4);
    for (let run = 0; run < 4; run++) {
      const values = inputValues(run, 4);
      device().queue.writeBuffer(x, 0, new Float32Array(values));
      const outputs = await session!.run({ x: gpuTensor(x, 4) });
      expectOutputs(values, (await outputs.y.getData()) as Float32Array, (await outputs.z.getData()) as Float32Array);
    }
  });

  it('mixes uncaptured runs with a captured graph', async () => {
    await createSession();
    const [x8, y8, z8, x4, y4, z4] = [
      createBuffer(8),
      createBuffer(8),
      createBuffer(8),
      createBuffer(4),
      createBuffer(4),
      createBuffer(4),
    ];
    await runPreallocated(0, '-1', x8, y8, z8, 8);
    await runPreallocated(1, '1', x4, y4, z4, 4); // capture
    await runPreallocated(2, '1', x4, y4, z4, 4); // replay
    await runPreallocated(3, '-1', x8, y8, z8, 8);
  });

  it('captures and replays two graphs', async () => {
    await createSession();
    const [x4, y4, z4, x8, y8, z8] = [
      createBuffer(4),
      createBuffer(4),
      createBuffer(4),
      createBuffer(8),
      createBuffer(8),
      createBuffer(8),
    ];
    await runPreallocated(0, '1', x4, y4, z4, 4);
    await runPreallocated(1, '2', x8, y8, z8, 8);
    await runPreallocated(2, '1', x4, y4, z4, 4);
    await runPreallocated(3, '2', x8, y8, z8, 8);
  });
});
