// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { InferenceSession, Tensor } from 'onnxruntime-common';

import { onnx } from '../../../../lib/onnxjs/ort-schema/protobuf/onnx';

// Exercises WebNN backend initialization (`jsepInit('webnn', ...)` or native `webnnInit(...)`), session creation and
// inference. CPU fallback is disabled so the tests fail if the WebNN EP is not used.

const dims = [2, 2];
const a = new Float32Array([1, 2, 3, 4]);
const b = new Float32Array([10, 20, 30, 40]);
const expected = [11, 22, 33, 44];

function createAddModel(): Uint8Array {
  const valueInfo = (name: string) => ({
    name,
    type: {
      tensorType: {
        elemType: onnx.TensorProto.DataType.FLOAT,
        shape: { dim: dims.map((dimValue) => ({ dimValue })) },
      },
    },
  });
  const model = onnx.ModelProto.create({
    irVersion: 8,
    opsetImport: [{ domain: '', version: 13 }],
    graph: {
      name: 'webnn_add',
      node: [{ opType: 'Add', input: ['a', 'b'], output: ['y'] }],
      input: [valueInfo('a'), valueInfo('b')],
      output: [valueInfo('y')],
    },
  });
  // Proxy mode transfers this buffer; keep protobufjs's shared allocation pool attached.
  return onnx.ModelProto.encode(model).finish().slice();
}

const noCpuFallback: InferenceSession.SessionOptions['extra'] = { session: { disable_cpu_ep_fallback: '1' } };

describe('#UnitTest# - webnn - session initialization and inference', function () {
  before(function (this: Mocha.Context) {
    if (typeof navigator === 'undefined' || !navigator.ml) {
      this.skip();
    }
  });

  it('runs a model with an ORT-created WebNN context', async () => {
    const session = await InferenceSession.create(createAddModel(), {
      executionProviders: [{ name: 'webnn', deviceType: 'cpu' }],
      extra: noCpuFallback,
    });
    try {
      const { y } = await session.run({ a: new Tensor('float32', a, dims), b: new Tensor('float32', b, dims) });
      expect(y.location).to.equal('cpu');
      expect(y.dims).to.deep.equal(dims);
      expect(Array.from(y.data as Float32Array)).to.deep.equal(expected);
    } finally {
      await session.release();
    }
  });

  it('runs a model with an application MLContext and MLTensor inputs and outputs', async () => {
    const context = await navigator.ml.createContext({ deviceType: 'cpu' });
    const createInput = async (data: Float32Array) => {
      const mlTensor = await context.createTensor({
        dataType: 'float32',
        shape: dims,
        usage: undefined,
        writable: true,
      });
      context.writeTensor(mlTensor, data);
      return Tensor.fromMLTensor(mlTensor, { dataType: 'float32', dims, dispose: () => mlTensor.destroy() });
    };

    const session = await InferenceSession.create(createAddModel(), {
      executionProviders: [{ name: 'webnn', context }],
      preferredOutputLocation: 'ml-tensor',
      extra: noCpuFallback,
    });
    const inputA = await createInput(a);
    const inputB = await createInput(b);
    try {
      const { y } = await session.run({ a: inputA, b: inputB });
      try {
        expect(y.location).to.equal('ml-tensor');
        expect(y.dims).to.deep.equal(dims);
        expect(Array.from((await y.getData()) as Float32Array)).to.deep.equal(expected);
      } finally {
        y.dispose();
      }
    } finally {
      inputA.dispose();
      inputB.dispose();
      await session.release();
    }
  });
});
