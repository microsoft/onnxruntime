// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { InferenceSession, Tensor } from 'onnxruntime-common';

import { onnx } from '../../../../lib/onnxjs/ort-schema/protobuf/onnx';

type ViewOp = 'Reshape' | 'Unsqueeze';

const sessionOptions: InferenceSession.SessionOptions = {
  executionProviders: ['webgpu'],
  graphOptimizationLevel: 'disabled',
  extra: { session: { disable_cpu_ep_fallback: '1' } },
};

function createModel(
  opType: ViewOp,
  opset: number,
  elemType: number,
  inputDims: readonly number[],
  parameter: readonly number[],
  outputDims: readonly number[],
  allowzero = 0,
  parameterIsInput = false,
): Uint8Array {
  const valueInfo = (name: string, type: number, dims: readonly number[]) => ({
    name,
    type: { tensorType: { elemType: type, shape: { dim: dims.map((dimValue) => ({ dimValue })) } } },
  });
  const model = onnx.ModelProto.create({
    irVersion: opset >= 25 ? 13 : elemType >= 23 ? 12 : 10,
    opsetImport: [{ domain: '', version: opset }],
    graph: {
      name: `${opType}_${opset}`,
      node: [
        {
          opType,
          input: ['x', 'parameter'],
          output: ['y'],
          attribute:
            opType === 'Reshape'
              ? [{ name: 'allowzero', type: onnx.AttributeProto.AttributeType.INT, i: allowzero }]
              : [],
        },
      ],
      input: [
        valueInfo('x', elemType, inputDims),
        ...(parameterIsInput ? [valueInfo('parameter', onnx.TensorProto.DataType.INT64, [parameter.length])] : []),
      ],
      initializer: parameterIsInput
        ? []
        : [
            {
              name: 'parameter',
              dataType: onnx.TensorProto.DataType.INT64,
              dims: [parameter.length],
              int64Data: [...parameter],
            },
          ],
      output: [valueInfo('y', elemType, outputDims)],
    },
  });
  // Proxy mode transfers this buffer; keep protobufjs's shared allocation pool attached.
  return onnx.ModelProto.encode(model).finish().slice();
}

describe('WebGPU view operator registrations', function () {
  let adapter: GPUAdapter;

  before(async function (this: Mocha.Context) {
    const availableAdapter = await navigator.gpu?.requestAdapter();
    if (!availableAdapter) {
      this.skip();
    }
    adapter = availableAdapter;
  });

  const dataTypes = [
    { type: 'float32', elemType: onnx.TensorProto.DataType.FLOAT, data: new Float32Array([1, 2, 3, 4]) },
    {
      type: 'float16',
      elemType: onnx.TensorProto.DataType.FLOAT16,
      data: new Uint16Array([0x3c00, 0x4000, 0x4200, 0x4400]),
    },
    { type: 'int32', elemType: onnx.TensorProto.DataType.INT32, data: new Int32Array([-1, 2, -3, 4]) },
    { type: 'uint32', elemType: onnx.TensorProto.DataType.UINT32, data: new Uint32Array([1, 2, 3, 0xffffffff]) },
  ] as const;

  for (const opset of [21, 22, 23, 24, 25, 26, 27, 28]) {
    for (const opType of ['Reshape', 'Unsqueeze'] as const) {
      for (const { type, elemType, data } of dataTypes) {
        it(`${opType}-${opset} preserves ${type} data without CPU fallback`, async function (this: Mocha.Context) {
          if (type === 'float16' && !adapter.features.has('shader-f16')) {
            this.skip();
          }
          const inputDims = opType === 'Reshape' ? [2, 2] : [4];
          const parameter = opType === 'Reshape' ? [4] : [0];
          const outputDims = opType === 'Reshape' ? [4] : [1, 4];
          const session = await InferenceSession.create(
            createModel(opType, opset, elemType, inputDims, parameter, outputDims),
            sessionOptions,
          );
          try {
            const { y } = await session.run({ x: new Tensor(type, data.slice(), inputDims) });
            expect(y.type).to.equal(type);
            expect(y.dims).to.deep.equal(outputDims);
            const actual = y.data as typeof data;
            expect(new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength)).to.deep.equal(
              new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
            );
          } finally {
            await session.release();
          }
        });
      }
    }

    for (const { opType, inputDims, parameter, outputDims, allowzero } of [
      { opType: 'Reshape', inputDims: [2, 2], parameter: [0, -1], outputDims: [2, 2], allowzero: 0 },
      { opType: 'Reshape', inputDims: [0, 2], parameter: [2, 0], outputDims: [2, 0], allowzero: 1 },
      { opType: 'Unsqueeze', inputDims: [2, 2], parameter: [-1, 0], outputDims: [1, 2, 2, 1], allowzero: 0 },
    ] as const) {
      it(`${opType}-${opset} handles dynamic ${parameter} with allowzero=${allowzero}`, async () => {
        const data = inputDims[0] === 0 ? new Float32Array() : new Float32Array([1, 2, 3, 4]);
        const session = await InferenceSession.create(
          createModel(
            opType,
            opset,
            onnx.TensorProto.DataType.FLOAT,
            inputDims,
            parameter,
            outputDims,
            allowzero,
            true,
          ),
          sessionOptions,
        );
        try {
          const { y } = await session.run({
            x: new Tensor('float32', data.slice(), inputDims),
            parameter: new Tensor('int64', BigInt64Array.from(parameter, BigInt), [parameter.length]),
          });
          expect(y.type).to.equal('float32');
          expect(y.dims).to.deep.equal(outputDims);
          expect(Array.from(y.data as Float32Array)).to.deep.equal(Array.from(data));
        } finally {
          await session.release();
        }
      });
    }

    // The JS protobuf enum predates FLOAT4E2M1 (23), FLOAT8E8M0 (24), UINT2 (25), and INT2 (26).
    const unsupportedTypes = [
      { type: 'float64', elemType: onnx.TensorProto.DataType.DOUBLE },
      ...(opset >= 23 ? [{ type: 'float4e2m1', elemType: 23 }] : []),
      ...(opset >= 24 ? [{ type: 'float8e8m0', elemType: 24 }] : []),
      ...(opset >= 25
        ? [
            { type: 'uint2', elemType: 25 },
            { type: 'int2', elemType: 26 },
          ]
        : []),
    ];
    for (const opType of ['Reshape', 'Unsqueeze'] as const) {
      for (const { type, elemType } of unsupportedTypes) {
        it(`${opType}-${opset} does not claim unsupported ${type} tensors`, async () => {
          let session: InferenceSession | undefined;
          try {
            session = await InferenceSession.create(
              createModel(
                opType,
                opset,
                elemType,
                [4],
                opType === 'Reshape' ? [2, 2] : [0],
                opType === 'Reshape' ? [2, 2] : [1, 4],
              ),
              sessionOptions,
            );
          } catch (error) {
            // Low-precision types can also lack CPU kernels or be disabled in reduced builds.
            const reasons = [
              'fallback to CPU EP has been explicitly disabled',
              `Could not find an implementation for ${opType}`,
              `Provider type for ${opType} node with name '' is not set.`,
              `MLDataType for: tensor(${type}) is not currently registered or supported`,
            ];
            expect(
              reasons.some((reason) => String(error).includes(reason)),
              String(error),
            ).to.equal(true);
            return;
          } finally {
            await session?.release();
          }
          expect.fail(`An unsupported ${type} view must not be assigned to WebGPU.`);
        });
      }
    }
  }
});
