// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { InferenceSession, Tensor } from 'onnxruntime-common';

import { onnx } from '../../../../lib/onnxjs/ort-schema/protobuf/onnx';

// The native WebGPU `.asyncify` and `.jspi` artifacts are built with reduced-size options that remove these
// operators and types. Every other artifact, including `.asyncify-full`, must keep them.
const isReducedBuild =
  BUILD_DEFS.DISABLE_JSEP && !BUILD_DEFS.DISABLE_WEBGPU && (BUILD_DEFS.ENABLE_JSPI || !BUILD_DEFS.ENABLE_ASYNCIFY_FULL);

const sessionOptions: InferenceSession.SessionOptions = { executionProviders: ['wasm'] };

interface CoverageCase {
  readonly name: string;
  readonly opType: string;
  readonly domain?: string;
  readonly opset: number;
  readonly irVersion: number;
  readonly elemType: number;
  readonly inputs: Readonly<Record<string, Tensor>>;
  readonly outputDims: readonly number[];
  readonly attribute?: onnx.IAttributeProto[];
  readonly expected: readonly unknown[];
  readonly availableInReducedBuild: boolean;
}

function createModel(testCase: CoverageCase): Uint8Array {
  const valueInfo = (name: string, dims: readonly number[]) => ({
    name,
    type: { tensorType: { elemType: testCase.elemType, shape: { dim: dims.map((dimValue) => ({ dimValue })) } } },
  });
  const model = onnx.ModelProto.create({
    irVersion: testCase.irVersion,
    opsetImport: [
      { domain: '', version: testCase.domain ? 20 : testCase.opset },
      ...(testCase.domain ? [{ domain: testCase.domain, version: testCase.opset }] : []),
    ],
    graph: {
      name: testCase.name,
      node: [
        {
          opType: testCase.opType,
          domain: testCase.domain ?? '',
          input: Object.keys(testCase.inputs),
          output: ['y'],
          attribute: testCase.attribute ?? [],
        },
      ],
      input: Object.entries(testCase.inputs).map(([name, tensor]) => valueInfo(name, tensor.dims)),
      output: [valueInfo('y', testCase.outputDims)],
    },
  });
  return onnx.ModelProto.encode(model).finish().slice();
}

const cases: readonly CoverageCase[] = [
  {
    name: 'float32 Add',
    opType: 'Add',
    opset: 14,
    irVersion: 8,
    elemType: onnx.TensorProto.DataType.FLOAT,
    inputs: { a: new Tensor('float32', [1, 2], [2]), b: new Tensor('float32', [3, 4], [2]) },
    outputDims: [2],
    expected: [4, 6],
    availableInReducedBuild: true,
  },
  {
    name: 'float64 Add',
    opType: 'Add',
    opset: 14,
    irVersion: 8,
    elemType: onnx.TensorProto.DataType.DOUBLE,
    inputs: { a: new Tensor('float64', [1.5, 2], [2]), b: new Tensor('float64', [3, 4.25], [2]) },
    outputDims: [2],
    expected: [4.5, 6.25],
    availableInReducedBuild: false,
  },
  {
    name: 'int16 Add',
    opType: 'Add',
    opset: 14,
    irVersion: 8,
    elemType: onnx.TensorProto.DataType.INT16,
    inputs: { a: new Tensor('int16', [-1, 2], [2]), b: new Tensor('int16', [3, -4], [2]) },
    outputDims: [2],
    expected: [2, -2],
    availableInReducedBuild: false,
  },
  {
    name: 'string StringConcat',
    opType: 'StringConcat',
    opset: 20,
    irVersion: 9,
    elemType: onnx.TensorProto.DataType.STRING,
    inputs: { a: new Tensor('string', ['a', 'b'], [2]), b: new Tensor('string', ['c', 'd'], [2]) },
    outputDims: [2],
    expected: ['ac', 'bd'],
    availableInReducedBuild: false,
  },
  {
    name: 'ai.onnx.ml LinearRegressor',
    opType: 'LinearRegressor',
    domain: 'ai.onnx.ml',
    opset: 1,
    irVersion: 9,
    elemType: onnx.TensorProto.DataType.FLOAT,
    inputs: { x: new Tensor('float32', [1, 2], [1, 2]) },
    outputDims: [1, 1],
    attribute: [
      { name: 'coefficients', type: onnx.AttributeProto.AttributeType.FLOATS, floats: [3, 4] },
      { name: 'intercepts', type: onnx.AttributeProto.AttributeType.FLOATS, floats: [0.5] },
    ],
    expected: [11.5],
    availableInReducedBuild: false,
  },
];

describe('#UnitTest# - wasm - operator and type coverage of the WebAssembly build', () => {
  for (const testCase of cases) {
    const expectSupported = !isReducedBuild || testCase.availableInReducedBuild;
    it(`${expectSupported ? 'supports' : 'excludes'} ${testCase.name}`, async () => {
      let session: InferenceSession;
      try {
        session = await InferenceSession.create(createModel(testCase), sessionOptions);
      } catch (error) {
        if (expectSupported) {
          throw error;
        }
        return;
      }
      try {
        expect(expectSupported, `${testCase.name} must not be available in a reduced build.`).to.equal(true);
        const { y } = await session.run(testCase.inputs);
        expect(y.dims).to.deep.equal(testCase.outputDims);
        expect(Array.from(y.data as ArrayLike<unknown>)).to.deep.equal(testCase.expected);
      } finally {
        await session.release();
      }
    });
  }
});
