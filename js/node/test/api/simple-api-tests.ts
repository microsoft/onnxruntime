// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import { InferenceSession, Tensor } from 'onnxruntime-common';
import * as path from 'path';

import { assertDataEqual, TEST_DATA_ROOT } from '../test-utils';

const MODEL_TEST_TYPES_CASES: Array<{
  model: string;
  type: Tensor.Type;
  input0: Tensor.DataType;
  expectedOutput0: Tensor.DataType;
}> = [
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_bool.onnx'),
    type: 'bool',
    input0: Uint8Array.from([1, 0, 0, 1, 0]),
    expectedOutput0: Uint8Array.from([1, 0, 0, 1, 0]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_double.onnx'),
    type: 'float64',
    input0: Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
    expectedOutput0: Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_float.onnx'),
    type: 'float32',
    input0: Float32Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
    expectedOutput0: Float32Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_int8.onnx'),
    type: 'int8',
    input0: Int8Array.from([1, -2, 3, 4, -5]),
    expectedOutput0: Int8Array.from([1, -2, 3, 4, -5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_int16.onnx'),
    type: 'int16',
    input0: Int16Array.from([1, -2, 3, 4, -5]),
    expectedOutput0: Int16Array.from([1, -2, 3, 4, -5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_int32.onnx'),
    type: 'int32',
    input0: Int32Array.from([1, -2, 3, 4, -5]),
    expectedOutput0: Int32Array.from([1, -2, 3, 4, -5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_int64.onnx'),
    type: 'int64',
    input0: BigInt64Array.from([BigInt(1), BigInt(-2), BigInt(3), BigInt(4), BigInt(-5)]),
    expectedOutput0: BigInt64Array.from([BigInt(1), BigInt(-2), BigInt(3), BigInt(4), BigInt(-5)]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_string.onnx'),
    type: 'string',
    input0: ['a', 'b', 'c', 'd', 'e'],
    expectedOutput0: ['a', 'b', 'c', 'd', 'e'],
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_uint8.onnx'),
    type: 'uint8',
    input0: Uint8Array.from([1, 2, 3, 4, 5]),
    expectedOutput0: Uint8Array.from([1, 2, 3, 4, 5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_uint16.onnx'),
    type: 'uint16',
    input0: Uint16Array.from([1, 2, 3, 4, 5]),
    expectedOutput0: Uint16Array.from([1, 2, 3, 4, 5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_uint32.onnx'),
    type: 'uint32',
    input0: Uint32Array.from([1, 2, 3, 4, 5]),
    expectedOutput0: Uint32Array.from([1, 2, 3, 4, 5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_uint64.onnx'),
    type: 'uint64',
    input0: BigUint64Array.from([BigInt(1), BigInt(2), BigInt(3), BigInt(4), BigInt(5)]),
    expectedOutput0: BigUint64Array.from([BigInt(1), BigInt(2), BigInt(3), BigInt(4), BigInt(5)]),
  },
];

describe('API Tests - simple API tests', () => {
  MODEL_TEST_TYPES_CASES.forEach((testCase) => {
    it(`${testCase.model}`, async () => {
      const session = await InferenceSession.create(testCase.model);
      const output = await session.run({ input: new Tensor(testCase.type, testCase.input0, [1, 5]) });
      assert(Object.prototype.hasOwnProperty.call(output, 'output'), "'output' should be in the result object.");
      assert(output.output instanceof Tensor, 'result[output] should be a Tensor object.');
      assert.strictEqual(output.output.size, 5, `output size expected 5, got ${output.output.size}.`);
      assert.strictEqual(
        output.output.type,
        testCase.type,
        `tensor type expected ${testCase.type}, got ${output.output.type}.`,
      );
      assert.strictEqual(
        Object.getPrototypeOf(output.output.data),
        Object.getPrototypeOf(testCase.expectedOutput0),
        `tensor data expected ${Object.getPrototypeOf(testCase.expectedOutput0).constructor.name}, got ${
          Object.getPrototypeOf(output.output.data).constructor.name
        }`,
      );
      assertDataEqual(testCase.type, output.output.data, testCase.expectedOutput0);
    });
  });
});

describe('API Tests - float16 tensor input', () => {
  const MODEL = path.join(TEST_DATA_ROOT, 'test_types_float16.onnx');
  const INPUT_BITS = [0x3c00];
  const Float16ArrayCtor = (globalThis as { Float16Array?: new (length: number) => object }).Float16Array;

  function assertFloat16Output(output: Tensor): void {
    assert.strictEqual(output.type, 'float16');
    const view = output.data as unknown as Uint16Array;
    assert.deepStrictEqual(Array.from(new Uint16Array(view.buffer, view.byteOffset, view.length)), INPUT_BITS);
  }

  it('test_types_float16.onnx accepts Uint16Array input data', async () => {
    const data = Uint16Array.from([0x7fff, INPUT_BITS[0]]).subarray(1);
    const output = await (
      await InferenceSession.create(MODEL)
    ).run({ input: new Tensor('float16', data, [1, 1, 1, 1]) });
    assertFloat16Output(output.output);
  });

  it('test_types_float16.onnx accepts Float16Array input data', async function () {
    if (!Float16ArrayCtor) {
      this.skip();
    }
    const data = new Float16ArrayCtor(1) as unknown as Uint16Array;
    new Uint16Array(data.buffer)[0] = INPUT_BITS[0];
    const output = await (await InferenceSession.create(MODEL)).run({
      input: new Tensor('float16', data, [1, 1, 1, 1]),
    });
    assertFloat16Output(output.output);
  });
});
