// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import {InferenceSession} from 'onnxruntime-common';
import * as path from 'path';

import {Tensor} from '../../lib';
import {assertDataEqual, TEST_DATA_ROOT} from '../test-utils';


const MODEL_TEST_TYPES_CASES:
    Array<{model: string; type: Tensor.Type; input0: Tensor.DataType; expectedOutput0: Tensor.DataType}> = [
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_BOOL.pb'),
        type: 'bool',
        input0: Uint8Array.from([1, 0, 0, 1, 0]),
        expectedOutput0: Uint8Array.from([1, 0, 0, 1, 0])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_DOUBLE.pb'),
        type: 'float64',
        input0: Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
        expectedOutput0: Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_FLOAT.pb'),
        type: 'float32',
        input0: Float32Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
        expectedOutput0: Float32Array.from([1.0, 2.0, 3.0, 4.0, 5.0])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_INT8.pb'),
        type: 'int8',
        input0: Int8Array.from([1, -2, 3, 4, -5]),
        expectedOutput0: Int8Array.from([1, -2, 3, 4, -5])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_INT16.pb'),
        type: 'int16',
        input0: Int16Array.from([1, -2, 3, 4, -5]),
        expectedOutput0: Int16Array.from([1, -2, 3, 4, -5])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_INT32.pb'),
        type: 'int32',
        input0: Int32Array.from([1, -2, 3, 4, -5]),
        expectedOutput0: Int32Array.from([1, -2, 3, 4, -5])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_INT64.pb'),
        type: 'int64',
        input0: BigInt64Array.from([BigInt(1), BigInt(-2), BigInt(3), BigInt(4), BigInt(-5)]),
        expectedOutput0: BigInt64Array.from([BigInt(1), BigInt(-2), BigInt(3), BigInt(4), BigInt(-5)])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_STRING.pb'),
        type: 'string',
        input0: ['a', 'b', 'c', 'd', 'e'],
        expectedOutput0: ['a', 'b', 'c', 'd', 'e']
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_UINT8.pb'),
        type: 'uint8',
        input0: Uint8Array.from([1, 2, 3, 4, 5]),
        expectedOutput0: Uint8Array.from([1, 2, 3, 4, 5])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_UINT16.pb'),
        type: 'uint16',
        input0: Uint16Array.from([1, 2, 3, 4, 5]),
        expectedOutput0: Uint16Array.from([1, 2, 3, 4, 5])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_UINT32.pb'),
        type: 'uint32',
        input0: Uint32Array.from([1, 2, 3, 4, 5]),
        expectedOutput0: Uint32Array.from([1, 2, 3, 4, 5])
      },
      {
        model: path.join(TEST_DATA_ROOT, 'test_types_UINT64.pb'),
        type: 'uint64',
        input0: BigUint64Array.from([BigInt(1), BigInt(2), BigInt(3), BigInt(4), BigInt(5)]),
        expectedOutput0: BigUint64Array.from([BigInt(1), BigInt(2), BigInt(3), BigInt(4), BigInt(5)])
      },
    ];

describe('E2E Tests - simple E2E tests', () => {
  MODEL_TEST_TYPES_CASES.forEach(testCase => {
    it(`${testCase.model}`, async () => {
      const session = await InferenceSession.create(testCase.model);
      const output = await session.run({'input': new Tensor(testCase.type, testCase.input0, [1, 5])});
      assert(Object.prototype.hasOwnProperty.call(output, 'output'), '\'output\' should be in the result object.');
      assert(output.output instanceof Tensor, 'result[output] should be a Tensor object.');
      assert.strictEqual(output.output.size, 5, `output size expected 5, got ${output.output.size}.`);
      assert.strictEqual(
          output.output.type, testCase.type, `tensor type expected ${testCase.type}, got ${output.output.type}.`);
      assert.strictEqual(
          Object.getPrototypeOf(output.output.data), Object.getPrototypeOf(testCase.expectedOutput0),
          `tensor data expected ${Object.getPrototypeOf(testCase.expectedOutput0).constructor.name}, got ${
              Object.getPrototypeOf(output.output.data).constructor.name}`);
      assertDataEqual(testCase.type, output.output.data, testCase.expectedOutput0);
    });
  });
});
