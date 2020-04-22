import assert from 'assert';
import * as path from 'path';

import {Tensor} from '../lib/tensor';

export const TEST_ROOT = __dirname;
export const TEST_DATA_ROOT = path.join(TEST_ROOT, 'testdata');

export const SQUEEZENET_INPUT0_DATA: number[] = require(path.join(TEST_DATA_ROOT, 'squeezenet.input0.json'));
export const SQUEEZENET_OUTPUT0_DATA: number[] = require(path.join(TEST_DATA_ROOT, 'squeezenet.output0.json'));


export const NUMERIC_TYPE_MAP = new Map<Tensor.Type, new (len: number) => Tensor.DataType>([
  ['float32', Float32Array],
  ['bool', Uint8Array],
  ['uint8', Uint8Array],
  ['int8', Int8Array],
  ['uint16', Uint16Array],
  ['int16', Int16Array],
  ['int32', Int32Array],
  ['int64', BigInt64Array],
  ['bool', Uint8Array],
  ['float64', Float64Array],
  ['uint32', Uint32Array],
  ['uint64', BigUint64Array],
]);

export function createTestData(type: Tensor.Type, length: number): Tensor.DataType {
  let data: Tensor.DataType;
  if (type === 'string') {
    data = new Array<string>(length);
    for (let i = 0; i < length; i++) {
      data[i] = `str${i}`;
    }
  } else {
    data = new (NUMERIC_TYPE_MAP.get(type)!)(length);
    for (let i = 0; i < length; i++) {
      data[i] = (type === 'uint64' || type === 'int64') ? BigInt(i) : i;
    }
  }
  return data;
}

export function createTestTensor(type: Tensor.Type, lengthOrDims?: number|number[]): Tensor {
  let length = 100;
  let dims = [100];
  if (typeof lengthOrDims === 'number') {
    length = lengthOrDims;
    dims = [length];
  } else if (Array.isArray(lengthOrDims)) {
    dims = lengthOrDims;
    length = dims.reduce((a, b) => a * b, 1);
  }

  return new Tensor(type, createTestData(type, length), dims);
}


// This function check whether 2 tensors should be considered as 'match' or not
export function assertTensorEqual(actual: Tensor, expected: Tensor) {
  assert(typeof actual === 'object');
  assert(typeof expected === 'object');

  assert(Array.isArray(actual.dims));
  assert(Array.isArray(expected.dims));

  const actualDims = actual.dims;
  const actualType = actual.type;
  const expectedDims = expected.dims;
  const expectedType = expected.type;

  assert.strictEqual(actualType, expectedType);
  assert.deepStrictEqual(actualDims, expectedDims);

  assertDataEqual(actualType, actual.data, expected.data);
}

export function assertDataEqual(type: Tensor.Type, actual: Tensor.DataType, expected: Tensor.DataType) {
  switch (type) {
    case 'float32':
    case 'float64':
      assertFloatEqual(
          actual as number[] | Float32Array | Float64Array, expected as number[] | Float32Array | Float64Array);
      break;

    case 'uint8':
    case 'int8':
    case 'uint16':
    case 'int16':
    case 'uint32':
    case 'int32':
    case 'uint64':
    case 'int64':
    case 'bool':
    case 'string':
      assert.deepStrictEqual(actual, expected);
      break;

    default:
      throw new Error('type not implemented or not supported');
  }
}

export function assertFloatEqual(
    actual: number[]|Float32Array|Float64Array, expected: number[]|Float32Array|Float64Array) {
  const THRESHOLD_ABSOLUTE_ERROR = 1.0e-4;
  const THRESHOLD_RELATIVE_ERROR = 1.000001;

  assert.strictEqual(actual.length, expected.length);

  for (let i = actual.length - 1; i >= 0; i--) {
    const a = actual[i], b = expected[i];

    if (a === b) {
      continue;
    }

    // check for NaN
    //
    if (Number.isNaN(a) && Number.isNaN(b)) {
      continue;  // 2 numbers are NaN, treat as equal
    }
    if (Number.isNaN(a) || Number.isNaN(b)) {
      return false;  // one is NaN and the other is not
    }

    // Comparing 2 float numbers: (Suppose a >= b)
    //
    // if ( a - b < ABSOLUTE_ERROR || 1.0 < a / b < RELATIVE_ERROR)
    //   test pass
    // else
    //   test fail
    // endif
    //
    if (Math.abs(a - b) < THRESHOLD_ABSOLUTE_ERROR) {
      continue;  // absolute error check pass
    }
    if (a !== 0 && b !== 0 && a * b > 0 && a / b < THRESHOLD_RELATIVE_ERROR && b / a < THRESHOLD_RELATIVE_ERROR) {
      continue;  // relative error check pass
    }

    // if code goes here, it means both (abs/rel) check failed.
    assert.fail(`actual[${i}]=${a}, expected[${i}]=${b}`);
  }

  return true;
}
