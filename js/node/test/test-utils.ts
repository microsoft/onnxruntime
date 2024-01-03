// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import * as fs from 'fs-extra';
import {jsonc} from 'jsonc';
import {InferenceSession, Tensor} from 'onnxruntime-common';
import * as path from 'path';

import * as onnx_proto from './ort-schema/protobuf/onnx';

export const TEST_ROOT = __dirname;
export const TEST_DATA_ROOT = path.join(TEST_ROOT, 'testdata');

export const ORT_ROOT = path.join(__dirname, '../../..');
export const NODE_TESTS_ROOT = path.join(ORT_ROOT, 'js/test/data/node');

export const SQUEEZENET_INPUT0_DATA: number[] = require(path.join(TEST_DATA_ROOT, 'squeezenet.input0.json'));
export const SQUEEZENET_OUTPUT0_DATA: number[] = require(path.join(TEST_DATA_ROOT, 'squeezenet.output0.json'));

const BACKEND_TEST_SERIES_FILTERS: {[name: string]: Array<string|[string, string]>} =
    jsonc.readSync(path.join(ORT_ROOT, 'onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc'));

const OVERRIDES: {
  atol_default: number; rtol_default: number; atol_overrides: {[name: string]: number};
  rtol_overrides: {[name: string]: number};
} = jsonc.readSync(path.join(ORT_ROOT, 'onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc'));

const ATOL_DEFAULT = OVERRIDES.atol_default;
const RTOL_DEFAULT = OVERRIDES.rtol_default;

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

// a simple function to create a tensor data for test
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

// a simple function to create a tensor for test
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

// call the addon directly to make sure DLL is loaded
export function warmup(): void {
  describe('Warmup', async function() {
    // eslint-disable-next-line no-invalid-this
    this.timeout(0);
    // we have test cases to verify correctness in other place, so do no check here.
    try {
      const session = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_int32.onnx'));
      await session.run({input: new Tensor(new Float32Array(5), [1, 5])}, {output: null}, {});
    } catch (e) {
    }
  });
}

export function assertFloatEqual(
    actual: number[]|Float32Array|Float64Array, expected: number[]|Float32Array|Float64Array, atol?: number,
    rtol?: number): void {
  const absolute_tol: number = atol ?? 1.0e-4;
  const relative_tol: number = 1 + (rtol ?? 1.0e-6);

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
      // one is NaN and the other is not
      assert.fail(`actual[${i}]=${a}, expected[${i}]=${b}`);
    }

    // Comparing 2 float numbers: (Suppose a >= b)
    //
    // if ( a - b < ABSOLUTE_ERROR || 1.0 < a / b < RELATIVE_ERROR)
    //   test pass
    // else
    //   test fail
    // endif
    //
    if (Math.abs(a - b) < absolute_tol) {
      continue;  // absolute error check pass
    }
    if (a !== 0 && b !== 0 && a * b > 0 && a / b < relative_tol && b / a < relative_tol) {
      continue;  // relative error check pass
    }

    // if code goes here, it means both (abs/rel) check failed.
    assert.fail(`actual[${i}]=${a}, expected[${i}]=${b}`);
  }
}

export function assertDataEqual(
    type: Tensor.Type, actual: Tensor.DataType, expected: Tensor.DataType, atol?: number, rtol?: number): void {
  switch (type) {
    case 'float32':
    case 'float64':
      assertFloatEqual(
          actual as number[] | Float32Array | Float64Array, expected as number[] | Float32Array | Float64Array, atol,
          rtol);
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

// This function check whether 2 tensors should be considered as 'match' or not
export function assertTensorEqual(actual: Tensor, expected: Tensor, atol?: number, rtol?: number): void {
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

  assertDataEqual(actualType, actual.data, expected.data, atol, rtol);
}

export function loadTensorFromFile(pbFile: string): Tensor {
  const tensorProto = onnx_proto.onnx.TensorProto.decode(fs.readFileSync(pbFile));
  let transferredTypedArray: Tensor.DataType;
  let type: Tensor.Type;
  const dims = tensorProto.dims.map((dim) => typeof dim === 'number' ? dim : dim.toNumber());


  if (tensorProto.dataType === 8) {  // string
    return new Tensor('string', tensorProto.stringData.map(i => i.toString()), dims);
  } else {
    switch (tensorProto.dataType) {
      //     FLOAT = 1,
      //     UINT8 = 2,
      //     INT8 = 3,
      //     UINT16 = 4,
      //     INT16 = 5,
      //     INT32 = 6,
      //     INT64 = 7,
      //     STRING = 8,
      //     BOOL = 9,
      //     FLOAT16 = 10,
      //     DOUBLE = 11,
      //     UINT32 = 12,
      //     UINT64 = 13,
      case onnx_proto.onnx.TensorProto.DataType.FLOAT:
        transferredTypedArray = new Float32Array(tensorProto.rawData.byteLength / 4);
        type = 'float32';
        break;
      case onnx_proto.onnx.TensorProto.DataType.UINT8:
        transferredTypedArray = new Uint8Array(tensorProto.rawData.byteLength);
        type = 'uint8';
        break;
      case onnx_proto.onnx.TensorProto.DataType.INT8:
        transferredTypedArray = new Int8Array(tensorProto.rawData.byteLength);
        type = 'int8';
        break;
      case onnx_proto.onnx.TensorProto.DataType.UINT16:
        transferredTypedArray = new Uint16Array(tensorProto.rawData.byteLength / 2);
        type = 'uint16';
        break;
      case onnx_proto.onnx.TensorProto.DataType.INT16:
        transferredTypedArray = new Int16Array(tensorProto.rawData.byteLength / 2);
        type = 'int16';
        break;
      case onnx_proto.onnx.TensorProto.DataType.INT32:
        transferredTypedArray = new Int32Array(tensorProto.rawData.byteLength / 4);
        type = 'int32';
        break;
      case onnx_proto.onnx.TensorProto.DataType.INT64:
        transferredTypedArray = new BigInt64Array(tensorProto.rawData.byteLength / 8);
        type = 'int64';
        break;
      case onnx_proto.onnx.TensorProto.DataType.BOOL:
        transferredTypedArray = new Uint8Array(tensorProto.rawData.byteLength);
        type = 'bool';
        break;
      case onnx_proto.onnx.TensorProto.DataType.DOUBLE:
        transferredTypedArray = new Float64Array(tensorProto.rawData.byteLength / 8);
        type = 'float64';
        break;
      case onnx_proto.onnx.TensorProto.DataType.UINT32:
        transferredTypedArray = new Uint32Array(tensorProto.rawData.byteLength / 4);
        type = 'uint32';
        break;
      case onnx_proto.onnx.TensorProto.DataType.UINT64:
        transferredTypedArray = new BigUint64Array(tensorProto.rawData.byteLength / 8);
        type = 'uint64';
        break;
      default:
        throw new Error(`not supported tensor type: ${tensorProto.dataType}`);
    }
    const transferredTypedArrayRawDataView =
        new Uint8Array(transferredTypedArray.buffer, transferredTypedArray.byteOffset, tensorProto.rawData.byteLength);
    transferredTypedArrayRawDataView.set(tensorProto.rawData);

    return new Tensor(type, transferredTypedArray, dims);
  }
}

function loadFiltersRegex(): Array<{opset?: RegExp | undefined; name: RegExp}> {
  const filters: Array<string|[string, string]> = ['(FLOAT16)'];
  filters.push(...BACKEND_TEST_SERIES_FILTERS.current_failing_tests);

  if (process.arch === 'ia32') {
    filters.push(...BACKEND_TEST_SERIES_FILTERS.current_failing_tests_x86);
  }

  filters.push(...BACKEND_TEST_SERIES_FILTERS.tests_with_pre_opset7_dependencies);
  filters.push(...BACKEND_TEST_SERIES_FILTERS.unsupported_usages);
  filters.push(...BACKEND_TEST_SERIES_FILTERS.failing_permanently);
  filters.push(...BACKEND_TEST_SERIES_FILTERS.test_with_types_disabled_due_to_binary_size_concerns);

  filters.push(...BACKEND_TEST_SERIES_FILTERS.failing_permanently_nodejs_binding);

  return filters.map(
      filter => typeof filter === 'string' ? {name: new RegExp(filter)} :
                                             {opset: new RegExp(filter[0]), name: new RegExp(filter[1])});
}

const BACKEND_TEST_SERIES_FILTERS_REGEX = loadFiltersRegex();

export function shouldSkipModel(model: string, opset: string, eps: string[]): boolean {
  for (const regex of BACKEND_TEST_SERIES_FILTERS_REGEX) {
    if (regex.opset) {
      if (!regex.opset.test(opset)) {
        continue;
      }
    }
    for (const ep of eps) {
      if (regex.name.test(`${model}_${ep}`)) {
        return true;
      }
    }
  }

  return false;
}

export function atol(model: string): number {
  return OVERRIDES.atol_overrides[model] ?? ATOL_DEFAULT;
}

export function rtol(model: string): number {
  return OVERRIDES.rtol_overrides[model] ?? RTOL_DEFAULT;
}
