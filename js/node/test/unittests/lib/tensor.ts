// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as assert from 'assert';

// tensor with type information
import {Tensor} from 'onnxruntime-common';
import {createTestData, NUMERIC_TYPE_MAP} from '../../test-utils';

// tensor with no type information, used for testing type check
const TensorAny = Tensor as any;

function testAllTensortypes(
    title: string, length: number,
    funcNumerictypes: (passtypeParam: boolean, type: Tensor.Type, data: Tensor.DataType) => void,
    funcStringtype?: (passtypeParam: boolean, data: string[]) => void): void {
  NUMERIC_TYPE_MAP.forEach((ctor, type) => {
    it(`${title} - (${type}, ${ctor.name})`, () => {
      funcNumerictypes(true, type, createTestData(type, length));
    });
    if (type !== 'bool') {
      it(`${title} - (${ctor.name})`, () => {
        funcNumerictypes(false, type, createTestData(type, length));
      });
    }
  });
  if (!funcStringtype) {
    it(`${title} - (string, ${Array.name})`, () => {
      funcNumerictypes(true, 'string', createTestData('string', length));
    });
    it(`${title} - (${Array.name})`, () => {
      funcNumerictypes(false, 'string', createTestData('string', length));
    });
  } else {
    it(`${title} - (string, string[])`, () => {
      funcStringtype(true, createTestData('string', length) as string[]);
    });
    it(`${title} - (string[])`, () => {
      funcStringtype(false, createTestData('string', length) as string[]);
    });
  }
}

describe('UnitTests - tensor', () => {
  testAllTensortypes('check data and type', 100, (passtypeParam, type, data) => {  // numeric and string tensors
    const tensor0 = passtypeParam ? new Tensor(type, data) : new Tensor(data);
    assert.strictEqual(tensor0.data, data, 'tensor.data and data should be the same object.');
    assert.strictEqual(tensor0.type, type, 'tensor.type and type should be equal.');
  });

  testAllTensortypes('check dims (omitted)', 200, (passtypeParam, type, data) => {  // numeric and string tensors
    const tensor0 = passtypeParam ? new Tensor(type, data) : new Tensor(data);
    assert.deepStrictEqual(
        tensor0.dims, [200],
        'tensor.dims should be a number array with exactly 1 item, with value of the array length.');
  });

  testAllTensortypes('check dims (specified)', 60, (passtypeParam, type, data) => {  // numeric and string tensors
    const tensor0 = passtypeParam ? new Tensor(type, data, [3, 4, 5]) : new Tensor(data, [3, 4, 5]);
    assert.deepStrictEqual(tensor0.dims, [3, 4, 5], 'tensor.dims should be a number array with the given 3 items.');
  });

  testAllTensortypes(
      'BAD CALL - invalid dims type', 100, (passtypeParam, type, data) => {  // numeric and string tensors
        assert.throws(() => {
          const badDims = {};
          passtypeParam ? new TensorAny(type, data, badDims) : new TensorAny(data, badDims);
        }, {name: 'TypeError', message: /must be a number array/});
      });
  testAllTensortypes(
      'BAD CALL - invalid dims element type', 100, (passtypeParam, type, data) => {  // numeric and string tensors
        assert.throws(() => {
          const badDims = [1, 2, ''];
          passtypeParam ? new TensorAny(type, data, badDims) : new TensorAny(data, badDims);
        }, {name: 'TypeError', message: /must be an integer/});
      });
  testAllTensortypes(
      'BAD CALL - invalid dims number type (negative)', 100,
      (passtypeParam, type, data) => {  // numeric and string tensors
        assert.throws(() => {
          const badDims = [1, 2, -1];
          passtypeParam ? new TensorAny(type, data, badDims) : new TensorAny(data, badDims);
        }, {name: 'RangeError', message: /must be a non-negative integer/});
      });
  testAllTensortypes(
      'BAD CALL - invalid dims number type (non-integer)', 100,
      (passtypeParam, type, data) => {  // numeric and string tensors
        assert.throws(() => {
          const badDims = [1, 2, 1.5];
          passtypeParam ? new TensorAny(type, data, badDims) : new TensorAny(data, badDims);
        }, {name: 'TypeError', message: /must be an integer/});
      });

  testAllTensortypes(
      'BAD CALL - length and dims does not match', 100, (passtypeParam, type, data) => {  // numeric and string tensors
        assert.throws(() => {
          const badDims = [10, 8];
          passtypeParam ? new TensorAny(type, data, badDims) : new TensorAny(data, badDims);
        }, {name: 'Error', message: /does not match data length/});
      });
});
