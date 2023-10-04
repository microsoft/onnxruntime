// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert/strict';
import {Tensor} from 'onnxruntime-common';

import {ALL_NUMERICAL_TYPES, assertIsArrayOf, BIGINT_TYPES, NUMBER_COMPATIBLE_NUMERICAL_TYPES, TensorAny} from '../common.js';

describe('Tensor Constructor Tests - check types', () => {
  for (const [type, typedArrayConstructor, canBeInferredFromType] of ALL_NUMERICAL_TYPES) {
    it(`[${type}] new Tensor(type, typedArray, dims): "tensor.type" should match type passed in`, () => {
      const tensor = new Tensor(type, new typedArrayConstructor(4), [2, 2]);
      assert.equal(tensor.type, type, `tensor.type should be '${type}'`);
    });

    it(`[${type}] new Tensor(type, typedArray, dims): "tensor.data" should be instance of expected typed array`, () => {
      const tensor = new Tensor(type, new typedArrayConstructor(4), [2, 2]);
      assert(
          tensor.data instanceof typedArrayConstructor,
          `tensor.data should be an instance of '${typedArrayConstructor.name}'`);
    });

    if (canBeInferredFromType) {
      it(`[${type}] new Tensor(typedArray, dims): "tensor.type" should match inferred type`, () => {
        const tensor = new Tensor(new typedArrayConstructor(4), [2, 2]);
        assert.equal(tensor.type, type, `tensor.type should be '${type}'`);
      });
    }

    it(`[${type}] new Tensor(type, {}, dims): expect to throw because data is invalid`, () => {
      assert.throws(() => new TensorAny(type, {}, [2, 2]), TypeError);
    });

    it(`[${type}] new Tensor(type, arrayBuffer, dims): expect to throw because data is invalid`, () => {
      assert.throws(() => new TensorAny(type, new ArrayBuffer(100), [2, 2]), TypeError);
    });
  }

  for (const [type, ] of NUMBER_COMPATIBLE_NUMERICAL_TYPES) {
    it(`[${type}] new Tensor(type, numbers, dims): tensor can be constructed from number array`, () => {
      const tensor = new Tensor(type, [1, 2, 3, 4], [2, 2]);
      assert.equal(tensor.type, type, `tensor.type should be '${type}'`);
    });
  }

  for (const [type, ] of BIGINT_TYPES) {
    it(`[${type}] new Tensor(type, numbers, dims): tensor can be constructed from number array`, () => {
      const tensor = new Tensor(type, [1, 2, 3, 4], [2, 2]);
      assert.equal(tensor.type, type, `tensor.type should be '${type}'`);
    });

    it(`[${type}] new Tensor(type, bigints, dims): tensor can be constructed from bigint array`, () => {
      const tensor = new Tensor(type, [1n, 2n, 3n, 4n], [2, 2]);
      assert.equal(tensor.type, type, `tensor.type should be '${type}'`);
    });
  }

  it('[string] new Tensor(\'string\', strings, dims): "tensor.type" should match type passed in', () => {
    const tensor = new Tensor('string', ['a', 'b', 'c', 'd'], [2, 2]);
    assert.equal(tensor.type, 'string', 'tensor.type should be \'string\'');
  });

  it('[string] new Tensor(strings, dims): "tensor.data" should match inferred type', () => {
    const tensor = new Tensor(['a', 'b', 'c', 'd'], [2, 2]);
    assert.equal(tensor.type, 'string', 'tensor.type should be \'string\'');
  });

  it('[string] new Tensor(\'string\', strings, dims): "tensor.data" should be a string array', () => {
    const tensor = new Tensor('string', ['a', 'b', 'c', 'd'], [2, 2]);
    assertIsArrayOf(tensor.data, 'string');
  });

  it('[bool] new Tensor(\'bool\', booleans, dims): "tensor.type" should match type passed in', () => {
    const tensor = new Tensor('bool', [true, false, true, false], [2, 2]);
    assert.equal(tensor.type, 'bool', 'tensor.type should be \'bool\'');
  });

  it('[bool] new Tensor(\'bool\', uint8Array, dims): tensor can be constructed from Uint8Array', () => {
    const tensor = new Tensor('bool', new Uint8Array([1, 0, 1, 0]), [2, 2]);
    assert.equal(tensor.type, 'bool', 'tensor.type should be \'bool\'');
  });

  it('[bool] new Tensor(booleans, dims): "tensor.data" should match inferred type', () => {
    const tensor = new Tensor([true, false, true, false], [2, 2]);
    assert.equal(tensor.type, 'bool', 'tensor.type should be \'bool\'');
  });

  it('[bool] new Tensor(\'bool\', booleans, dims): "tensor.data" should be a boolean array', () => {
    const tensor = new Tensor('bool', [true, false, true, false], [2, 2]);
    assert(tensor.data instanceof Uint8Array, 'tensor.data should be an instance of \'Uint8Array\'');
  });

  it('[float16] new Tensor(\'float16\', numbers, dims): ' +
         'expect to throw because it\'s not allowed to construct \'float16\' tensor from number array',
     () => {
       assert.throws(() => new Tensor('float16', [1, 2, 3, 4], [2, 2]), TypeError);
     });

  it('[badtype] new Tensor(\'a\', numbers, dims): expect to throw because \'a\' is an invalid type', () => {
    assert.throws(() => new TensorAny('a', [1, 2, 3, 4], [2, 2]), TypeError);
  });
});
