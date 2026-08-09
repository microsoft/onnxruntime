// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert/strict';
import { Tensor } from 'onnxruntime-common';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const globalF16 = (globalThis as any).Float16Array;

(globalF16 ? describe : describe.skip)('Tensor Constructor Tests - check type float16 (Float16Array available)', () => {
  it("[float16] new Tensor('float16', numbers, dims): allow number array when Float16Array is available", () => {
    const tensor = new Tensor('float16', [1, 2, 3, 4], [2, 2]);
    assert.equal(tensor.type, 'float16', "tensor.type should be 'float16'");
    assert(tensor.data instanceof globalF16, "tensor.data should be an instance of 'Float16Array'");
    assert.equal(tensor.data[0], 1, 'tensor.data[0] should be 1');
    assert.equal(tensor.data[1], 2, 'tensor.data[1] should be 2');
    assert.equal(tensor.data[2], 3, 'tensor.data[2] should be 3');
    assert.equal(tensor.data[3], 4, 'tensor.data[3] should be 4');
    assert.equal(tensor.data.length, 4, 'tensor.data.length should be 4');
  });

  it("[float16] new Tensor('float16', float16array, dims): allow Float16Array when Float16Array is available", () => {
    const tensor = new Tensor('float16', new globalF16([1, 2, 3, 4]), [2, 2]);
    assert.equal(tensor.type, 'float16', "tensor.type should be 'float16'");
    assert(tensor.data instanceof globalF16, "tensor.data should be an instance of 'Float16Array'");
    assert.equal(tensor.data[0], 1, 'tensor.data[0] should be 1');
    assert.equal(tensor.data[1], 2, 'tensor.data[1] should be 2');
    assert.equal(tensor.data[2], 3, 'tensor.data[2] should be 3');
    assert.equal(tensor.data[3], 4, 'tensor.data[3] should be 4');
    assert.equal(tensor.data.length, 4, 'tensor.data.length should be 4');
  });

  it("[float16] new Tensor('float16', uint16array, dims): allow Uint16Array when Float16Array is available", () => {
    const tensor = new Tensor('float16', new Uint16Array([15360, 16384, 16896, 17408]), [2, 2]);
    assert.equal(tensor.type, 'float16', "tensor.type should be 'float16'");
    assert(tensor.data instanceof globalF16, "tensor.data should be an instance of 'Float16Array'");
    assert.equal(tensor.data[0], 1, 'tensor.data[0] should be 1');
    assert.equal(tensor.data[1], 2, 'tensor.data[1] should be 2');
    assert.equal(tensor.data[2], 3, 'tensor.data[2] should be 3');
    assert.equal(tensor.data[3], 4, 'tensor.data[3] should be 4');
    assert.equal(tensor.data.length, 4, 'tensor.data.length should be 4');
  });
});

(globalF16 ? describe.skip : describe)(
  'Tensor Constructor Tests - check type float16 (Float16Array not available)',
  () => {
    it(
      "[float16] new Tensor('float16', numbers, dims): " +
        "expect to throw because it's not allowed to construct 'float16' tensor from number array",
      () => {
        assert.throws(() => new Tensor('float16', [1, 2, 3, 4], [2, 2]), TypeError);
      },
    );

    it("[float16] new Tensor('float16', uint16array, dims): allow Uint16Array", () => {
      const tensor = new Tensor('float16', new Uint16Array([15360, 16384, 16896, 17408]), [2, 2]);
      assert.equal(tensor.type, 'float16', "tensor.type should be 'float16'");
      assert(tensor.data instanceof Uint16Array, "tensor.data should be an instance of 'Uint16Array'");
    });
  },
);
