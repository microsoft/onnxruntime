// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import { expect } from 'chai';
import * as ort from 'onnxruntime-common';
// A tiny ABS model with static shape [2,4]
const ONNX_MODEL_TEST_ABS_STATIC = Uint8Array.from([
  8, 9, 58, 83, 10, 31, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 8, 111, 117, 116, 112, 117, 116, 95, 48, 26, 3, 65,
  98, 115, 34, 3, 65, 98, 115, 58, 0, 18, 3, 97, 98, 115, 90, 25, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 14, 10,
  12, 8, 1, 18, 8, 10, 2, 8, 2, 10, 2, 8, 4, 98, 16, 10, 8, 111, 117, 116, 112, 117, 116, 95, 48, 18, 4, 10, 2, 8, 1,
  66, 4, 10, 0, 16, 21,
]);
describe('#UnitTest# - wasm - preallocated CPU output', () => {
  it('fills pre-allocated CPU tensor with results', async () => {
    const session = await ort.InferenceSession.create(ONNX_MODEL_TEST_ABS_STATIC);
    const dims = [2, 4];
    const inputData = new Float32Array([-1, 2, -3, 4, -5, 6, -7, 8]);
    const feeds: Record<string, ort.Tensor> = {
      input_0: new ort.Tensor('float32', inputData, dims),
    } as unknown as Record<string, ort.Tensor>;
    const prealloc = new ort.Tensor('float32', new Float32Array(inputData.length), dims);
    const results = await session.run(feeds, { output_0: prealloc });
    // Should return the exact same Tensor instance
    expect(results.output_0).to.equal(prealloc);
    const expected = Array.from(inputData, (v) => Math.abs(v));
    const got = Array.from(prealloc.data as Float32Array);
    expect(got).to.deep.equal(expected);
  });
});
