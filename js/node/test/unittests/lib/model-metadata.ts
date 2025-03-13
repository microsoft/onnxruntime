// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as assert from 'assert';
import { InferenceSession } from 'onnxruntime-common';

const ONNX_MODEL_TEST_ABS_NO_SHAPE = Uint8Array.from([
  8, 9, 58, 73, 10, 31, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 8, 111, 117, 116, 112, 117, 116, 95, 48, 26, 3, 65,
  98, 115, 34, 3, 65, 98, 115, 58, 0, 18, 3, 97, 98, 115, 90, 15, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 4, 10, 2,
  8, 1, 98, 16, 10, 8, 111, 117, 116, 112, 117, 116, 95, 48, 18, 4, 10, 2, 8, 1, 66, 4, 10, 0, 16, 21,
]);

const ONNX_MODEL_TEST_ABS_SYMBOL = Uint8Array.from([
  8, 9, 58, 105, 10, 31, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 8, 111, 117, 116, 112, 117, 116, 95, 48, 26, 3, 65,
  98, 115, 34, 3, 65, 98, 115, 58, 0, 18, 3, 97, 98, 115, 90, 47, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 36, 10,
  34, 8, 1, 18, 30, 10, 13, 18, 11, 95, 105, 110, 112, 117, 116, 95, 48, 95, 100, 48, 10, 13, 18, 11, 95, 105, 110, 112,
  117, 116, 95, 48, 95, 100, 49, 98, 16, 10, 8, 111, 117, 116, 112, 117, 116, 95, 48, 18, 4, 10, 2, 8, 1, 66, 4, 10, 0,
  16, 21,
]);

const ONNX_MODEL_TEST_ABS_STATIC = Uint8Array.from([
  8, 9, 58, 83, 10, 31, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 8, 111, 117, 116, 112, 117, 116, 95, 48, 26, 3, 65,
  98, 115, 34, 3, 65, 98, 115, 58, 0, 18, 3, 97, 98, 115, 90, 25, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 14, 10,
  12, 8, 1, 18, 8, 10, 2, 8, 2, 10, 2, 8, 4, 98, 16, 10, 8, 111, 117, 116, 112, 117, 116, 95, 48, 18, 4, 10, 2, 8, 1,
  66, 4, 10, 0, 16, 21,
]);

const testModelMetadata = async (
  model: Uint8Array,
  expectedInputNames: string[],
  expectedOutputNames: string[],
  expectedInputMetadata: InferenceSession.ValueMetadata[],
  expectedOutputMetadata: InferenceSession.ValueMetadata[],
) => {
  const session = await InferenceSession.create(model);
  assert.deepStrictEqual(session.inputNames, expectedInputNames);
  assert.deepStrictEqual(session.outputNames, expectedOutputNames);
  assert.deepStrictEqual(session.inputMetadata, expectedInputMetadata);
  assert.deepStrictEqual(session.outputMetadata, expectedOutputMetadata);
};

describe('#UnitTest# - test model input/output metadata', () => {
  it('model input/output with no shape', async () => {
    await testModelMetadata(
      ONNX_MODEL_TEST_ABS_NO_SHAPE,
      ['input_0'],
      ['output_0'],
      [{ name: 'input_0', isTensor: true, type: 'float32', shape: [] }],
      [{ name: 'output_0', isTensor: true, type: 'float32', shape: [] }],
    );
  });

  it('model input/output with symbol shape', async () => {
    await testModelMetadata(
      ONNX_MODEL_TEST_ABS_SYMBOL,
      ['input_0'],
      ['output_0'],
      [
        {
          name: 'input_0',
          isTensor: true,
          type: 'float32',
          shape: ['_input_0_d0', '_input_0_d1'],
        },
      ],
      [
        {
          name: 'output_0',
          isTensor: true,
          type: 'float32',
          shape: ['_input_0_d0', '_input_0_d1'],
        },
      ],
    );
  });

  it('model input/output with static shape', async () => {
    await testModelMetadata(
      ONNX_MODEL_TEST_ABS_STATIC,
      ['input_0'],
      ['output_0'],
      [{ name: 'input_0', isTensor: true, type: 'float32', shape: [2, 4] }],
      [{ name: 'output_0', isTensor: true, type: 'float32', shape: [2, 4] }],
    );
  });
});
