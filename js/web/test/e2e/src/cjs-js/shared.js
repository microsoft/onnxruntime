// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// Model data for "test_abs/model.onnx"
const testModelData =
  'CAcSDGJhY2tlbmQtdGVzdDpJCgsKAXgSAXkiA0FicxIIdGVzdF9hYnNaFwoBeBISChAIARIMCgIIAwoCCAQKAggFYhcKAXkSEgoQCAESDAoCCAMKAggECgIIBUIECgAQDQ==';

const base64StringToUint8Array = (base64String) => {
  const charArray = atob(base64String);
  const length = charArray.length;
  const buffer = new Uint8Array(new ArrayBuffer(length));
  for (let i = 0; i < length; i++) {
    buffer[i] = charArray.charCodeAt(i);
  }
  return buffer;
};

const assert = (cond) => {
  if (!cond) throw new Error();
};

const setupMultipleThreads = (ort) => {
  ort.env.wasm.numThreads = 2;
  assert(typeof SharedArrayBuffer !== 'undefined');
};

const testInferenceAndValidate = async (ort, options) => {
  const model = base64StringToUint8Array(testModelData);
  const session = await ort.InferenceSession.create(model, options);

  // test data: [0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, ... 58, -59]
  const inputData = [...Array(60).keys()].map((i) => (i % 2 === 0 ? i : -i));
  const expectedOutputData = inputData.map((i) => Math.abs(i));

  const fetches = await session.run({ x: new ort.Tensor('float32', inputData, [3, 4, 5]) });

  const y = fetches.y;

  assert(y instanceof ort.Tensor);
  assert(y.dims.length === 3 && y.dims[0] === 3 && y.dims[1] === 4 && y.dims[2] === 5);

  for (let i = 0; i < expectedOutputData.length; i++) {
    assert(y.data[i] === expectedOutputData[i]);
  }
};

module.exports = {
  setupMultipleThreads,
  testInferenceAndValidate,
};
