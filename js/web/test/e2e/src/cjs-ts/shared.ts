// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Model data for "test_abs/model.onnx"
const testModelData =
    // eslint-disable-next-line max-len
    'CAcSDGJhY2tlbmQtdGVzdDpJCgsKAXgSAXkiA0FicxIIdGVzdF9hYnNaFwoBeBISChAIARIMCgIIAwoCCAQKAggFYhcKAXkSEgoQCAESDAoCCAMKAggECgIIBUIECgAQDQ==';

const base64StringToUint8Array = (base64String: string): Uint8Array => {
  const charArray = atob(base64String);
  const length = charArray.length;
  const buffer = new Uint8Array(new ArrayBuffer(length));
  for (let i = 0; i < length; i++) {
    buffer[i] = charArray.charCodeAt(i);
  }
  return buffer;
};

module.exports = {
  testModelData,
  base64StringToUint8Array,
};
