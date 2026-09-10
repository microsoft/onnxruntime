// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { InferenceSession } from 'onnxruntime-common';

import { getInstance } from '../../../../lib/wasm/wasm-factory';

const ONNX_MODEL_TEST_ABS_STATIC = Uint8Array.from([
  8, 9, 58, 83, 10, 31, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 8, 111, 117, 116, 112, 117, 116, 95, 48, 26, 3, 65,
  98, 115, 34, 3, 65, 98, 115, 58, 0, 18, 3, 97, 98, 115, 90, 25, 10, 7, 105, 110, 112, 117, 116, 95, 48, 18, 14, 10,
  12, 8, 1, 18, 8, 10, 2, 8, 2, 10, 2, 8, 4, 98, 16, 10, 8, 111, 117, 116, 112, 117, 116, 95, 48, 18, 4, 10, 2, 8, 1,
  66, 4, 10, 0, 16, 21,
]);

// Calls the exports directly rather than through a higher level API, because what needs covering is that the symbols
// survive into the emitted module and are callable with this signature.
const getAvailableProviders = (): string[] => {
  const wasm = getInstance();
  const ptrSize = wasm.PTR_SIZE;

  const stack = wasm.stackSave();
  try {
    const providersOffset = wasm.stackAlloc(ptrSize);
    const lengthOffset = wasm.stackAlloc(4);

    expect(wasm._OrtGetAvailableProviders(providersOffset, lengthOffset)).to.equal(0);

    const buffer = Number(wasm.getValue(providersOffset, '*'));
    const length = Number(wasm.getValue(lengthOffset, 'i32'));
    expect(buffer).to.not.equal(0);

    const providers: string[] = [];
    for (let i = 0; i < length; i++) {
      providers.push(wasm.UTF8ToString(Number(wasm.getValue(buffer + i * ptrSize, '*'))));
    }

    expect(wasm._OrtReleaseAvailableProviders(buffer, length)).to.equal(0);
    return providers;
  } finally {
    wasm.stackRestore(stack);
  }
};

const testAvailableProviders = async (model: Uint8Array, expectedProviders: string[]) => {
  // Loading a model first, so that the WebAssembly module is initialized before the exports are called.
  await InferenceSession.create(model);

  const providers = getAvailableProviders();
  expect(providers).to.include.members(expectedProviders);
  providers.forEach((provider) => expect(provider.length).to.be.greaterThan(0));
};

describe('#UnitTest# - wasm - test available providers', () => {
  it('providers the build was compiled with', async () => {
    await testAvailableProviders(ONNX_MODEL_TEST_ABS_STATIC, ['CPUExecutionProvider']);
  });
});
