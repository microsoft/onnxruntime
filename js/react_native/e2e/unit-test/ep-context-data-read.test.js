// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import EPContextDataReadTest from '../src/EPContextDataReadTest';

jest.mock('onnxruntime-react-native', () => ({ InferenceSession: {}, Tensor: jest.fn() }), { virtual: true });
jest.mock('react-native-fs', () => ({}), { virtual: true });

const createOrtApi = () => ({
  testEpContextDataReadCallback: jest.fn((callback, maxDataSize, name) => {
    const worker = {
      isFinished: false,
      wasAborted: false,
      abort: () => {
        worker.wasAborted = true;
      },
      forceInvalidate: jest.fn(),
    };
    const promise = Promise.resolve().then(() => {
      try {
        const data = callback(name);
        if (!(data instanceof Uint8Array)) {
          throw new TypeError('Callback must return a Uint8Array');
        }
        if (data.byteLength > maxDataSize) {
          throw new RangeError('Callback result exceeds maxDataSize');
        }
        return new Uint8Array(data);
      } finally {
        worker.isFinished = true;
      }
    });
    promise.testWorker = worker;
    return promise;
  }),
});

const createComponent = () => {
  const component = new EPContextDataReadTest({});
  component.setState = (update) => {
    const nextState = typeof update === 'function' ? update(component.state) : update;
    component.state = { ...component.state, ...nextState };
  };
  return component;
};

afterEach(() => {
  delete globalThis.OrtApi;
});

test('callback checks use bindings installed after the component module was evaluated', async () => {
  expect(globalThis.OrtApi).toBeUndefined();
  const component = createComponent();
  globalThis.OrtApi = createOrtApi();

  await component.runCallbackBridgeCheck(11);
  await component.runCallbackBridgeFailureCheck(12);
  await component.runCallbackBridgeCancellationCheck(13);

  expect(component.state.testResults.slice(11)).toEqual([
    expect.objectContaining({ status: 'success' }),
    expect.objectContaining({ status: 'success' }),
    expect.objectContaining({ status: 'success' }),
  ]);
  expect(globalThis.OrtApi.testEpContextDataReadCallback).toHaveBeenCalledTimes(6);
});

test('callback checks resolve the current bindings after they are replaced', async () => {
  const component = createComponent();
  const firstApi = createOrtApi();
  globalThis.OrtApi = firstApi;
  await component.runCallbackBridgeCheck(11);

  const nextApi = createOrtApi();
  globalThis.OrtApi = nextApi;
  await component.runCallbackBridgeCheck(11);

  expect(component.state.testResults[11].status).toBe('success');
  expect(firstApi.testEpContextDataReadCallback).toHaveBeenCalledTimes(1);
  expect(nextApi.testEpContextDataReadCallback).toHaveBeenCalledTimes(1);
});
