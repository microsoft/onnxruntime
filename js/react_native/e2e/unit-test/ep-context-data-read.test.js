// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import EPContextDataReadTest from '../src/EPContextDataReadTest';

jest.mock('onnxruntime-react-native', () => ({ InferenceSession: {}, Tensor: jest.fn() }), { virtual: true });
jest.mock('react-native-fs', () => ({}), { virtual: true });

const createOrtApi = () => ({
  __testEpContextDataReadCallback: jest.fn((callback, maxDataSize, name) => {
    const worker = {
      isFinished: false,
      wasAborted: false,
      envInvalidated: false,
      callbackCalls: 0,
      dispatchQueued: false,
      abort: jest.fn(() => {
        worker.wasAborted = true;
      }),
      invalidateEnv: jest.fn(() => {
        worker.envInvalidated = true;
      }),
      waitForDispatch: jest.fn(() => {
        worker.dispatchQueued = true;
      }),
      forceInvalidate: jest.fn(),
    };
    const promise = Promise.resolve().then(() => {
      try {
        if (worker.wasAborted || worker.envInvalidated) {
          throw new Error('EPContext callback was released before the read');
        }
        worker.callbackCalls++;
        const data = callback(name);
        if (!(data instanceof Uint8Array)) {
          throw new TypeError('Callback must return a Uint8Array');
        }
        if (data.byteLength > maxDataSize) {
          throw new RangeError('Callback result exceeds maxDataSize');
        }
        if (worker.wasAborted) {
          throw new Error('EPContext callback was released during the read');
        }
        return new Uint8Array(data);
      } finally {
        worker.isFinished = true;
      }
    });
    promise.__testWorker = worker;
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
  expect(globalThis.OrtApi.__testEpContextDataReadCallback).toHaveBeenCalledTimes(8);
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
  expect(firstApi.__testEpContextDataReadCallback).toHaveBeenCalledTimes(1);
  expect(nextApi.__testEpContextDataReadCallback).toHaveBeenCalledTimes(1);
});

test('cancellation checks distinguish in-flight, queued, and Env-listener teardown', async () => {
  const component = createComponent();
  globalThis.OrtApi = createOrtApi();

  await component.runCallbackBridgeCancellationCheck(13);

  expect(component.state.testResults[13].status).toBe('success');
  const workers = globalThis.OrtApi.__testEpContextDataReadCallback.mock.results.map(({ value }) => value.__testWorker);
  expect(workers.map((worker) => worker.callbackCalls)).toEqual([1, 0, 0]);
  expect(workers.map((worker) => worker.wasAborted)).toEqual([true, true, false]);
  expect(workers.map((worker) => worker.envInvalidated)).toEqual([false, false, true]);
  expect(workers.every((worker) => worker.isFinished)).toBe(true);
  expect(workers[2].invalidateEnv).toHaveBeenCalledTimes(1);
  expect(workers[2].waitForDispatch).toHaveBeenCalledTimes(1);
  expect(workers[2].waitForDispatch.mock.invocationCallOrder[0]).toBeLessThan(
    workers[2].invalidateEnv.mock.invocationCallOrder[0],
  );
});
