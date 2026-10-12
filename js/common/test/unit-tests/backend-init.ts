// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert/strict';
import { Backend, InferenceSession, InferenceSessionHandler, LoraAdapter, registerBackend } from 'onnxruntime-common';

const notImplemented = (): never => {
  throw new Error('not implemented.');
};

/**
 * A backend that fails when init() is called again before the previous call finishes, like the WebAssembly backend.
 */
class MockBackend implements Backend {
  initializedNames: string[] = [];
  private initializing = false;

  async init(backendName: string): Promise<void> {
    if (this.initializing) {
      throw new Error('init() is called concurrently.');
    }
    this.initializing = true;
    await new Promise((resolve) => setTimeout(resolve, 10));
    this.initializing = false;
    this.initializedNames.push(backendName);
  }

  async createInferenceSessionHandler(): Promise<InferenceSessionHandler> {
    return {
      inputNames: [],
      outputNames: [],
      inputMetadata: [],
      outputMetadata: [],
      dispose: notImplemented,
      startProfiling: notImplemented,
      endProfiling: notImplemented,
      run: notImplemented,
    };
  }

  async createLoraAdapterHandler() {
    return { dispose: notImplemented };
  }
}

describe('Backend - initialization', () => {
  // LoraAdapter.create() uses the registered name with the highest priority, and the session uses the other name. Each
  // test uses new names with higher priorities, so that the names are not initialized yet.
  const registerMockBackend = (adapterName: string, sessionName: string, priority: number): MockBackend => {
    const backend = new MockBackend();
    registerBackend(adapterName, backend, priority + 1);
    registerBackend(sessionName, backend, priority);
    return backend;
  };

  it('create LoRA adapter and session with another name of the same backend concurrently', async () => {
    const backend = registerMockBackend('test-concurrent-1-a', 'test-concurrent-1-b', 1000);
    await Promise.all([
      LoraAdapter.create(new Uint8Array(1)),
      InferenceSession.create(new Uint8Array(1), { executionProviders: ['test-concurrent-1-b'] }),
    ]);
    assert.deepEqual(backend.initializedNames, ['test-concurrent-1-a', 'test-concurrent-1-b']);
  });

  it('create session and LoRA adapter with another name of the same backend concurrently', async () => {
    const backend = registerMockBackend('test-concurrent-2-a', 'test-concurrent-2-b', 2000);
    await Promise.all([
      InferenceSession.create(new Uint8Array(1), { executionProviders: ['test-concurrent-2-b'] }),
      LoraAdapter.create(new Uint8Array(1)),
    ]);
    assert.deepEqual(backend.initializedNames, ['test-concurrent-2-b', 'test-concurrent-2-a']);
    // both names are usable afterwards
    await InferenceSession.create(new Uint8Array(1), { executionProviders: ['test-concurrent-2-a'] });
    await InferenceSession.create(new Uint8Array(1), { executionProviders: ['test-concurrent-2-b'] });
  });
});
