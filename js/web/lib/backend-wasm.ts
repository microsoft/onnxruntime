// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Backend, env, InferenceSession, SessionHandler} from 'onnxruntime-common';

import {OnnxruntimeWebAssemblySessionHandler} from './wasm/session-handler';
import {initializeWebAssembly} from './wasm/wasm-factory';

/**
 * This function initializes all flags for WebAssembly.
 *
 * Those flags are accessible from `ort.env.wasm`. Users are allow to set those flags before the first inference session
 * being created, to override default value.
 */
export const initializeFlags = (): void => {
  if (typeof env.wasm.initTimeout !== 'number' || env.wasm.initTimeout < 0) {
    env.wasm.initTimeout = 0;
  }

  if (typeof env.wasm.numThreads !== 'number' || !Number.isInteger(env.wasm.numThreads) || env.wasm.numThreads < 0) {
    env.wasm.numThreads = Math.ceil((navigator.hardwareConcurrency || 1) / 2);
  }
  env.wasm.numThreads = Math.min(4, env.wasm.numThreads);
};

class OnnxruntimeWebAssemblyBackend implements Backend {
  async init(): Promise<void> {
    // populate wasm flags
    initializeFlags();

    // init wasm
    await initializeWebAssembly();
  }
  createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  async createSessionHandler(pathOrBuffer: string|Uint8Array, _options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    let buffer: Uint8Array;
    if (typeof pathOrBuffer === 'string') {
      const response = await fetch(pathOrBuffer);
      const arrayBuffer = await response.arrayBuffer();
      buffer = new Uint8Array(arrayBuffer);
    } else {
      buffer = pathOrBuffer;
    }
    const handler = new OnnxruntimeWebAssemblySessionHandler();
    // TODO: support SessionOptions
    handler.loadModel(buffer);
    return Promise.resolve(handler);
  }
}

export const wasmBackend = new OnnxruntimeWebAssemblyBackend();
