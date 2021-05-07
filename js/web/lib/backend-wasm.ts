// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Backend, env, InferenceSession, SessionHandler} from 'onnxruntime-common';

import {init, OnnxruntimeWebAssemblySessionHandler} from './wasm';

class OnnxruntimeWebAssemblyBackend implements Backend {
  async init(): Promise<void> {
    await init();
  }
  createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  async createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    let buffer: Uint8Array;
    if (typeof pathOrBuffer === 'string') {
      const response = await fetch(pathOrBuffer);
      const arrayBuffer = await response.arrayBuffer();
      buffer = new Uint8Array(arrayBuffer);
    } else {
      buffer = pathOrBuffer;
    }

    if (flags.worker !== undefined) {
      if (options !== undefined) {
        options.intraOpNumThreads = flags.worker;
      } else {
        options = { intraOpNumThreads: flags.worker }
      }
    }

    const handler = new OnnxruntimeWebAssemblySessionHandler();
    handler.loadModel(buffer, options);
    return Promise.resolve(handler);
  }
}

export const wasmBackend = new OnnxruntimeWebAssemblyBackend();

export interface WebAssemblyFlags {
  /**
   * set or get number of worker(s)
   *
   * This setting is available only when WebAssembly multithread feature is available in current context.
   */
  worker?: number;

  /**
   * set or get a number specifying the timeout for initialization of WebAssembly backend, in milliseconds.
   */
  initTimeout?: number;
}

/**
 * Represent a set of flags for WebAssembly backend.
 */
export const flags: WebAssemblyFlags = env.wasm = env.wasm as WebAssemblyFlags || {};
