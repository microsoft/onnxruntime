// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {cpus} from 'node:os';
import {Backend, env, InferenceSession, InferenceSessionHandler} from 'onnxruntime-common';

import {initializeWebAssemblyInstance} from './wasm/proxy-wrapper';
import {OnnxruntimeWebAssemblySessionHandler} from './wasm/session-handler-inference';

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

  if (typeof env.wasm.simd !== 'boolean') {
    env.wasm.simd = true;
  }

  if (typeof env.wasm.proxy !== 'boolean') {
    env.wasm.proxy = false;
  }

  if (typeof env.wasm.numThreads !== 'number' || !Number.isInteger(env.wasm.numThreads) || env.wasm.numThreads <= 0) {
    const numCpuLogicalCores = typeof navigator === 'undefined' ? cpus().length : navigator.hardwareConcurrency;
    env.wasm.numThreads = Math.min(4, Math.ceil((numCpuLogicalCores || 1) / 2));
  }
};

export class OnnxruntimeWebAssemblyBackend implements Backend {
  /**
   * This function initializes the WebAssembly backend.
   *
   * This function will be called only once for each backend name. It will be called the first time when
   * `ort.InferenceSession.create()` is called with a registered backend name.
   *
   * @param backendName - the registered backend name.
   */
  async init(backendName: string): Promise<void> {
    // populate wasm flags
    initializeFlags();

    // perform WebGPU availability check
    if (!BUILD_DEFS.DISABLE_WEBGPU && backendName === 'webgpu') {
      if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error('WebGPU is not supported in current environment');
      }

      if (!await navigator.gpu.requestAdapter()) {
        throw new Error(
            'Failed to get GPU adapter. You may need to enable flag "--enable-unsafe-webgpu" if you are using Chrome.');
      }
    }

    // init wasm
    await initializeWebAssemblyInstance();
  }
  createInferenceSessionHandler(path: string, options?: InferenceSession.SessionOptions):
      Promise<InferenceSessionHandler>;
  createInferenceSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<InferenceSessionHandler>;
  async createInferenceSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<InferenceSessionHandler> {
    const handler = new OnnxruntimeWebAssemblySessionHandler();
    await handler.loadModel(pathOrBuffer, options);
    return Promise.resolve(handler);
  }
}
